# scripts/sympto.py
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import os
import joblib
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# 0. Project-root helpers
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # <repo>/
sys.path.append(str(PROJECT_ROOT))                         # allow imports from root

from models.chat_model import add_message, get_chat, create_chat

# ----------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------
load_dotenv()
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "medical_conditions_dataset.csv"
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 500
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"

# ----------------------------------------------------------------------
# 1. Load & clean dataset (once)
# ----------------------------------------------------------------------
def _load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["gender", "smoking_status"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    required = ["age", "gender", "smoking_status", "bmi",
                "blood_pressure", "glucose_levels", "condition"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    return df.dropna(subset=["condition"])

_DATASET: pd.DataFrame = _load_dataset()

# ----------------------------------------------------------------------
# 2. Load RandomForest model (supports dict or plain model)
# ----------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

rf_model_raw = joblib.load(MODEL_PATH)
rf_model = rf_model_raw["model"] if isinstance(rf_model_raw, dict) else rf_model_raw
print(f"[INFO] RandomForest loaded from {MODEL_PATH}")

# ----------------------------------------------------------------------
# 3. Encoders – **exactly the same order used during training**
# ----------------------------------------------------------------------
le_gender    = LabelEncoder().fit(_DATASET["gender"])
le_smoking   = LabelEncoder().fit(_DATASET["smoking_status"])
le_condition = LabelEncoder().fit(_DATASET["condition"])

# ----------------------------------------------------------------------
# 4. Random true-case selector (stores hidden system message)
# ----------------------------------------------------------------------
def get_random_true_case(chat_id: str) -> Dict[str, Any]:
    """Pick a real patient and hide the ground-truth in the chat."""
    row = _DATASET.sample(n=1, random_state=random.randint(0, 2**31)).iloc[0]

    case = {
        "age": float(row["age"]) if pd.notna(row["age"]) else None,
        "gender": str(row["gender"]).strip(),
        "smoking_status": str(row["smoking_status"]).strip(),
        "bmi": float(row["bmi"]) if pd.notna(row["bmi"]) else None,
        "blood_pressure": float(row["blood_pressure"])
        if pd.notna(row["blood_pressure"]) else None,
        "glucose_levels": float(row["glucose_levels"])
        if pd.notna(row["glucose_levels"]) else None,
        "true_condition": str(row["condition"]).strip(),
    }

    add_message(chat_id, f"TRUE_CASE: {json.dumps(case)}", "system")
    return case

# ----------------------------------------------------------------------
# 5. OpenAI wrapper – graceful error handling
# ----------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing check .env")
client = OpenAI(api_key=api_key)


def ask_openai(
    chat_id: str,
    question: str,
    system_prompt: str = (
        "You are a helpful medical teaching assistant. "
        "Give concise, evidence-based answers. "
        "Never reveal the true diagnosis unless the student explicitly asks."
    ),
) -> str:
    """Send a question to OpenAI and store the full conversation."""
    add_message(chat_id, question, "doctor")

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in (get_chat(chat_id) or []):
        messages.append({"role": msg["role"], "content": msg["message"]})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,               # type: ignore[arg-type]
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        answer = resp.choices[0].message.content or ""
    except openai.OpenAIError as e:
        answer = f"[OpenAI error: {e}]"
    except Exception as e:
        answer = f"[Unexpected error: {e}]"

    add_message(chat_id, answer, "bot")
    return answer.strip()


# ----------------------------------------------------------------------
# 6. Conversation evaluator
# ----------------------------------------------------------------------
def _extract_true_case(chat_id: str) -> Dict[str, Any]:
    for msg in (get_chat(chat_id) or []):
        if msg["role"] == "system" and msg["message"].startswith("TRUE_CASE:"):
            return json.loads(msg["message"].replace("TRUE_CASE:", "").strip())
    raise ValueError(f"No TRUE_CASE for chat_id={chat_id}")


def evaluate_conversation(chat_id: str) -> Dict[str, Any]:
    """Return a rich evaluation dictionary."""
    true_case = _extract_true_case(chat_id)
    true_cond = true_case["true_condition"].lower()

    # ----- 1. Student diagnosis -----
    student_diagnosis: Optional[str] = None
    student_confidence: float = 0.0
    for msg in reversed(get_chat(chat_id) or []):
        if msg["role"] != "bot":
            continue
        txt = msg["message"].lower()
        for cond in ["diabetic", "pneumonia", "cancer"]:
            if cond in txt:
                student_diagnosis = cond
                m = re.search(rf"{cond}.*?(\d+\.\d+)%?", txt)
                if m:
                    try:
                        student_confidence = float(m.group(1)) / 100.0
                    except ValueError:
                        pass
                break
        if student_diagnosis:
            break

    # ----- 2. Accuracy -----
    if student_diagnosis is None:
        accuracy = 0.0
    elif student_diagnosis == true_cond:
        accuracy = 1.0
    elif any(w in student_diagnosis for w in true_cond.split()):
        accuracy = 0.5
    else:
        accuracy = 0.0

    # ----- 3. Model confidence on the true class -----
    row = pd.DataFrame([true_case])[
        ["age", "gender", "smoking_status", "bmi", "blood_pressure", "glucose_levels"]
    ]
    row["gender"] = le_gender.transform(row["gender"])
    row["smoking_status"] = le_smoking.transform(row["smoking_status"])

    # Impute missing numeric values with the **training-set medians**
    for col in ["age", "bmi", "blood_pressure", "glucose_levels"]:
        row[col] = row[col].fillna(_DATASET[col].median())

    probas = rf_model.predict_proba(row)[0]
    true_idx = np.where(le_condition.classes_ == true_case["true_condition"])[0][0]
    model_confidence = float(probas[true_idx])

    # ----- 4. Comfort / politeness -----
    polite = ["please", "thank you", "could you", "would you", "i think", "maybe", "perhaps"]
    comfort_score = sum(
        1 for m in (get_chat(chat_id) or [])
        if m["role"] == "bot" and any(p in m["message"].lower() for p in polite)
    )
    comfort_score = min(comfort_score / max(len(get_chat(chat_id) or []) - 1, 1), 1.0)

    # ----- 5. Explanation quality -----
    bot_len = sum(len(m["message"]) for m in (get_chat(chat_id) or []) if m["role"] == "bot")
    evidence = ["because", "due to", "evidence", "study", "guideline"]
    evidence_hits = sum(
        1 for m in (get_chat(chat_id) or [])
        if m["role"] == "bot" and any(k in m["message"].lower() for k in evidence)
    )
    explanation_quality = min((bot_len / 300) + (evidence_hits * 0.2), 1.0)

    # ----- 6. Assemble result -----
    return {
        "chat_id": chat_id,
        "true_condition": true_case["true_condition"],
        "student_diagnosis": student_diagnosis,
        "diagnosis_accuracy": accuracy,
        "model_confidence_on_truth": model_confidence,
        "student_confidence": student_confidence,
        "comfort_score": comfort_score,
        "explanation_quality": explanation_quality,
        "overall_score": round(
            0.4 * accuracy
            + 0.2 * model_confidence
            + 0.2 * comfort_score
            + 0.2 * explanation_quality,
            3,
        ),
    }


# ----------------------------------------------------------------------
# 7. Helper: save the trained model (call from notebook)
# ----------------------------------------------------------------------
def save_trained_model(model, path: str = "./models/rf_model.pkl") -> None:
    """Persist the model for later use by the DSS."""
    full_path = PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, full_path)
    print(f"[INFO] Model saved to {full_path}")


# ----------------------------------------------------------------------
# 8. Optional explicit buffer reset (useful for tests)
# ----------------------------------------------------------------------
def clear_chat_buffer(chat_id: str) -> None:
    """Delete all messages for a given chat_id."""
    # The original chat_model stores data in a module-level dict.
    # We simply re-create an empty list.
    from models.chat_model import chats
    chats[chat_id] = []


# ----------------------------------------------------------------------
# 9. Demo (run the file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cid = create_chat()
    case = get_random_true_case(cid)
    print("Random true case →", case)

    ask_openai(cid, "What symptoms does this patient present?")
    ask_openai(cid, "Based on the data, what is the most likely diagnosis?")

    report = evaluate_conversation(cid)
    print("\n=== Evaluation ===")
    print(json.dumps(report, indent=2))