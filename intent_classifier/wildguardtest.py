import os
from dotenv import load_dotenv
from datasets import load_dataset
from gliner2 import GLiNER2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

ds = load_dataset("walledai/WildGuardTest", split="train")
print("Dataset size:", len(ds))
# Load GLiGuard model
model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
model.to("cpu")

def classify_prompt(text: str) -> int:
    """Return 1 (unsafe) or 0 (safe)."""
    try:
        result = model.classify_text(text, {"prompt_safety": ["safe", "unsafe"]})
        return 1 if result.get("prompt_safety") == "unsafe" else 0
    except Exception:
        return 0  # fallback to safe

true_labels = []
pred_labels = []

for sample in ds:
    # Map string labels to integers
    label_str = sample["label"]
    if label_str == "unharmful":
        true_label = 0      # safe
    elif label_str == "harmful":
        true_label = 1      # unsafe
    else:
        # Skip any unexpected label
        continue

    pred_label = classify_prompt(sample["prompt"])
    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Compute metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix (rows: true, cols: pred):")
print(f"TN={cm[0,0]}  FP={cm[0,1]}")
print(f"FN={cm[1,0]}  TP={cm[1,1]}")