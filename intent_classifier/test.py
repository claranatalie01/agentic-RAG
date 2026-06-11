import os
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
ds = load_dataset("walledai/WildGuardTest", split="train")
print("Dataset size:", len(ds))

# ------------------------------------------------------------
# Load Saroku safety model (0.5B parameters, CPU-friendly)
# ------------------------------------------------------------
model_name = "Saroku/safety-classifier-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cpu")
model.eval()

def classify_prompt(text: str) -> int:
    """
    Returns 1 if unsafe, 0 if safe.
    The model outputs logits; we assume index 0 = safe, index 1 = unsafe.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred  # 0 = safe, 1 = unsafe

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
ds = load_dataset("walledai/WildGuardTest", split="train")
print(f"Dataset size: {len(ds)}")

true_labels = []
pred_labels = []

for i, sample in enumerate(ds):
    label_str = sample["label"]
    if label_str == "unharmful":
        true_label = 0
    elif label_str == "harmful":
        true_label = 1
    else:
        continue

    pred_label = classify_prompt(sample["prompt"])
    true_labels.append(true_label)
    pred_labels.append(pred_label)

    if (i + 1) % 200 == 0:
        print(f"Processed {i+1} samples...")

# ------------------------------------------------------------
# Compute metrics
# ------------------------------------------------------------
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix (rows: true, cols: pred):")
print(f"TN={cm[0,0]}  FP={cm[0,1]}")
print(f"FN={cm[1,0]}  TP={cm[1,1]}")