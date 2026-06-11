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

model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
model.to("cpu")

def classify_prompt(text: str) -> int:
    try:
        # Toxicity labels (including "benign")
        toxicity_labels = [
            "self_harm_and_suicide",
            "political_manipulation",
            "violence_and_weapons",
            "hate_and_discrimination",
            "jailbreak_attempt",
            "prompt_injection",
            "benign"
        ]
        toxicity_task = {
            "labels": toxicity_labels,
            "multi_label": True,
            "cls_threshold": 0.4
        }
        result = model.classify_text(
            text,
            {
                "prompt_safety": ["safe", "unsafe"],
                "prompt_toxicity": toxicity_task
            }
        )
        # Binary safety
        is_unsafe_binary = result.get("prompt_safety") == "unsafe"

        # Multi‑label toxicity: result["prompt_toxicity"] is a list of detected labels
        detected_toxicity = result.get("prompt_toxicity", [])
        # If any label other than "benign" is in the list, treat as unsafe
        is_unsafe_toxicity = any(label != "benign" for label in detected_toxicity)

        return 1 if (is_unsafe_binary or is_unsafe_toxicity) else 0
    except Exception as e:
        print(f"Classification error: {e}")
        return 0  # fallback to safe

true_labels = []
pred_labels = []

for sample in ds:
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