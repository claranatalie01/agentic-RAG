from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

query = "What time does Shatin Library close?"
doc = "Shatin Public Library hours: 9am-8pm weekdays, 10am-6pm weekends."

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
model.eval()

# Tokenize the pair
inputs = tokenizer(query, doc, return_tensors="pt", truncation=True, max_length=512)

# Compute score
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape [1, 2]
    probs = F.softmax(logits, dim=1)  # shape [1, 2]
    score = probs[0, 1].item()  # probability of relevant class (index 1)

print(f"Relevance score: {score:.4f}")