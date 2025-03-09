# 📱 SMS Classification using GPT-2 (125M)

## 📌 Overview
This project demonstrates SMS classification using a fine-tuned GPT-2 (125M) model. The model is trained to identify whether an SMS is **spam** or **ham (legitimate message)** based on its content.

## 🛠 Model Details
- **Model Used**: GPT-2 (125M)
- **Dataset**: A balanced sarcasm dataset with `merged_comments` and `labels`
- **Tokenizer**: GPT-2 Tokenizer
- **Max Context Length**: 1024 tokens
- **Device**: CUDA (GPU) for fast inference

## 🚀 Classification Results
The following examples showcase the model's classification on real-world SMS samples:

### **Example 1: Spam SMS**
#### 📩 Input SMS:
```text
Final Match Alert! Watch Ind vs NZ LIVE in HD on JioHotstar with Vi pack of Rs169. Get 3months JioHotstar+8GB data
```
#### ✅ Model Prediction:
```text
Spam
```
---

### **Example 2: Ham (Legitimate) SMS**
#### 📩 Input SMS:
```text
Hey, mom I would be late today. Going to have dinner at Jimmy's place!
```
#### ✅ Model Prediction:
```text
Ham
```
---

## 📌 Implementation Code
```python
import torch

def classify_review(text, model, tokenizer, device, max_length):
    model.eval()
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Ham"
```

## 📊 Model Performance

### 🔹 Accuracy vs. Epochs
![download (1)](https://github.com/user-attachments/assets/9e6d3194-8649-4369-b0fd-baa4164a74cd)


### 🔹 Loss vs. Epochs
![download](https://github.com/user-attachments/assets/35d3267a-9e7c-45bb-b88c-ae27612c246c)

## 🎯 Key Insights
- The model correctly identifies promotional messages as **spam**.
- It classifies personal messages as **ham**.
- **Performance**: Efficient inference on GPU (CUDA).
- **Optimization**: Messages exceeding 1024 tokens are truncated for processing efficiency.

## 📌 Future Improvements
- **Fine-tune on a larger SMS spam dataset**.
- **Improve handling of long messages** by using segment-based classification.
- **Deploy as an API for real-time SMS classification**.



