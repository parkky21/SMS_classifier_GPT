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
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
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



