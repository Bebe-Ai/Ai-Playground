# AI Playground

Welcome! This repo is my AI learning playground.  
Goal: Learn AI by building small projects and eventually create AI-powered businesses.

## Modules

### Week 1 - Basics
- **Notebook:** [Week 1: Spam Detector](week1_spam_detector.ipynb) 
- **What I did:** Built my first text classifier (spam vs ham) using TF-IDF + Logistic Regression.  
- **What I learned:** How ML models learn from examples and make predictions.

### Week 2 - Transformers
- **Notebook:** [Week 2: Transormers](week2_transformers.ipynb)  
- **What I did:** Ran a small transformer model (`distilgpt2`) to generate text.  
- **What I learned:** How LLMs generate text and what `max_new_tokens` does.

### Week 3 - Customer Support AI Bot

This project contains a Google Colab notebook that trains and tests a **custom Question-Answering AI model** for customer support.  
The goal is to help businesses automatically respond to customer queries such as shipping, returns, or product details.

---

## 🚀 Features
- Fine-tuned transformer model for **customer Q&A**  
- Runs entirely in **Google Colab**  
- Interactive testing inside the notebook  
- Save & reload your model for later use  
- Ready for deployment in a chatbot or web app  

---

## 📂 Notebook
- **[Week 3: Training](week3_training.ipynb)**  
  Use this notebook to train and interact with your bot.

---

## 🛠️ How It Works
1. Generate or upload a dataset of common **questions & answers**.  
2. Fine-tune a pre-trained transformer model (T5, BART, etc).  
3. Save the trained model and tokenizer for reuse.  
4. Test your bot interactively in Colab.  

---

## 🧑‍💻 Example Usage
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("./customer_support_model")
model = AutoModelForSeq2SeqLM.from_pretrained("./customer_support_model")

def ask_bot(question):
    input_text = "question: " + question
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(ask_bot("Do you ship internationally?"))
