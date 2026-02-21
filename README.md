# 🏥 Cost-Sensitive Disease Prediction System

A Machine Learning project focused on **minimizing False Negatives** in medical diagnosis using **Cost-Sensitive Learning**.

> ⚠ In healthcare, predicting a diseased patient as healthy can have serious consequences.  
> This project prioritizes **Recall** over Accuracy to reduce missed disease cases.

---

## 📌 Problem Statement

In medical diagnosis systems:

- False Positive → Patient is healthy but predicted diseased  
- False Negative → Patient has disease but predicted healthy  

Here, **False Negative cost is 10x higher** than False Positive.

🎯 Objective:
> Minimize False Negatives even if False Positives slightly increase.

---

## 🚀 Solution Approach

To handle cost-sensitive classification:

✔ Used **Logistic Regression**  
✔ Applied `class_weight = {0:1, 1:10}`  
✔ Optimized for **Recall instead of Accuracy**  
✔ Implemented modular ML architecture  

This forces the model to penalize missed disease cases more heavily during training.

---

## 🧠 Why Recall?

Recall formula:

\[
Recall = \frac{TP}{TP + FN}
\]

Maximizing recall ensures:
> No disease case is missed.

---

## 📊 Model Performance

**Confusion Matrix:**

[[171 4]
[ 0 25]]


- False Negatives (FN): **0**
- Recall Score: **1.0**
- Accuracy: **98%**

✅ The model successfully captured all disease cases.

---

## 🛠 Tech Stack

- Python 3.12
- NumPy
- Pandas
- Scikit-Learn
- Joblib