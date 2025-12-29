# Customer Churn Dataset (in progress)

## 📌 Overview
This dataset contains customer-level information for analyzing and predicting **customer churn** in a telecommunications-like service environment. Each row represents a single customer with demographic, service usage, contract, and support-related attributes, along with a churn label.

The dataset is suitable for:
- Binary classification (churn prediction)
- Exploratory data analysis (EDA)
- Feature engineering practice
- Machine learning and deep learning experiments

---

## 📂 Dataset Structure

- **Number of records:** ~20,000  
- **Target variable:** `churn`  
- **Data types:** Numerical + Categorical  
- **Granularity:** One row per customer  

---

## 🧾 Column Descriptions

| Column Name         | Data Type     | Description |
|---------------------|--------------|-------------|
| `customer_id`       | Integer       | Unique identifier for each customer |
| `tenure`            | Integer       | Number of months the customer has stayed with the company |
| `monthly_charges`   | Float         | Monthly amount charged to the customer |
| `total_charges`     | Float         | Total amount charged over the customer’s tenure |
| `contract`          | Categorical   | Type of contract (`Month-to-month`, `One year`, `Two year`) |
| `payment_method`    | Categorical   | Payment method (`Credit`, `Debit`, `UPI`, `Cash`) |
| `internet_service`  | Categorical   | Type of internet service (`DSL`, `Fiber`) |
| `tech_support`      | Categorical   | Whether the customer has tech support (`Yes`, `No`) |
| `online_security`   | Categorical   | Whether online security is enabled (`Yes`, `No`) |
| `support_calls`     | Integer       | Number of customer support calls made |
| `churn`             | Categorical   | Whether the customer churned (`Yes`, `No`) |

---

## 🎯 Target Variable

- **`churn`**
  - `Yes` → Customer has left the service
  - `No` → Customer is still active

This is a **binary classification** problem.

---

## 🔍 Potential Use Cases

- Predicting customer churn using ML models
- Identifying high-risk customers
- Feature importance and explainability studies
- Handling categorical data and class imbalance
- Customer lifetime value (CLV) analysis

---

## 🛠 Suggested Preprocessing Steps

- Encode categorical variables (One-Hot / Target Encoding)
- Scale numerical features (`tenure`, `monthly_charges`, `total_charges`)
- Check for class imbalance in `churn`
- Handle potential outliers in `support_calls` and charges

---

## 📊 Example Row

```text
customer_id: 5  
tenure: 21  
monthly_charges: 39.38  
total_charges: 826.98  
contract: Month-to-month  
payment_method: UPI  
internet_service: Fiber  
tech_support: No  
online_security: No  
support_calls: 4  
churn: Yes

