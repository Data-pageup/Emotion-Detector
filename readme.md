
# Customer Churn Classification – Model Evaluation & Analysis

This document summarizes the performance of multiple classification models trained on the cleaned **Customer Churn dataset**, along with detailed analysis and recommendations for improvement.

---

## Dataset Overview

- **Task**: Binary Classification (Churn Prediction)
- **Target Variable**: `churn`
  - `0` → No Churn  
  - `1` → Churn
- **Test Set Size**: 6000 samples  
- **Class Distribution (Test Set)**:
  - No Churn: 3947 (65.8%)
  - Churn: 2053 (34.2%)

---

## Model Performance Results

### 1. Decision Tree – Classification Report

```
              precision    recall  f1-score   support
    No Churn       0.80      0.79      0.79      3947
       Churn       0.60      0.62      0.61      2053
    accuracy                           0.73      6000
   macro avg       0.70      0.70      0.70      6000
weighted avg       0.73      0.73      0.73      6000
```

**Observations:**
- Moderate recall for churners (0.62)
- Simple and interpretable, but lower overall performance
- Tends to misclassify a significant number of churners

---

### 2. Random Forest – Classification Report

```
              precision    recall  f1-score   support
    No Churn       0.84      0.92      0.88      3947
       Churn       0.82      0.67      0.74      2053
    accuracy                           0.84      6000
   macro avg       0.83      0.80      0.81      6000
weighted avg       0.83      0.84      0.83      6000
```

**Observations:**
- Best overall accuracy (0.84)
- Strong precision for churners (0.82)
- Recall for churners (0.67) is better than Decision Tree but still misses some churn cases
- Good balance between stability and performance

---

### 3. Gradient Boosting – Classification Report

```
              precision    recall  f1-score   support
    No Churn       0.81      0.86      0.83      3947
       Churn       0.68      0.60      0.64      2053
    accuracy                           0.78      6000
   macro avg       0.77      0.73      0.74      6000
weighted avg       0.78      0.78      0.78      6000
```

**Observations:**
- Balanced performance
- Slightly lower recall for churners compared to Random Forest
- Performs better than a single Decision Tree but below Random Forest

---

## Model Comparison Summary

| Model             | Accuracy | Churn Recall | Churn Precision | Churn F1 | AUC-ROC |
|-------------------|----------|--------------|-----------------|----------|---------|
| Decision Tree     | 0.73     | 0.62         | 0.60            | 0.61     | 0.712   |
| Random Forest     | 0.84     | 0.67         | 0.82            | 0.74     | 0.809   |
| Gradient Boosting | 0.78     | 0.60         | 0.68            | 0.64     | 0.811   |
| XGBoost           | -        | -            | -               | -        | 0.808   |
| SVM               | -        | -            | -               | -        | 0.809   |
| KNN               | -        | -            | -               | -        | 0.771   |
| MLP               | -        | -            | -               | -        | 0.771   |

**Winner**: Random Forest provides the best overall performance with strong accuracy (0.84) and balanced AUC-ROC (0.809).

---

## ROC Curve Analysis

(<img width="588" height="464" alt="image" src="https://github.com/user-attachments/assets/b5ae3fd9-5079-473e-a98d-367f75526e29" />)

### Model Performance by AUC-ROC

The ROC (Receiver Operating Characteristic) curve visualizes the tradeoff between True Positive Rate (Recall) and False Positive Rate across different classification thresholds.

**AUC-ROC Rankings:**
1. **Gradient Boosting**: 0.811 (Best discrimination ability)
2. **SVM**: 0.809
3. **Random Forest**: 0.809
4. **XGBoost**: 0.808
5. **KNN**: 0.771
6. **MLP**: 0.771
7. **Decision Tree**: 0.712 (Weakest discrimination)

### Key Insights from ROC Analysis

**Top Performers (AUC > 0.80)**
- Gradient Boosting, SVM, Random Forest, and XGBoost all achieve excellent discrimination with AUC scores above 0.80
- These models effectively separate churners from non-churners across various threshold settings
- Gradient Boosting edges out slightly with 0.811, suggesting marginally better ranking ability

**Mid-Tier Models (AUC 0.77-0.78)**
- KNN and MLP show solid but not exceptional performance
- May struggle with complex decision boundaries in the feature space

**Baseline Performer (AUC 0.71)**
- Decision Tree significantly underperforms compared to ensemble methods
- Limited ability to capture complex patterns without overfitting

### ROC vs. Classification Report Trade-offs

**Important Note**: While Gradient Boosting has the highest AUC (0.811), Random Forest achieved:
- Higher overall accuracy (0.84 vs 0.78)
- Better precision for churn (0.82 vs 0.68)
- Slightly better recall (0.67 vs 0.60)

This demonstrates that **AUC alone doesn't tell the full story**. The choice between models depends on:
- **Gradient Boosting**: Best for ranking customers by churn probability
- **Random Forest**: Best for actual classification with balanced performance

### Practical Implications

**For Threshold Optimization:**
- All top models (AUC > 0.80) have room for threshold tuning
- The steep initial climb in ROC curves suggests significant gains possible by lowering threshold
- Models can achieve ~75-80% recall with acceptable false positive rates

**For Campaign Targeting:**
- Use AUC to rank customers by churn risk
- Use precision/recall metrics to set campaign reach
- Consider deploying multiple models and ensembling their predictions

---

## Critical Analysis

### 1. The Recall Gap Problem

The best model (Random Forest) still misses **33% of churners** (recall = 0.67).

**Business Impact:**
- Out of 2,053 actual churners in the test set, approximately 677 customers are missed
- These represent lost revenue opportunities where proactive retention could have helped
- Missing churners is typically more costly than targeting some false positives

### 2. Class Imbalance Impact

The dataset shows a 66:34 split (No Churn:Churn). While not extreme, this imbalance causes models to default toward predicting "No Churn" more conservatively, prioritizing overall accuracy over churn detection.

### 3. Precision-Recall Tradeoff

Random Forest achieves high precision (0.82) for churn, meaning when it predicts churn, it's usually correct. However, it's being too conservative, which hurts recall. For churn prediction, we typically want to err on the side of higher recall.

---

## Optimization Strategies

### Strategy 1: Threshold Adjustment
- Instead of using the default 0.5 probability threshold, optimize for higher recall
- Adjust decision boundary to classify more customers as "churn"
- **Expected Impact**: Could boost churn recall from 0.67 → 0.75-0.80, though precision may drop slightly

### Strategy 2: Class Weight Balancing
- Train models with adjusted class weights to penalize churn misclassification more heavily
- Use `class_weight='balanced'` or custom weights like `{0: 1, 1: 2}`
- **Expected Impact**: Should improve churn recall by 5-10 percentage points

### Strategy 3: Cost-Sensitive Learning
- Use algorithms like XGBoost with `scale_pos_weight` parameter
- Amplify the importance of the minority class (churners)
- **Expected Impact**: XGBoost often achieves 3-7% better recall than Random Forest on imbalanced datasets

### Strategy 4: Resampling Techniques
- Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance training data
- Generate synthetic examples of churners to help model learn patterns better
- Always evaluate on original test set, not resampled data

---

## Advanced Model Recommendations

Given the current baseline, these models are likely to improve recall:

1. **XGBoost** - Often best for imbalanced classification with built-in handling via `scale_pos_weight`
2. **LightGBM** - Fast and effective with `is_unbalance=True` parameter
3. **CatBoost** - Handles imbalance well by default with minimal tuning required
4. **Ensemble Stacking** - Combine predictions from Random Forest + Gradient Boosting + XGBoost

---

## Business-Focused Metrics

### Retention Campaign ROI Analysis

**Assumptions:**
- Cost to retain a customer: $50
- Value of retained customer: $500 (Customer Lifetime Value)
- Campaign reaches top N predicted churners

**Current Random Forest Performance:**

If targeting top 2,053 predictions (equal to actual churners):
- True Positives: ~1,376 (67% recall)
- False Positives: ~336 (based on precision = 0.82)
- Total campaign cost: 1,712 × $50 = $85,600
- Revenue from retained customers: 1,376 × $500 = $688,000
- **Net ROI: $602,400**

**With Improved Recall (0.80):**
- True Positives: ~1,642
- Revenue: 1,642 × $500 = $821,000
- **Net ROI: ~$735,000 (22% improvement)**

This demonstrates that even modest improvements in recall translate to significant business value.

---

## Evaluation Metrics for Imbalanced Classification

### Why Accuracy is Misleading
- A model predicting "No Churn" for all customers would achieve 66% accuracy
- This metric doesn't capture the model's ability to identify churners

### Better Metrics for This Problem

**F-Beta Score (Beta = 2)**
- Weights recall twice as much as precision
- Better reflects business priorities where missing churners is costly

**Precision-Recall Curves**
- Shows tradeoff between precision and recall at different thresholds
- Helps find optimal operating point for business needs

**Cost Matrix Analysis**
- Assign actual dollar values to true positives, false positives, false negatives
- Optimize for maximum business value, not just statistical metrics

---

## Key Insights & Recommendations

### What Works Well
- ✅ Random Forest provides strong baseline performance (84% accuracy)
- ✅ High precision (0.82) means confident predictions are usually correct
- ✅ Model successfully identifies 67% of churners

### Areas for Improvement
- ⚠️ Missing 33% of churners represents significant lost revenue
- ⚠️ Default decision threshold (0.5) is too conservative
- ⚠️ Class imbalance needs explicit handling through weights or resampling

### Priority Actions
1. **Immediate**: Optimize decision threshold to target 75-80% recall
2. **Short-term**: Retrain with class-weight balancing
3. **Medium-term**: Experiment with XGBoost and LightGBM
4. **Long-term**: Build ensemble model combining multiple approaches

---

## Success Metrics

### Technical Goals
- Achieve churn recall ≥ 0.75 while maintaining precision ≥ 0.70
- Improve F2 score by at least 10%
- Reduce false negative rate to under 25%

### Business Goals
- Increase retention campaign ROI by 20-30%
- Identify 200+ additional at-risk customers per month
- Reduce customer churn rate by 2-3 percentage points

---

## Conclusion

The Random Forest model provides a solid foundation with 84% accuracy and strong precision. However, the primary opportunity lies in improving **churn recall** from 67% to 75-80%. This improvement would:

- Identify an additional ~200-300 at-risk customers
- Generate ~$130,000+ in additional retention value
- Better align model performance with business objectives

The recommended approach combines quick wins (threshold tuning) with strategic improvements (class balancing, advanced algorithms) to maximize both technical performance and business impact.

---



#

**Last Updated**: December 2025
