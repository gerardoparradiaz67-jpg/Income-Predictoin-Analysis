# Census Income Prediction: A Comparative ML Study

## Executive Summary
This project evaluates the effectiveness of three different classification algorithms—**KNN, Bernoulli Naive Bayes, and Gaussian Naive Bayes**—in predicting whether an individual's income exceeds $50K/year based on census data.

## Key Results
* **Top Performance:** KNN and BernoulliNB achieved an accuracy of **81.3%** on unseen test data.
* **Key Insight:** GaussianNB performed poorly (45.02%) due to violations of the normality assumption in the dataset features.

## Model Evaluation
### Confusion Matrix (BernoulliNB)
![Confusion Matrix](confusion_matrix.png)
*The confusion matrix demonstrates the model's ability to balance True Positives and True Negatives, crucial for identifying high-income earners in a skewed dataset.*

### ROC/AUC Curve
![ROC Curve](roc_curve.png)
*By plotting the True Positive Rate against the False Positive Rate, we confirmed a robust AUC score, validating the model's discriminative power.*

## Technical Methodology
* **Preprocessing:** Feature scaling via `StandardScaler` and handling categorical variables.
* **Math Logic:** Leveraged BernoulliNB to better handle the binary-heavy feature space of the Census dataset.

## Future Work
To further improve predictive power, I plan to implement **Ensemble Methods** like Random Forest, which are natively robust against the skewed distributions found in this data.
