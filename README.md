## PROJECT 1
## **Overview**
Logistic Regression is a widely-used method for classifying categorical outcomes, particularly in binary scenarios such as Yes/No, Positive/Negative, or Success/Failure. This project focuses on implementing logistic regression in R to predict binary outcomes based on a set of predictor variables.

## **Implementation**
The project provides a practical example of logistic regression using R and a dataset related to predicting customers' acceptance of a personal loan offer. It covers steps such as data preprocessing, model fitting, prediction, and evaluation using confusion matrix and decile chart.

## **Key Steps**
Data Preparation: Loaded the dataset and preprocessed it as needed.
Model Training: Used the glm() function in R with family = "binomial" to fit the logistic regression model.
Prediction: Utilized the trained model to predict probabilities or class labels for new data.
Evaluation: Evaluated the model's performance using a confusion matrix, decile chart, and other relevant metrics.

## PROJECT 2
Classification and Regression Trees (CART) Project
## **Overview**
This project focuses on the implementation and application of Classification and Regression Trees (CART), a versatile and interpretable data-driven method used for both classification and prediction tasks. CART provides clear decision rules by recursively partitioning the dataset based on predictor variables.

## **Usage**
Setup: Ensure you have the required libraries (e.g., rpart, rpart.plot) installed in your R environment.
Data Preparation: Loaded dataset and preprocessed it as needed.
Model Training: Used the rpart() function to build a classification or regression tree.
Visualization: Utilized the prp() function to visualize the generated tree.
Parameter Tuning: Experimented with parameters like cp and minsplit to balance tree complexity and performance.
Evaluation: Assessed the model's performance using confusion matrices on training and validation datasets.
Considerations
