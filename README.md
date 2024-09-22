# Cost-sensitive-Learning-for-ML-models


#**😇** **Motivation**

1. I dealt with various ML modeling dorectly and indirectly in my current and previous roles. Imbalanced data has always been an issue for ML model training. In a balanced dataset, the number of Positive and Negative labels is about equal. However, if one label is more common than the other label, then the dataset is imbalanced .You might think you got good, historical traning data but removing bias was always an issue.
2. I knew 1-2 approaches on how to solve this imbalanced dataset issue but I wanted to know optimal approaches and what is true cost/penalty for  selecting  imbalanced dataset.
3. My current role deals with data pipeline,observability,tennacy, reliability of data infrastructure and AIOPs is becoming more nad more prominent in those areas. ML and AL has its differences but the core algorithms and basic steps of deployment are same. The MLOPS cycle is(Data collection->Clean->Train/test->validate->feature store reuse->deploy into CI CD pipeline->Monitor->Govern->Reuse/Analyze metrics)
4. This is why I wanted to understand how to take care of this issue and what is the impact/cost/penalty of not solving this.
5. Moreover I wanted to know the mathametical equation behind solving imbalanced data, the effect on various algorithms,ensembled learning and hyperparameter tuning which I was out of touch for a while
7. My next goal is to test this model with some new cases beyong credit risk analysis, build MLOPS pipeline in Azure data environment and then see how it works on GEN AI(Not sure if it can be deployed from normal PC without much GPU support)



# What is this Project About
1. This project is based on an udemy course of Data Scientist Soledid Galli. She is a great teacher,Open source contributor,instructor who is able to break down complex ML issues from scratch.She described various approaches for dealing with Imbalanced data set , how to choose and interpret performance metrics and how to train different ML models with each of those approaches.However since I was interested on just the cost sensitive ML training, I focused on the cost sensitive ML model approach and this github project is based on cost sensitive ML training part only
2. The goal of this project is to understand various approaches for solving imbalanced data set, what is cost sensitive ML Model traning, what are the different approaches for cost sensitive traning and howcost functionimpacts different ML classification models and how to interpret Model performance metrics.
3. We use a credit Risk analysis dataset to  tThis is a classification based Project. Classification is a predictive modeling technique which involves predicting a class label for a given observation.An observation from the domain (input) and an associated class label (output).A common example of classification comes with detecting spam emails. To write a program to filter out spam emails, a computer programmer can train a machine learning algorithm with a set of spam-like emails labelled as spam and regular emails labelled as not-spam
4. Imbalance dataset is a dataset where one category is disproportionately more present than the other .The results for using an imbalance dataset in an ML model could be disastrous. For example, in a credit risk model, the algorithm might incorrectly predict a "low-risk loan" when it’s actually “high risk”. Similarly, a medical model might diagnose a diabetic patient as “healthy” when they are, in fact, diabetic.
This is why cost sensitive model training for linear regression or other classification modelling is important. The cost function used in logistic regression is designed to adjust for incorrect predictions. Linear Regression predicts output for  agiven input  but the cost function is used to measure how far the predicted values are from the actual values.There are multiple approaches to solve this issue. In this project I will focus only on cost sensitive traning part
![image](https://github.com/user-attachments/assets/6106f97d-2c84-4a5d-b82d-3cfca3aea071)
5. Cost-sensitive learning is a subfield of machine learning that takes the costs of prediction errors (and potentially other costs) into account when training a machine learning model. It is a field of study that is closely related to the field of imbalanced learning that is concerned with classification on datasets with a skewed class distribution. As such, many conceptualizations and techniques developed and used for cost-sensitive learning can be adopted for imbalanced classification problems


6. The cost of a misclassification error is conditional on the circumstances.Cost depends on the nature of the case / observation
7. But how do you calculate that cost ? there are two approaches to that
8. ![image](https://github.com/user-attachments/assets/7a80392d-b40f-40e3-9b4c-ae5f6c2b0f5c)
  
9. In this project  I will use historical data set of Bank to understand credit default risk

10. Imbalanced datasets are those where there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class(User will Default) to the majority class(User will not default)
11. Some of the top-ranked machine learning algorithms for Classification are:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
K-Nearest Neighbors (KNN)
# How I worked on this project
1. I started with dataset download from IEEE,Kaggle etc sources followed by intensive data cleaning to remove unwanted features, duplicates, treating null values etc
2. There is 3 ways to get cost of misclassification. Domain Expert provides the cost(Not needed for this course),Balanced weight Ratio (code is in notebook) and Cross-validation: find cost as hyper-parameter (code is in notebook)
3. In notebook,There are 2 ways in which we can introduce cost into the learning function of the algorithm with Scikit-learn:

Defining the class_weight parameter for those estimators that allow it, when we set the estimator
Passing a sample_weight vector with the weights for every single observation, when we fit the estimator.
With both the class_weight parameter or the sample_weight vector, we indicate that the loss function should be modified to accommodate the class imbalance and the cost attributed to each misclassification.

parameters
class_weight: can take 'balanced' as argument, in which case it will use the balance ratio as weight. Alternatively, it can take a dictionary with {class: penalty}, pairs. In this case, it penalizes mistakes in samples of class[i] with penalty[i].

So if class_weight = {0:1, and 1:10}, misclassification of observations of class 1 are penalized 10 times more than misclassification of observations of class 0.

sample_weight is a vector of the same length as y, containing the weight or penalty for each individual observation. In principle, it is more flexible, because it allows us to set weights to the observations and not to the class as a whole. So in this case, for example we could set up higher penalties for fraudulent applications that are more costly (money-wise) than to those fraudulent applications that are of little money.
Estimating the Cost with Cross-Validation

We mentioned that there are 3 ways of estimating the cost:

Domain Expert provides the cost
Balance Ratio (we did this in previous notebook)
Cross-validation: find cost as hyper-parameter
In this notebook, we will find the cost with hyper parameter search and cross-validation.
Credit risk
Machine Learning with Imbalanced Data - Course

In this notebook, we'll create 3 models to assess credit risk by using:

Logistic regression
Random forests
XGBoost
And we'll compare their performance after applying cost-sensitive learning.
4. Tried to use this cost sensitive model into different ML algorithms(linear Regression, Random Forest, XGBOOST) for single use case (Credit Default Risk Analysis) to check whether incorporating cost sensitive training improve Performance metrics such as ROC/AUC,Preision/Recall etc  of those different ML Models
4.The cost function used in logistic regression is designed to adjust for incorrect predictions. Linear Regression predicts output for  agiven input  but the cost function is used to measure how far the predicted values are from the actual values.
5. Worked on another approach with Meta Label




# Next Steps/ Cost effect of misclassification on AI Models
I will try to play with few other methods for fixing Imbalance datasets such as Bagging,Boosting etc
Deploy this in Azure Pipeline in Azure data platform ?
Use the same method for traning Image classification (Neural Network based training) and see if it provides good performance metrics
ML oversampling methods such as SMOTE won't probably work with Neural Network so  An alternative to SMOTE is to obtain synthetic points using generative models such as Variational AutoEncoders. Generative models have demonstrated enormous potential when handling complex data distributions during the last few years. Their success at generating realistic data makes them a new paradigm to solve dataset oversampling.
The long tail problem in image classification refers to the imbalance in data distribution where a few classes have many samples, while most classes have few samples. This is a common issue in real-world image recognition problems, such as medical imaging





