# Cost-sensitive-Learning-for-ML-models


#**😇** **Motivation**

1. I dealt with various ML modeling dorectly and indirectly in my current and previous roles. Imbalanced data has always been an issue for ML model training. In a balanced dataset, the number of Positive and Negative labels is about equal. However, if one label is more common than the other label, then the dataset is imbalanced .You might think you got good, historical traning data but removing bias was always an issue.
2. I knew 1-2 approaches on how to solve this imbalanced dataset issue but I wanted to know optimal approaches and what is true cost/penalty for  selecting  imbalanced dataset.
3. My current role deals with data pipeline,observability,tennacy, reliability of data infrastructure and AIOPs is becoming more nad more prominent in those areas. ML and AL has its differences but the core algorithms and basic steps of deployment are same. The MLOPS cycle is(Data collection->Clean->Train/test->validate->feature store reuse->deploy into CI CD pipeline->Monitor->Govern->Reuse/Analyze metrics)
4. This is why I wanted to understand how to take care of this issue and what is the impact/cost/penalty of not solving this.
5. Moreover I wanted to know the mathametical equation behind solving imbalanced data, the effect on various algorithms,ensembled learning and hyperparameter tuning which I was out of touch for a while

# What is this Project About
1. This project is based on an udemy course of Data Scientist Soledad Galli(https://www.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22733079#overview
). She described various approaches for dealing with Imbalanced data set , how to choose and interpret performance metrics and how to train different ML models with each of those approaches.However since I was interested on just the cost sensitive ML training, I focused on the cost sensitive ML model approach and this github project is based on cost sensitive ML training part only
2. The goal of this project is to understand various approaches for solving imbalanced data set, what is cost sensitive ML Model traning, what are the different approaches for cost sensitive traning and how cost function impacts different ML classification models and how to interpret Model performance metrics.
3. We use a credit Risk analysis dataset to check the viability of cost function. This is a classification based Project. Classification is a predictive modeling technique which involves predicting a class label for a given observation.An observation from the domain (input) and an associated class label (output).A common example of classification comes with detecting spam emails. To write a program to filter out spam emails, a computer programmer can train a machine learning algorithm with a set of spam-like emails labelled as spam and regular emails labelled as not-spam
4. In this project  I will use historical data set of Bank to interpret credit default risk.This is Customer Transaction and Demographic related data , It holds Risky and Not Risky customer for specific banking products. This is an Imbalanced datasets because there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class(User will Default) to the majority class(User will not default)
5. Linear Regression predicts output for  a given input  but the cost function is used to measure how far the predicted values are from the actual values so the code will show how cost function improves performance metrics of Logistic regression and impact of using cost function on different other ML models(whether it works or not)
# How I worked on this project
1. I started with dataset download from IEEE,Kaggle etc sources followed by intensive data cleaning to remove unwanted features, duplicates, treating null values etc.Refer to python notebook data-preparation-credit-risk
2. In this project  I used historical data set from a Bank to interpret credit default risk..This is Customer Transaction and Demographic related data , It holds Risky and Not Risky customer for specific banking products. This is an Imbalanced datasets because there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class(User will Default) to the majority class(User will not default).Imbalance dataset is a dataset where one category is disproportionately more present than the other .The results for using an imbalance dataset in an ML model could be disastrous. For example, in a credit risk model, the algorithm might incorrectly predict a "low-risk loan" when it’s actually “high risk”.
3. Credit Default Risk gets predicted by using any type of classification based ML Algorithm.Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data. In classification, the model is fully trained using the training data, and then it is evaluated on test data before being used to perform prediction on new unseen data.
4. For instance, an algorithm can learn to predict whether a given email is spam or no spam, as illustrated below. 
![image](https://github.com/user-attachments/assets/b3414560-298f-4e97-b0b3-2aef475f81ed)

<br> Machine learning classification illustration for the email
 <br>Some of the top-ranked machine learning algorithms for Classification are:
<br>Logistic Regression
<br>Decision Tree
<br>Random Forest
<br>Support Vector Machine (SVM)
<br>Naive Bayes
<br>K-Nearest Neighbors (KNN)
<br>We tried only Logistic Regression, Random Forest and XGBoost for this project
A key component of machine learning classification tasks is handling unbalanced data, which is characterized by a skewed class distribution with a considerable overrepresentation of one class over the others.This is why fixing imbalanced dataset for linear regression or other classification modelling is important
 
5. Then I have to decide which approach to take to deal with Imbalanced data set. You can solve it at Data level(Changing Data distribution by Over or under sampling or generating synthetic data or removing noise from data ) or you can use ensemble method(Combing sampling technique with tuning traning data set by modifying weak learners/high bias/variance training dataset) or you can use cost-function method(adding a cost/penalty parameter to existing ML algorithm to minimize error/bias of prediction thus inimizing chances of misclassifying minority class)
There are multiple approaches to solve this issue. In this project I will focus only on cost sensitive traning part
![image](https://github.com/user-attachments/assets/6106f97d-2c84-4a5d-b82d-3cfca3aea071)

6. Cost-sensitive learning is a subfield of machine learning that takes the costs of prediction errors (and potentially other costs) into account when training a machine learning model. It is a field of study that is closely related to the field of imbalanced learning that is concerned with classification on datasets with a skewed class distribution. As such, many conceptualizations and techniques developed and used for cost-sensitive learning can be adopted for imbalanced classification problemsThe cost function used in logistic regression is designed to adjust for incorrect predictions. Linear Regression predicts output for  agiven input  but the cost function is used to measure how far the predicted values are from the actual values.T
7. The cost of a misclassification error is conditional on the circumstances.Cost depends on the nature of the case / observation


8. But how do you calculate that cost ? there are two approaches to that
![image](https://github.com/user-attachments/assets/7a80392d-b40f-40e3-9b4c-ae5f6c2b0f5c)
  
# Logic behind each Python Code and how imbalanced data is fixed in Linear Regression or other ML Models
 1. There is 3 ways to get cost of misclassification. Domain Expert provides the cost(Not needed for this course),Balanced weight Ratio (code is in notebook) and Cross-validation: find cost as hyper-parameter (code is in notebook)

  
2. In Python notebook "Cost sensitive learning with Balanced weight class",We explained There are 2 ways in which we can introduce cost into the learning function of the algorithm with Scikit-learn:

Defining the class_weight parameter for those estimators that allow it, when we set the estimator
Passing a sample_weight vector with the weights for every single observation, when we fit the estimator.
With both the class_weight parameter or the sample_weight vector, we indicate that the loss function should be modified to accommodate the class imbalance and the cost attributed to each misclassification.

parameters
class_weight: can take 'balanced' as argument, in which case it will use the balance ratio as weight. Alternatively, it can take a dictionary with {class: penalty}, pairs. In this case, it penalizes mistakes in samples of class[i] with penalty[i].

So if class_weight = {0:1, and 1:10}, misclassification of observations of class 1 are penalized 10 times more than misclassification of observations of class 0.

sample_weight is a vector of the same length as y, containing the weight or penalty for each individual observation. In principle, it is more flexible, because it allows us to set weights to the observations and not to the class as a whole. So in this case, for example we could set up higher penalties for fraudulent applications that are more costly (money-wise) than to those fraudulent applications that are of little money.
Estimating the Cost with Cross-Validation.


3.In Python notebook "Training Cost with Cross-Validation",We explained  We mentioned that there are 3 ways of estimating the cost:
<br> Domain Expert provides the cost
<br>  Balance Ratio (we did this in previous notebook)
<br>  Cross-validation: find cost as hyper-parameter
<br> In this notebook, we will find the cost with hyper parameter search and cross-validation.Cross validation provides a more robust estimate of a model's performance than a single train-test split, while hyperparameter tuning helps to find the optimal set of hyperparameters for a model. By using these techniques, we can build more accurate and reliable machine learning models

<br> 4. In Python notebook "Credit risk analysis with 3 models",We explained  we'll create 3 models to assess credit risk by using:
<br> Logistic regression,Random forests and XGBoost for single use case (Credit Default Risk Analysis) to check whether incorporating cost sensitive training improve Performance metrics such as ROC-AUC Curve ,Preision-Recall curve etc  of those different ML Models
<br>5.The cost function used in logistic regression is designed to adjust for incorrect predictions. Linear Regression predicts output for  agiven input  but the cost function is used to measure how far the predicted values are from the actual values.
<br>6.Then we will analyze how cost function is able to improve or failed to improve Performance metrics such as  ROC-AUC Curve ,Preision-Recall curve for each model
<br> And we'll compare their performance after applying cost-sensitive learning.
<br>7.We'll carry out different feature engineering steps for logistic regression and tree based models.For logistic regression we'll impute with the mean and add missing indicators. For tree based models we'll impute with an arbitrary number.
<br> For logistic regression we'll do one hot encoding, for tree based models, we'll carry out ordinal encoding.
<br>8.we'll pass class_weight as parameter into ML models
<br>9. Worked on another approach with Meta Label in another notebook



# How to interpret Performance Metrics such as ROC-AUC Curve or Precision-Recall Curve
<br>1.When making a prediction for a binary or two-class classification problem, there are two types of errors that we could make.

<br>False Positive. Predict an event when there was no event.
<br>False Negative. Predict no event when in fact there was an event.
<br>2.ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
<br>3.Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
<br>4.ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets
<br> True Positive Rate (Recall) = True Positives / (True Positives + False Negatives)

<br>Positive Predictive Power (Precision) = True Positives / (True Positives + False Positives)

![image](https://github.com/user-attachments/assets/8fc7c1e5-0bdf-4437-a18b-79e0cdbb6a68)
# How to interpret Formula behind Linear Regression with Cost function vs normal Linear Regression Function
![image](https://github.com/user-attachments/assets/af9f7e41-a43e-4327-bbda-4c6a7874ec61)
![image](https://github.com/user-attachments/assets/87efb72b-54a0-4e84-b9ca-6414a5de0031)
![image](https://github.com/user-attachments/assets/fd471460-ce8d-43fc-8e8d-2d63ebade677)
![image](https://github.com/user-attachments/assets/65a6ed27-68f4-47c7-9f64-7bbb08817d32)





# Next Steps/ Cost effect of misclassification on AI Models
I will try to play with few other methods for fixing Imbalance datasets such as Bagging,Boosting etc
Deploy this in Azure Pipeline in Azure data platform ?
Use the same method for traning Image classification (Neural Network based training) and see if it provides good performance metrics
ML oversampling methods such as SMOTE won't probably work with Neural Network so  An alternative to SMOTE is to obtain synthetic points using generative models such as Variational AutoEncoders. Generative models have demonstrated enormous potential when handling complex data distributions during the last few years. Their success at generating realistic data makes them a new paradigm to solve dataset oversampling.
The long tail problem in image classification refers to the imbalance in data distribution where a few classes have many samples, while most classes have few samples. This is a common issue in real-world image recognition problems, such as medical imaging





