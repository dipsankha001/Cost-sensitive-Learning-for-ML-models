# Cost-sensitive-Learning-for-ML-models


#**😇** **Motivation**

1. I dealt with various ML modeling dorectly and indirectly in my current and previous roles. Imbalanced data has always been an issue for ML model training. In a balanced dataset, the number of Positive and Negative labels is about equal. However, if one label is more common than the other label, then the dataset is imbalancedYou might think you got good, historical traning data but removing bias was always an issue.
# 2. Cost-sensitive learning is a subfield of machine learning that takes the costs of prediction errors (and potentially other costs) into account when training a machine learning model. It is a field of study that is closely related to the field of imbalanced learning that is concerned with classification on datasets with a skewed class distribution. As such, many conceptualizations and techniques developed and used for cost-sensitive learning can be adopted for imbalanced classification problems
3. I knew 1-2 approaches on how to solve this but I wanted to know modern approache and what is true cost/penalty for  selecting  imbalanced datasetMy current role deals with data pipeline,observability,tennacy, reliability of data infrastructure and AIOPs is becoming more nad more prominent in those areas. ML and AL has its differences but the core algorithms and basic steps of deployment are same. The MLOPS cycle is(Data collection->Clean->Train/test->validate->feature store reuse->deploy into CI CD pipeline->Monitor->Govern->Reuse/Analyze metrics)
4. This is why I wanted to understand how to take care of this issue and what is the impact/cost/penalty of not solving this. This is why I wanted to do a small project on ML.
5. Moreover I wanted to know the mathametical equation behind solving imbalanced data, the effect on various algorithms,ensembled learning and hyperparameter tuning which I was out of touch for a while
6. My next goal is to test this model with some new cases beyong credit risk analysis, build MLOPS pipeline in Azure data environment and then see how it works on GEN AI(Not sure if it can be deployed from normal PC without much GPU support)



# What is this Project About
1. This is a classification based Project. Classification is a predictive modeling technique which involves predicting a class label for a given observation.An observation from the domain (input) and an associated class label (output).A common example of classification comes with detecting spam emails. To write a program to filter out spam emails, a computer programmer can train a machine learning algorithm with a set of spam-like emails labelled as spam and regular emails labelled as not-spam
2. This project is based on an udemy course of Data Scientist and owner of Soledid Galli. She is a great teacher,contributor,instructor who is able to break down complex ML issues from scratch.She starts with  and then describes and concludes with
3. However since I was interested on just the cost sensitive ML training, i focused on that part and this github project is based on cost sensitive ML training part only
4. In this project  I will use historical data set of Bank to predict whether a user they will default or not
5. Imbalanced classification refers to a classification predictive modeling problem where the number of examples in the training dataset for each class label is not balanced.
6. Example : for my project if the vast majority of the users  become marked in the "Not-default” class then it is imbalanced dataset issue
7. Imbalanced datasets are those where there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class(User will Default) to the majority class(User will not default)
8. Some of the top-ranked machine learning algorithms for Classification are:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
K-Nearest Neighbors (KNN)
# How I worked on this project
1. I started with dataset download from IEEE,Kaggle etc sources followed by intensive data cleaning to remove unwanted features, duplicates, treating null values etc
2. There is 3 ways to get cost of mis classification. 

Domain Expert provides the cost(Not needed for this course)
Balance Ratio (code is in notebook)
Cross-validation: find cost as hyper-parameter (code is in notebook)
3. Tried to use this cost sensitive model into different algorithms for single use case (Credit Default Risk Analysis) to check whether incorporating cost sensitive training improve ROC/AUC,Preision/Recall etc metrics of ML Models
4. Worked on another approach with Meta Label



