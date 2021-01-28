# News recommendation algorithm

## Solutions
For this problem, I have two solutions: supervised and unsupervised algorithms. 
1) Treat the problem as a classification problem and can solve it with supervised algorithms 
    (such as decision tree and SVM). 
2) Treat the problem as an outlier detection problem and can solve it with unsupervised algorithms 
    (such as OCSVM and AutoEncoder)
    
In this work, I solve the problem with supervised algorithms (i.e., Decision Tree and Multilayer perceptron (MLP))
 and mainly focus on MLP implemented by Pytorch. 

      
## MLP-based model
    My whole purpose is to illustruate how to solve the problem with MLP (not to find the optimal soluation), 
    so the architecture is quite simple (3 layers) and I don't tune parameters. 
    model = Sequential(
    (0): Linear(in_features=97, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=2, bias=True)
    (5): Sigmoid()

## Data preprocessing
    Due to the limit resource of my laptop, I don't use the whole data. I parse the first 10,000 lines to build and test the model.
    To avoid the imblanced effect to the performance, undersampling techniquce is used to random sample a part of larger class.
    
## Model performance
    AUC: ~84%
    Accuracy: ~ 75%
    classifier report:
              precision    recall  f1-score   support

           0       0.75      0.76      0.75      1413
           1       0.76      0.74      0.75      1428

    accuracy                           0.75      2841
    macro avg      0.75      0.75      0.75      2841
    weighted avg   0.75      0.75      0.75      2841

## Model deployment
There are many tools and services (e.g., flask) that can help us to deploy our model. 
Here I list the main steps for deploying a model using flask:
1) Training a machine learning model on a local system.
2) Wrapping the inference logic into a flask application.
3) Using docker to containerize the flask application.
4) Hosting the docker container on an AWS ec2 instance and consuming the web-service.


## Model monitoring
1) Model input monitoring
    Check if input values fall within an allowed set or range
    Check missing values 
    Check outlier values
2) Model prediction monitoring
    Compare model prediction distributions with statistical metrics:
    such as, median, mean, standard deviation, and max/min values
3) System performance monitoring 
    Check the latency, IO/Memory/Disk Utilisation, etc.
    
 
 
 
# Second questions:
Given the following conditions: 
1) Two groups are split randomly (i.e., all group draw data from the same distribution)
2) Each group random splits the users as test users and holdout users 
3) The ratios between test users and holdout users for the two groups are the same
4) Each group has a larger sample (users) to ensure that the data is not biased.
4) There no special events happen  (e.g., Cyber Monday) 
    Because on Cyber Monday, consumers are more likely to complete purchases in any way, regardless of their 
    exposure to an advertisement. 
5) Evaluate the same product (i.e., regardless of the impact caused by different brands)
    E.g., some products are more familiar to consumers than others.

If all the conditions are met, the answer is yes, i.e., the new marketing methodology will provide an increase in 
incremental conversions over the old methodology.

Note: 
    The calculation of statistical significance is subject to a certain degree of error, which could mislead the conclusion.
    A statistically significant result cannot prove that a research hypothesis is 100% correct.

