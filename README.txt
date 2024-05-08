Ensure you have these libraries on your device:
pandas, numpy, matplot, scipy, sklearn
You need to have A2_Data_files under the same directory with the 4 code files

1. model_selection.ipynb
Read file and do some data pre-processing,
Prepare data which will be used to do general tests,
Set a baseline using ZeroR model,
Test different train sets with different feature configuration on 5 common machine learning models,
Draw a clustered bar chart to visualise.

2. feature_selection.ipynb
Use RFE to capture the most important features,
Visualise the accuracies when trained by various amount of features,
With the features selected, add in different combinations of text features to test accuracy,
finds that title and production companies can help improve a little in behaviour
The final block is used to generate test_prediction.csv for kaggle submission


3. self_training.ipynb
The release year in unlabelled data needs pre-processing
First, using all unlabelled data, combined with varying amount of labelled data,
Use several models to do self training and visualise the accuracy trend,
Then, using all labelled data, combined with varying amount of unlabelled data,
Use several models to do self training and visualise the accuracy trend.

4. evaluation&adjust_param.ipynb
Use GridSearchCV to tune hyperparameters for DT and RF,
You may need to wait for several hours to get the optimal setting of parameters,
Using the tuned parameters, test the model on evaluate set and compare with the former outcome.
