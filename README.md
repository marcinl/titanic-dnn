# titanic-dnn

This is a simple pytorch implementation of DNN model training and evaluating Kaggle Titanic dataset:
http://www.kaggle.com/c/titanic-gettingStarted

The used model imports data and corrects following columns values:
- AGE - median is being used for missing values
- EMBARKED - missing values are replaced with most freqenltly missing category
following columns are not being used at all:
- PASSENGER ID - does not provide any value
- TICKET - type of ticket
- CABIN - the type of cabin allocated

Current performance on test set data is:
- Accuracy - ~85%
- Loss - 6.2%

