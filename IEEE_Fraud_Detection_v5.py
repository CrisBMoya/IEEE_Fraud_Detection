%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import plotly as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import pandas_profiling

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

#Load Training data
TrainTransaction=pd.read_csv(zip.ZipFile('Data/train_transaction.csv.zip').open("train_transaction.csv"))
TestSet_dev=pd.read_csv(zip.ZipFile('Data/test_transaction.csv.zip').open("test_transaction.csv"))

#Generate quantiles
BinNum=20
TrainTransaction['QuantileAmt']=pd.qcut(x=TrainTransaction['TransactionAmt'], q=BinNum, labels=['Q'+str(X) for X in range(1,(BinNum+1))])
TestSet_dev['QuantileAmt']=pd.qcut(x=TestSet_dev['TransactionAmt'], q=BinNum, labels=['Q'+str(X) for X in range(1,(BinNum+1))])


############################################################
############################################################
############################################################
############################################################
ColumnSelect=np.asarray(["C"+str(X) for X in range(1,15)])
TempTrain=TrainTransaction[ColumnSelect]
TempTrain=TempTrain.join([pd.get_dummies(data=TrainTransaction["ProductCD"]), pd.get_dummies(data=TrainTransaction["P_emaildomain"]), pd.get_dummies(data=TrainTransaction["QuantileAmt"])])

#Train and test sets
X_train, X_test, y_train, y_test = train_test_split(TempTrain, TrainTransaction['isFraud'], test_size=0.1, random_state=42)

#Set up SDG Model with Grid Search
LGBMModel=LGBMClassifier()
LGBMModel.fit(X_train, y_train)

#Predict
Predictions=LGBMModel.predict(TempTrain)

#Metrics
print(confusion_matrix(y_test, Predictions))
print(classification_report(y_test, Predictions))

#Save Parameters
text_file = open("Params_V5.txt", "w")
text_file.write("%s\n" % confusion_matrix(y_test, Predictions))
text_file.write("%s\n" % classification_report(y_test, Predictions))
text_file.close()

#Try with test
TestSet_dev_Temp=TestSet_dev[ColumnSelect]
TestSet_dev_Temp=TestSet_dev_Temp.join([pd.get_dummies(data=TestSet_dev["ProductCD"]), pd.get_dummies(data=TestSet_dev["P_emaildomain"]), pd.get_dummies(data=TestSet_dev["QuantileAmt"])])
TestSet_dev_Temp.columns.values
TestSet_dev_Temp.drop(columns='scranton.edu', inplace=True)
TempTrain.columns.values
##################
#Submit predictions
PredictedValues_Dev=LGBMModel.predict(TestSet_dev_Temp)

#Generate file
SubmitResults=pd.DataFrame(data={'TransactionID':TestSet_dev['TransactionID'], 'isFraud':PredictedValues_Dev})
SubmitResults.head()
SubmitResults.to_csv(path_or_buf='SubmitResults_V5.csv',index=False)

#Submit through API
import os
RE=True
if RE==True:
    os.system('kaggle competitions submit -c ieee-fraud-detection -f SubmitResults_V5.csv -m "V5 Submission from API with EDA"')
pass