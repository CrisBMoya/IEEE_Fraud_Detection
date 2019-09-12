%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import plotly as plt
import numpy as np
import plotly.express as px
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#Load Training data
TrainTransaction=pd.read_csv(zip.ZipFile('Data/train_transaction.csv.zip').open("train_transaction.csv"))
TrainIdentity=pd.read_csv(zip.ZipFile('Data/train_identity.csv.zip').open("train_identity.csv"))

#Check data
TrainTransaction.shape
TrainIdentity.shape
TrainTransaction.columns
TrainIdentity.columns

#Data Transformation
#C1 to C14 are described as a counting columns, the actual meaning are masked, so we will sum theese.
TrainTransaction["CSum"]=TrainTransaction[["C"+str(X) for X in range(1,15)]].sum(axis=1)

#Also, create a pattern for the columns by putting them together as a categorical value
for i in TrainTransaction[["C"+str(X) for X in range(1,15)]]:
    if(i=='C1'):
        CPattern=TrainTransaction[i].astype('str')
    else:
        CPattern=CPattern + '|' + TrainTransaction[i].astype('str')
pass
TrainTransaction['CPattern']=CPattern 
TrainTransaction['CPattern'].nunique()

#Delete C columns
TrainTransaction.drop(columns=["C"+str(X) for X in range(1,15)], inplace=True)

#EDA -- Exploratory Data Analysis - Cant do shit, data is too big.
_, Sampled, _, _=train_test_split(TrainTransaction, TrainTransaction["isFraud"], test_size=100/TrainTransaction.shape[0])
px.bar(data_frame=Sampled, x="isFraud", y="CSum", color="isFraud")

#Check if a certain CPattern cause more frauds than others
CPatternFraud=TrainTransaction[['CPattern','isFraud']].groupby(by='CPattern', as_index=False).sum().sort_values(by='isFraud',ascending=False)

#Take top 10 fraudulent CPatterns and test for common patterns between those 10
Top10=CPatternFraud.iloc[0:10,]
Splitted=pd.DataFrame(Top10['CPattern'].str.split('|').tolist(), columns=["C"+str(X) for X in range(1,15)])
Splitted=Splitted.apply(lambda x: x.astype(float))
Splitted['SumFraud']=Top10['isFraud'].values

#There must be a esier way to do this, I just do this and works so...
FrequencyPattern=[]
for i in ["C"+str(X) for X in range(1,15)]:
    FrequencyPattern.append(Splitted[i].value_counts()*100/10)
pass

TEMPList=[]
for i in range(0,len(FrequencyPattern)):
    if len(FrequencyPattern[i])==1:
        TEMPList.append(FrequencyPattern[i].name + '|' + str(FrequencyPattern[i].index[0]) + "|" + str(FrequencyPattern[i].values[0]))
    else:
        for x in range(0,len(FrequencyPattern[i])):
            TEMPList.append(FrequencyPattern[i].name + '|' + str(FrequencyPattern[i].index[x]) + '|' + str(FrequencyPattern[i].values[x]))
pass
TEMPList=np.asarray(TEMPList)
FreqDF=pd.DataFrame(pd.DataFrame(TEMPList)[0].str.split('|').tolist(), columns=['C','Value','Frequency'])
FreqDF['Frequency']=FreqDF['Frequency'].astype('float')
MostPresservedPatterns=FreqDF[FreqDF['Frequency']>70]
MostPresservedPatterns

##Create Predictions based on theese values: C1 C2 C3 C5 C6 C7 C11 C12
#Train and test sets
X_train, X_test, y_train, y_test = train_test_split(TrainTransaction[["C"+str(X) for X in [1,2,3,5,6,7,11,12]]], TrainTransaction['isFraud'], test_size=0.3, random_state=42)

#Set up SDG Model with Grid Search
Parameter_Grid={'loss':['log','hinge','modified_huber','squared_hinge','perceptron','squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'], 'penalty':['none','l2','l1','elasticnet']}
Grid_Search=GridSearchCV(estimator=SGDClassifier(), param_grid=Parameter_Grid, verbose=3, refit=True)
Grid_Search.fit(X_train, y_train)
Grid_Search.best_params_
Grid_Search.best_estimator_

#Improve SDG model results
Grid_Predictions=Grid_Search.predict(X_test)

#Metrics
print(confusion_matrix(y_test, Grid_Predictions))
print(classification_report(y_test, Grid_Predictions))

#Save Parameters
text_file = open("Params_V2.txt", "w")
text_file.write("%s\n" % SDGModel.get_params())
text_file.write("%s\n" % confusion_matrix(y_test, Grid_Predictions))
text_file.write("%s\n" % classification_report(y_test, Grid_Predictions))
text_file.close()

#Try with test
TestSet_dev=pd.read_csv(zip.ZipFile('Data/test_transaction.csv.zip').open("test_transaction.csv"))
X_test_dev=TestSet_dev[["C"+str(X) for X in [1,2,3,5,6,7,11,12]]]
X_test_dev.shape
X_test_dev.dropna().shape
X_test_dev.fillna(value=0, inplace=True)


##################
#Submit predictions
PredictedValues_Dev=SDGModel.predict(X_test_dev)

#Generate file
SubmitResults=pd.DataFrame(data={'TransactionID':TestSet_dev['TransactionID'], 'isFraud':PredictedValues_Dev})
SubmitResults.head()
SubmitResults.to_csv(path_or_buf='SubmitResults.csv',index=False)

#Submit through API
RE=True
if RE==True:
    os.system('kaggle competitions submit -c ieee-fraud-detection -f SubmitResults.csv -m "V2 Submission from API"')
pass
