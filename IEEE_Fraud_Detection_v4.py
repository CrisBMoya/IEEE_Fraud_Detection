%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import plotly as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


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

#DEA and Feature Engineering
TrainTransaction['isFraud'].sum()
TrainTransaction.shape[0]-TrainTransaction['isFraud'].sum()

#Data Transformation
#C1 to C14 are described as a counting columns, the actual meaning are masked, so we will sum theese.
TrainTransaction["CSum"]=TrainTransaction[["C"+str(X) for X in range(1,15)]].sum(axis=1)

#Barplot of the mean
sns.barplot(data=TrainTransaction[['isFraud','CSum']].groupby(by='isFraud', as_index=False).mean(), x='isFraud', y='CSum')
#Same as this BTW:
sns.barplot(x=TrainTransaction['isFraud'], y=TrainTransaction['CSum'])

#Also, create a pattern for the columns by putting them together as a categorical value
for i in TrainTransaction[["C"+str(X) for X in range(1,15)]]:
    if(i=='C1'):
        CPattern=TrainTransaction[i].astype('str')
    else:
        CPattern=CPattern + '|' + TrainTransaction[i].astype('str')
pass
TrainTransaction['CPattern']=CPattern 

#Check if a certain CPattern cause more frauds than others
CPatternFraud=TrainTransaction[['CPattern','isFraud']].groupby(by='CPattern', as_index=False).sum().sort_values(by='isFraud',ascending=False)

#Take top 20 fraudulent CPatterns and test for common patterns between those 10
Top20=CPatternFraud.iloc[0:20,]

#Split patterns to its original form and turn values to float numbers
Splitted=pd.DataFrame(Top20['CPattern'].str.split('|').tolist(), columns=["C"+str(X) for X in range(1,15)]).apply(lambda x: x.astype(float))

Top20['isFraud'].values

#Count how many unique values repeats per C column
Splitted=Splitted.apply(axis=0, func=lambda x: pd.value_counts(x))

#Add CPattern column
Splitted['CPattern']=Splitted.index

#Metl DF
Splitted=Splitted.melt(id_vars=['CPattern']).dropna()

#Apply treshold. Values must be repeated above 90%
Splitted[Splitted['value']>9]['variable'].unique().size #That's like... all C columns... incredibly useless!
Splitted[Splitted['value']>9]['variable'].unique()


##Lets do more visualizations!

#Ammount of transaction and Fraud
sns.barplot(data=TrainTransaction[['isFraud','TransactionAmt']].groupby(by='isFraud', as_index=False).mean(), x='isFraud', y='TransactionAmt')

#See if theres a relation between ammount of transaction and fraud frequency
GroupTransactionAmmount=TrainTransaction[['isFraud','TransactionAmt']].groupby(by='TransactionAmt', as_index=False)
Top20TranAmt=GroupTransactionAmmount.count().sort_values(by='isFraud',ascending=False).iloc[0:50,].sort_values(by='isFraud',ascending=True)

#Plot -- it doesnt say much. Use quantiles to analyze more data
go.Figure(data=[go.Bar(x=Top20TranAmt['isFraud'], y=Top20TranAmt['TransactionAmt'])], layout=go.Layout(xaxis={'type': 'category'}))

#Getting quantiles -- 25?
BinNum=25
TrainTransaction['QuantileAmt']=pd.qcut(x=TrainTransaction['TransactionAmt'], q=BinNum, labels=['Q'+str(X) for X in range(1,(BinNum+1))])
GroupTransactionAmmount=TrainTransaction[['isFraud','QuantileAmt']].groupby(by='QuantileAmt', as_index=False).count()
GroupTransactionAmmount=GroupTransactionAmmount.sort_values(by='isFraud', ascending=True)
P1=go.Figure(data=[go.Bar(x=GroupTransactionAmmount['QuantileAmt'], y=GroupTransactionAmmount['isFraud'])], layout=go.Layout(xaxis={'type': 'category'}))
P1.add_trace(go.Scatter(x=['Q'+str(X) for X in range(1,(BinNum+1))], y=[GroupTransactionAmmount['isFraud'].max()/2 for X in range(0,BinNum)], mode='lines'))

#Investigate Product ID
TrainTransaction['ProductCD'].nunique() #Just 5 classes
sns.barplot(data=TrainTransaction[['isFraud','ProductCD']].groupby(by='ProductCD', as_index=False).mean(), x='isFraud', y='ProductCD')

#Cross Product ID and Price

#Create grouped object
CrossDFGroup=TrainTransaction[['ProductCD','QuantileAmt','isFraud']].groupby(by=['QuantileAmt','ProductCD'], as_index=False)

#Compute Mean
CrossDF=CrossDFGroup.mean()
#Compute non fraud frequency
CrossDF["CountNonFraud"]=CrossDFGroup.apply(lambda x: np.sum(x==0))["isFraud"]
#Compute fraud frequency
CrossDF["CountFraud"]=CrossDFGroup.apply(lambda x: np.sum(x==1))["isFraud"]
CrossDF.head()

#Plot has outliers
Fig=px.bar(data_frame=CrossDF, x="QuantileAmt", y="isFraud", color='ProductCD', barmode='group', hover_data=['CountNonFraud', 'CountFraud'])
Fig.show(renderer="browser")

#Remove outliers
CrossDF.dropna(inplace=True)
CrossDF=CrossDF[(CrossDF["CountNonFraud"]>0) & (CrossDF["CountFraud"]>0)]

#Plot again -- it seems like product C always get scammed. Also H in higher quantiles. S looks erratic.
Fig=px.bar(data_frame=CrossDF, x="ProductCD", y="isFraud", color='QuantileAmt', barmode='group', hover_data=['CountNonFraud', 'CountFraud'])
Fig.show(renderer="browser")

#P_emaildomain
GroupP=TrainTransaction[['isFraud','P_emaildomain']].groupby(by='P_emaildomain', as_index=False)
GroupPMean=GroupP.mean()
GroupPMean["CountNonFraud"]=GroupP.apply(lambda x: np.sum(x==0))["isFraud"]
GroupPMean["CountFraud"]=GroupP.apply(lambda x: np.sum(x==1))["isFraud"]

Fig=px.bar(data_frame=GroupPMean, x="P_emaildomain", y="isFraud", hover_data=['CountNonFraud', 'CountFraud'])
Fig.show(renderer="browser")


############################################################
############################################################
############################################################
############################################################
##Create Predictions based on C values
#Train and test sets
ColumnSelect=np.asarray(["C"+str(X) for X in range(1,15)])
TempTrain=TrainTransaction[ColumnSelect]
TempTrain=TempTrain.join([pd.get_dummies(data=TrainTransaction["ProductCD"]), pd.get_dummies(data=TrainTransaction["P_emaildomain"]), pd.get_dummies(data=TrainTransaction["QuantileAmt"])])
TempTrain.columns.values
X_train, X_test, y_train, y_test = train_test_split(TempTrain, TrainTransaction['isFraud'], test_size=0.1, random_state=42)

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
text_file = open("Params_V4.txt", "w")
text_file.write("%s\n" % Grid_Search.get_params())
text_file.write("%s\n" % confusion_matrix(y_test, Grid_Predictions))
text_file.write("%s\n" % classification_report(y_test, Grid_Predictions))
text_file.close()

#Try with test
TestSet_dev=pd.read_csv(zip.ZipFile('Data/test_transaction.csv.zip').open("test_transaction.csv"))
TestSet_dev['QuantileAmt']=pd.qcut(x=TestSet_dev['TransactionAmt'], q=BinNum, duplicates="drop", labels=['Q'+str(X) for X in range(1,(24+1))])

X_test_dev=TestSet_dev[ColumnSelect]
X_test_dev=X_test_dev.join([pd.get_dummies(data=TestSet_dev["ProductCD"]), pd.get_dummies(data=TestSet_dev["P_emaildomain"]), pd.get_dummies(data=TestSet_dev["QuantileAmt"])])
X_test_dev.shape
X_test_dev.dropna().shape
X_test_dev.fillna(value=0, inplace=True)

##################
#Submit predictions
PredictedValues_Dev=Grid_Search.predict(X_test_dev)

#Generate file
SubmitResults=pd.DataFrame(data={'TransactionID':TestSet_dev['TransactionID'], 'isFraud':PredictedValues_Dev})
SubmitResults.head()
SubmitResults.to_csv(path_or_buf='SubmitResults_V4.csv',index=False)

#Submit through API
import os
RE=False
if RE==True:
    os.system('kaggle competitions submit -c ieee-fraud-detection -f SubmitResults_V4.csv -m "V4 Submission from API"')
pass