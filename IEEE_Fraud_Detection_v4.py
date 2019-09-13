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
TEMP=TrainTransaction[['ProductCD','QuantileAmt','isFraud']].groupby(by=['QuantileAmt','ProductCD'], as_index=False).mean()
TEMP.plot(kind='col', x='isFraud')
TEMP.sort_values(by='isFraud', ascending=False)
TrainTransaction[(TrainTransaction['ProductCD']=='C') & (TrainTransaction['QuantileAmt']=='Q25')]


sns.distplot(TrainTransaction[['ProductCD','QuantileAmt']])











##Create Predictions based on C values
#Train and test sets
TempTrain=TrainTransaction[np.concatenate((["C"+str(X) for X in [1,2,3,5,6,7,11,12]],["D"+str(X) for X in range(1,16)]))]
TempTrain.fillna(value=0, inplace=True)
TempTrain.head()
X_train, X_test, y_train, y_test = train_test_split(TempTrain, TrainTransaction['isFraud'], test_size=0.1, random_state=42)

#Set up SDG Model
SDGModel=SGDClassifier(loss="log", penalty="l2", max_iter=1000)
SDGModel.fit(X_train, y_train)

#Predict values
PredictedValues=SDGModel.predict(X_test)

#Metrics
print(confusion_matrix(y_test, PredictedValues))
print(classification_report(y_test, PredictedValues))

#Save Parameters
text_file = open("Params_V3.txt", "w")
text_file.write("%s\n" % SDGModel.get_params())
text_file.write("%s\n" % confusion_matrix(y_test, PredictedValues))
text_file.write("%s\n" % classification_report(y_test, PredictedValues))
text_file.close()

#Try with test
TestSet_dev=pd.read_csv(zip.ZipFile('Data/test_transaction.csv.zip').open("test_transaction.csv"))
X_test_dev=TestSet_dev[np.concatenate((["C"+str(X) for X in [1,2,3,5,6,7,11,12]],["D"+str(X) for X in range(1,16)]))]
X_test_dev.shape
X_test_dev.dropna().shape
X_test_dev.fillna(value=0, inplace=True)

##################
#Submit predictions
PredictedValues_Dev=SDGModel.predict(X_test_dev)

#Generate file
SubmitResults=pd.DataFrame(data={'TransactionID':TestSet_dev['TransactionID'], 'isFraud':PredictedValues_Dev})
SubmitResults.head()
SubmitResults.to_csv(path_or_buf='SubmitResults_V3.csv',index=False)

#Submit through API
import os
RE=False
if RE==True:
    os.system('kaggle competitions submit -c ieee-fraud-detection -f SubmitResults.csv -m "V3 Submission from API"')
pass
