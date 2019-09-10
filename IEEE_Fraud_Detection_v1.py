%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import tensorflow as tf
import plotly as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

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
TrainTransaction.drop(columns=["C"+str(X) for X in range(1,15)], inplace=True)
TrainTransaction[["D"+str(X) for X in range(1,16)]].head()
TrainTransaction["isFraud"]=TrainTransaction["isFraud"].astype("category")


#EDA -- Exploratory Data Analysis
Sampled=TrainTransaction.sample(n=100)
px.bar(data_frame=Sampled, x="isFraud", y="CSum", color="isFraud")

#
X_train, X_test, _, _ = train_test_split(TrainTransaction, TrainTransaction["isFraud"], test_size=0.3, random_state=101)
X_test.shape

Temp=TrainTransaction
for i in range(0,10):
    _, Temp, _, _=train_test_split(Temp, Temp["isFraud"], test_size=0.3)
pass
Temp.shape    
Temp
TrainTransaction.shape[0]*(0.1**10)
0.1**x=10
0.1**2.3
np.log(10)
