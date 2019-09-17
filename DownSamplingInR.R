
library(tidyverse)
library(dummies)
library(UBL)

TrainTransaction=read_delim(file='~/Data/train_transaction.csv.zip', delim=',')
TestTransaction=read_delim(file='~/Data/test_transaction.csv.zip', delim=',')

TrainTransaction$QuantileAmt=cut(x=TrainTransaction$TransactionAmt, breaks=20, labels=paste0("C",1:20))
TestTransaction$QuantileAmt=cut(x=TestTransaction$TransactionAmt, breaks=20, labels=paste0("C",1:20))

#

TempTrain=TrainTransaction[,grep(pattern='^C', x=colnames(TrainTransaction))]
TempTrain=cbind(TempTrain, dummy(TrainTransaction$ProductCD))
TempTrain=cbind(TempTrain, dummy(TrainTransaction$P_emaildomain))
TempTrain=cbind(TempTrain, dummy(TrainTransaction$QuantileAmt))

TempTrain$isFraud=as.factor(TrainTransaction$isFraud)

TrainNorm=ENNClassif(form=isFraud~., dat=TempTrain)
