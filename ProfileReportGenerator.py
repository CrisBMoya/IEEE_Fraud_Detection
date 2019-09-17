import pandas_profiling
import pandas as pd
import zipfile as zip

DF=pd.read_csv(zip.ZipFile('Data/train_transaction.csv.zip').open("train_transaction.csv"))
DF.profile_report().to_file(output_file="ProfileReport.html")