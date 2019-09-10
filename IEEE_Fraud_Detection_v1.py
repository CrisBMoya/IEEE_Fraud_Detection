%config Completer.use_jedi = False

import pandas as pd
import zipfile as zip

TrainData=pd.read_csv(zip.ZipFile('Data/train_transaction.csv.zip').open("train_transaction.csv"))
