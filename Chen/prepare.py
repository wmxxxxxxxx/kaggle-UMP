import pandas

data = pandas.read_csv('./train.csv')
data.to_parquet('./train.parquet')