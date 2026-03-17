import pandas as pd

#读取数据
test_data = pd.read_csv('../data/test.csv')
train_data = pd.read_csv('../data/train.csv')

#观察数据
print(train_data.head(),
      test_data.head(),
      train_data.describe(),
      test_data.describe())

#保留数值，除去空值栏
numeric_data = train_data.select_dtypes(include=['int64','float64'])
numeric_data=numeric_data.dropna(axis=1)
print(list(numeric_data.columns))
