import pandas as pd
import numpy
import sklearn
import lightgbm
from sklearn.linear_model import LinearRegression


full_path = r'C:\Users\astha\Downloads\Tariff Calculations plus Population.csv'
test_path = r'C:\Users\astha\Downloads\fake_tariffs_500.csv'

train_df = pd.read_csv(full_path, sep=';', usecols=["Country", "Trump Tariffs Alleged", "Trump Response"])

test_df = pd.read_csv(test_path)
train_df["Trump Tariffs Alleged"] = train_df["Trump Tariffs Alleged"].str.rstrip('%').astype(float)
train_df["Trump Response"] = train_df["Trump Response"].str.rstrip('%').astype(float)


x_axis = train_df[["Trump Tariffs Alleged"]]
y_axis = train_df[["Trump Response"]]

model = LinearRegression()
model.fit(x_axis, y_axis)

slope = model.coef_[0]
y_intercept = model.intercept_

test_df["Trump Response"] = model.predict(test_df[["Trump Tariffs Alleged"]])

test_df.to_csv(r'C:\Users\astha\Downloads\tariffstest1.csv', index=False)
