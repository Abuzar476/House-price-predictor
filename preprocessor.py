import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess(df):
    df = df.dropna()
    x = df[['total_rooms','households','median_income',
             'housing_median_age', 'total_bedrooms']]
    y = df['median_house_value']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test