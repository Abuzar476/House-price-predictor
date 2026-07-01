import pandas as pd
from sklearn.ensemble import RandomForestRegressor
def train_model( x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    return model
def predict_price(model, total_rooms, households,
                  median_income, housing_median_age,total_bedrooms):
    input_data = pd.DataFrame([[total_rooms, households, median_income, #double brackets cuz we need list of rows
                            housing_median_age, total_bedrooms]],
                            columns=['total_rooms','households','median_income'
                                    ,'housing_median_age','total_bedrooms'])
    predicted_price = model.predict(input_data)[0]
    print(predicted_price)
    return predicted_price
