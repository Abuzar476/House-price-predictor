import loader
import model
import preprocessor
def main():
    filepath = input("Enter file path: ")
    df = loader.load_csv(filepath)
    if df is None:
        return
    x_train, x_test, y_train, y_test = preprocessor.preprocess(df)
    trained_model = model.train_model(x_train, x_test, y_train, y_test)
    households = float(input("Enter housholds: "))
    median_income = float(input("Enter median income: "))
    total_rooms = float(input("Enter total rooms: "))
    total_bedrooms = float(input("Enter total bedrooms: "))
    housing_median_age = float(input("Enter house median age: "))
    predicted_price = model.predict_price(trained_model, total_rooms, households,
                                        median_income, housing_median_age, total_bedrooms)
    asking_price = float(input("Write your asking price: "))
    if asking_price > predicted_price:
        print("House is overpriced!")
    elif asking_price < predicted_price:
        print("This is a good deal!")
    else:
        print("This is fairly priced.")
if __name__ == "__main__":
    main()

