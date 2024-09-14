import argparse
import pandas as pd
import pickle

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    predictions = model.predict(data)
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict using a scikit-learn model')
    parser.add_argument('model_file', type=str, help='Path to the .pkl file containing the model')
    parser.add_argument('data_file', type=str, help='Path to the CSV file containing input data')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_file)

    # Read the input data
    data = pd.read_csv(args.data_file)

    # Predict
    predictions = predict(model, data)

    # Print the predictions
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
