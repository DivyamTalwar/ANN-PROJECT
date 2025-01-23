# src/predict.py
import pickle
import pandas as pd
from src.model import create_model
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_data, preprocess_data
import warnings
warnings.filterwarnings("ignore")

def predict(input_data):
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    with open('models/label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('models/onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    input_data_df = pd.DataFrame([input_data])
    processed_data, _, _ = preprocess_data(input_data_df)

    processed_data_scaled = scaler.transform(processed_data)

    model = load_model('models/model.h5')

    prediction = model.predict(processed_data_scaled)

    # Apply threshold to convert the probability to binary (0 or 1)
    if prediction[0][0] > 0.5:
        return 1
    else:
        return 0
