import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


def load_data(file_path):
    return pd.read_csv(file_path)

#Applying Feature Engineering On The Given Data
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode 'Gender' with LabelEncoder
    label_encoder_gender = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

    # OneHot encode 'Geography'
    onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
    geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

    return data, label_encoder_gender, onehot_encoder_geo


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler

#Saving All the created objects as pickle(.pkl) file for future use
def save_objects(label_encoder_gender, onehot_encoder_geo, scaler):
    
    if not os.path.exists('models'):
        os.makedirs('models')

    with open('models/label_encoder_gender.pkl', 'wb') as file:
        pickle.dump(label_encoder_gender, file)

    with open('models/onehot_encoder_geo.pkl', 'wb') as file:
        pickle.dump(onehot_encoder_geo, file)

    with open('models/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
