from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier
import pandas as pd
from src.model import create_model
from src.data_preprocessing import load_data, preprocess_data, scale_data, save_objects
import warnings

warnings.filterwarnings("ignore")

# Applying Hyperparameter Tuning to Fine-tune our model
def train():
    data = load_data('data/Churn_Modelling.csv')
    data, label_encoder_gender, onehot_encoder_geo = preprocess_data(data)

    X = data.drop('Exited', axis=1)  # Independent features
    y = data['Exited']  # Dependent feature

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale data
    X_train, X_test, scaler = scale_data(X_train, X_test)

    # Save preprocessing objects
    save_objects(label_encoder_gender, onehot_encoder_geo, scaler)

    model = KerasClassifier(layers=1,neurons=32,build_fn=create_model,input_shape=(X_train.shape[1],),verbose=1)

    param_grid = {
        'neurons': [16, 32, 64, 128],
        'layers': [1, 2],
        'epochs': [50, 100]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)
    grid_result = grid.fit(X_train, y_train)

    print(f'Best Parameters: {grid_result.best_params_}')

    # Save the best model
    grid.best_estimator_.model.save('models/model.h5')
