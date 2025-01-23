import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")

def create_model(input_shape, neurons=32, layers=1):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(input_shape,)))  # Input layer + first hidden layer

    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu')) #Adding Additional hidden layers passed as parameter

    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)  # Adam optimizer with a specified learning rate
    loss = tensorflow.keras.losses.BinaryCrossentropy()  # Binary Crossentropy loss function
    
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    return model
