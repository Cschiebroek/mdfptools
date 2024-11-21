# models/neural_network.py
import tensorflow as tf
from tensorflow.keras import layers, models

class NeuralNetworkModel:
    def __init__(self, input_shape):
        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        # Convert X_train and y_train to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)

        # If validation data is provided, ensure it's in the correct format
        if validation_data is not None:
            val_X, val_y = validation_data
            val_X = tf.convert_to_tensor(val_X)
            val_y = tf.convert_to_tensor(val_y)
            validation_data = (val_X, val_y)

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data,verbose=0)


    def predict(self, X_test):
        # Convert X_test to a TensorFlow tensor
        X_test = tf.convert_to_tensor(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test)

        return y_pred.flatten()
    
    def set_model_seed(self, seed):
        tf.random.set_seed(seed)

class NeuralNetworkModelGregstyle(NeuralNetworkModel):
    def __init__(self, input_shape):
        M = input_shape[0]
        self.model = models.Sequential([
            layers.Dense(M, activation='relu', input_shape=input_shape),
            layers.Dense(M//4, activation='relu'),
            layers.Dense(M//2, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
