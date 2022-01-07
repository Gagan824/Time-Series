from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
import tensorflow as tf
import os

class Model():
    def __init__(self, input_size=100, model_weight=None):

        if model_weight is None:

            print('Stacked LSTM model Loading')
            # LSTM Stacked Model
            self.model = Sequential()
            self.model.add(LSTM(50, return_sequences=True, input_shape=(input_size, 1)))
            self.model.add(LSTM(50, return_sequences=True))
            self.model.add(LSTM(50))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            print('Model Loaded Successfully')
            print('Model Summary:')
            self.model.summary()

        else:

            print('Stacked LSTM model Loading')
            self.load_model(weight=model_weight)
            print('Model Loaded Successfully')

    def summary(self, model):
        model.summary()

    def train(self, x_train, y_train, x_test, y_test, batches=30, epochs=100):

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batches, verbose=1)

        if os.path.exists('weights'):
            out_dir = 'weights'
        else:
            os.mkdir('weigths')
            out_dir = 'weights'

        model_name = 'lstm_model'

        self.save_model(output_dir=out_dir, model_name=model_name)

    def predict(self, x=None, verbose=None):
        if x is None:
            print('-----> x expected some values but got none')
            raise ValueError('x expected some values but got none')

        else:
            if verbose is None:

                predictions = self.model.predict(x)
                predictions = predictions

                return predictions

            else:

                predictions = self.model.predict(x, verbose=verbose)
                predictions = predictions

                return predictions

    def save_model(self, output_dir='', model_name='lstm_model'):
        self.model.save(f"{output_dir}/{model_name}.h5")
        print('Model saved to Weight directory with name lstm_model.h5')

    def load_model(self, weight='lstm_model.h5'):

        model = tf.keras.models.load_model(weight)
        self.model = model
        print('Model Summary:')
        self.summary(model)

        return model
