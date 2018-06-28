from keras.models import Model
from keras.layers import Input, Dense, concatenate, Activation, add
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D, Permute, Dropout, LSTM,\
    Bidirectional, Flatten, Reshape
from keras.callbacks import Callback
from keras.regularizers import l2


def generate_model(input_length):
    ip = Input(shape=[input_length, 1])
    #ip = Input(shape=(input_length,))
    #lstm_input = Permute((2, 1))(ip)

    y = LSTM(400, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(ip)
    #y = LSTM(512, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(y)
    y = LSTM(400, dropout=0.1, recurrent_dropout=0.1)(y)

    out = Dense(1, activation='sigmoid', name='output')(y)

    model = Model(inputs=[ip], outputs=[out])

    model.summary()

    return model

