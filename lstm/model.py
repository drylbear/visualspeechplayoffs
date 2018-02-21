from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications import ResNet50 as ResNet
from collections import deque
from keras import backend as K
from keras.layers import Activation, Lambda, Input, Concatenate, Reshape, Flatten, concatenate
from keras.layers import Input, Dense, Activation, Dropout, Conv3D, MaxPooling3D, Flatten,ZeroPadding3D, \
    TimeDistributed, SpatialDropout3D,BatchNormalization, Lambda, GRU, SpatialDropout1D
from keras.layers import concatenate


def get_model(features_length, image_shape, len_classes, seq_length, model_name, optimizer_name, learning_rate, decay,
              lstm_units, lstm_dropout, dense_dropout, dense_units, resnet=False):
    """Convenience getter for our model with all necessary parameters

    :param features_length: The length of the feature we input
    :param image_shape: The image shape
    :param len_classes: The length of the classes we want
    :param seq_length: The length of the sequence
    :param model_name: the name of the model. All models have the same first and final layers, they only differ
                        in the number of LSTM layers
                            lstm - will create a single layer LSTM model
                            two-layer - will create a two layer LSTM model
                            three-layer - will create a three layer LSTM model
    :param optimizer_name: Name of the opzimizer to choose (rmsprop - adam)
    :param learning_rate: The learning rate
    :param decay: The decay of the learning rate
    :param lstm_units: The number of units, which should be used for the LSTM layers
    :param lstm_dropout: The dropout which should be used for the LSTM layers
    :param dense_dropout: The dropout, which should be used for the dense dropout layers
    :param dense_units: The number of units, which should be used for the dense layers
    :param resnet; If true, use ResNet into LSTMs else only use LSTMs
    :return: a model wrapped with some more information we may need during training
    """
    if not resnet:
        rm = LSTMModel(lstm_units, len_classes, seq_length, features_length=features_length,
                       learning_rate=learning_rate, decay=decay, model_name=model_name, lstm_dropout=lstm_dropout,
                       optimizer_name=optimizer_name, dense_units=dense_units, dense_dpo=dense_dropout)
    else:
        rm = ResNetLSTMModel(lstm_units, len_classes, seq_length, image_shape=image_shape, decay=decay,
                             learning_rate=learning_rate, lstm_dropout=lstm_dropout, optimizer_name=optimizer_name,
                             dense_units=dense_units, dense_dropout=dense_dropout)
    return rm


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class ResNetLSTMModel:
    def __init__(self, units, nb_classes, seq_length, model_name="resnet", image_shape=(48, 48, 1),
                 optimizer_name="adam",
                 learning_rate=1e-3, decay=1e-6, lstm_dropout=0.5, dense_units=512, dense_dropout=0.5):
        self.feature_queue = deque()
        self.image_shape = image_shape
        self.seq_length = seq_length
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.decay = decay

        self.model_name = model_name + "-" + "LstmUnits_" + str(units) + "-" + "LstmDO_" + str(lstm_dropout) + "-" \
                          + "Lr_" + str(learning_rate)

        metrics = ['accuracy']

        self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])

        self.model = self.build_resnet(units, lstm_dropout, dense_units, dense_dropout)
        self.layer = 1
        self.direction = "single direction"

        self.compile(self.learning_rate)

        """if self.optimizer_name == "adam":
            optimizer = Adam(lr=self.learning_rate, decay=self.decay)
        elif self.optimizer_name == "rmsprop":
            optimizer = RMSprop(self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])"""

        print(self.model.summary())

    def build_resnet(self, units, dropout=0.5, dense_units=512, dense_dropout=0.5):
        resnet = ResNet(include_top=False, pooling=None)

        input_data = Input(name='the_input', shape=self.input_shape, dtype='float32')
        labels = Input(name='labels', shape=[self.nb_classes], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        res_list = []
        for j in range(self.seq_length):
            def slice(x):
                return x[:, j, :, :]
            inner = resnet(Lambda(slice)(input_data))
            #inner = Reshape((2048, ))(inner)
            res_list.append(inner)

        m = concatenate(res_list, axis=1)
        inner = Reshape((self.seq_length, 2048))(m)

        # cut down size going into RNN
        inner = TimeDistributed(Dense(dense_units // 16, activation='relu'))(inner)

        # start with one layer of lstm
        lstm = LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5))(inner)
        y_pred = TimeDistributed(Dense(self.nb_classes, activation="softmax"))(lstm)

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        #model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

        model = Model(inputs=input_data, outputs=y_pred)
        model.summary()

        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=optimizer)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        """model = Sequential()
        model.add(TimeDistributed(resnet, input_shape=self.input_shape))
        model.add(Reshape((-1, 2048)))
        model.summary()
        model.add(TimeDistributed(Dense(dense_units // 16, activation='relu')))
        
        model.summary()
        model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
        model.add(Dropout(dense_dropout))
        model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
        model.add(Dropout(dense_dropout))
        model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
        model.add(TimeDistributed(Dense(dense_units, activation='relu')))
        model.add(Dropout(dense_dropout))
        model.add(TimeDistributed(Dense(self.nb_classes, activation='softmax')))"""


        self.layer = 3
        self.direction = "single direction"

        return model


    def compile(self, lr=None):
        """Compile the model

            :param lr: The given learning rate, if None choose default one from model
            """
        optimizer = None
        if lr is not None:
            self.learning_rate = lr
        if self.optimizer_name == "adam":
            optimizer = Adam(lr=self.learning_rate, decay=self.decay)
        elif self.optimizer_name == "rmsprop":
            optimizer = RMSprop(self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



class LSTMModel:
    """Wraps a Keras sequential model with some further information and also compiles the model"""

    def __init__(self, units, nb_classes, seq_length, model_name="lstm", features_length=2048, optimizer_name="adam",
                 learning_rate=1e-3, decay=1e-6, lstm_dropout=0.5, dense_units=512, dense_dpo=0.5):
        """Initialze model with given parameters

        :param units: Number of units to use for the LSTM layers
        :param nb_classes: Number of classes
        :param seq_length: The sequence length
        :param model_name: the name of the model. All models have the same first and final layers, they only differ
                      in the number of LSTM layers
                            lstm - will create a single layer LSTM model
                            two-layer - will create a two layer LSTM model
                            three-layer - will create a three layer LSTM model
        :param features_length: Length of the feature, which is fed into the LSTMs
        :param optimizer_name: Name of the opzimizer to choose (rmsprop - adam)
        :param learning_rate: The learning rate
        :param decay: The decay of the learning rate (only for adam)
        :param lstm_dropout: The dropout which should be used for the LSTM layers
        :param dense_units: The dropout, which should be used for the dense dropout layers
        :param dense_dpo: The number of units, which should be used for the dense layers
        """
        self.seq_length = seq_length
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.decay = decay
        # set model name, to later save weights
        self.model_name = model_name + "-" + "LstmUnits_" + str(units) + "-" + "LstmDO_" + str(lstm_dropout) + "-" + \
                          "Lr_" + str(learning_rate)

        # Choose the wanted architecture
        self.input_shape = (seq_length, features_length)
        if model_name == "lstm":
            m, l, d = lstm(self.input_shape, self.nb_classes, units, lstm_dropout, dense_units, dense_dpo)
        elif model_name == 'two-layer':
            m, l, d = two_layer_lstm(self.input_shape, self.nb_classes, units, lstm_dropout, dense_units, dense_dpo)
        elif model_name == "three-layer":
            m, l, d = three_layer_lstm(self.input_shape, self.nb_classes, units, lstm_dropout, dense_units, dense_dpo)
        else:
            raise ValueError("Unknown model type \"{}\", choose out of (lstm, only)".format(model_name))
        self.model = m
        self.layer = l
        self.direction = d

        print(self.model.summary())

    def compile(self, lr=None):
        """Compile the model

        :param lr: The given learning rate, if None choose default one from model
        """
        optimizer = None
        if lr is not None:
            self.learning_rate = lr
        if self.optimizer_name == "adam":
            optimizer = Adam(lr=self.learning_rate, decay=self.decay)
        elif self.optimizer_name == "rmsprop":
            optimizer = RMSprop(self.learning_rate)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def two_layer_lstm(input_shape, nb_classes, units, dropout=0.5, dense_units=512, dense_dropout=0.5):
    model = Sequential()
    model.add(TimeDistributed(Dense(dense_units), input_shape=input_shape))
    model.add(LSTM(units, return_sequences=True, bias_initializer=Constant(value=5), dropout=dropout))
    model.add(Dropout(dense_dropout))
    model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
    model.add(TimeDistributed(Dense(dense_units, activation='relu')))
    model.add(Dropout(dense_dropout))
    model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

    return model, 2, "single direciton"


def three_layer_lstm(input_shape, nb_classes, units, dropout=0.5, dense_units=512, dense_dropout=0.5):
    model = Sequential()
    model.add(TimeDistributed(Dense(dense_units), input_shape=input_shape))
    model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
    model.add(Dropout(dense_dropout))
    model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
    model.add(Dropout(dense_dropout))
    model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
    model.add(TimeDistributed(Dense(dense_units, activation='relu')))
    model.add(Dropout(dense_dropout))
    model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

    return model, 3, "single direciton"


def lstm(input_shape, nb_classes, units, dropout=0.5, dense_units=512, dense_dropout=0.5):
    model = Sequential()
    model.add(TimeDistributed(Dense(dense_units), input_shape=input_shape))
    model.add(LSTM(units, return_sequences=True, dropout=dropout, bias_initializer=Constant(value=5)))
    model.add(TimeDistributed(Dense(dense_units, activation='relu')))
    model.add(Dropout(dense_dropout))
    model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

    return model, 1, "single direction"
