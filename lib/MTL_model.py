import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Layer
from keras.initializers import Constant
from keras import backend as K
from keras import Model
from keras import regularizers

np.random.seed(0)

# https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=4, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            bool_finite = tf.math.is_finite(y_true)
            loss += K.sum(
                precision * ((tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)) ** 2.) +
                log_var[0], axis=-1)
        return K.mean(loss)

    def call(self, inputs, **kwargs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def get_prediction_model(MDNN_hyperparams, input_size, phenotypes):
    k_reg = MDNN_hyperparams["k_reg"]
    inner_activation = MDNN_hyperparams["inner_activation"]
    dropout = MDNN_hyperparams["dropout"]  # dropout rate
    hidden_sizes_shared = MDNN_hyperparams["hidden_sizes_shared"]
    hidden_sizes_separate = MDNN_hyperparams["hidden_sizes_separate"]

    shared_layers = []
    separate_layers = {}

    main_inputs = Input(shape=(input_size,), name='main_inputs')

    # shared layers
    for i, s in enumerate(hidden_sizes_shared):
        if i == 0:
            shared_layers.append(Dense(s, activation=inner_activation,
                                       kernel_regularizer=regularizers.l2(k_reg),
                                       kernel_initializer='he_uniform',
                                       name="Shared_dense_%i" % i)(main_inputs))
        else:
            shared_layers.append(Dense(s, activation=inner_activation,
                                       kernel_regularizer=regularizers.l2(k_reg),
                                       kernel_initializer='he_uniform',
                                       name="Shared_dense_%i" % i)(shared_layers[-1]))
        shared_layers.append(Dropout(dropout, name="Shared_dropout_%i" % i)(shared_layers[-1]))

    # task-specific layers
    for p in phenotypes:
        separate_layers[p] = []
        for i, s in enumerate(hidden_sizes_separate):
            if i == 0:
                separate_layers[p].append(Dense(s, activation=inner_activation,
                                                kernel_regularizer=regularizers.l2(k_reg),
                                                kernel_initializer='he_uniform',
                                                name="%s_dense_%i" % (p, i))(shared_layers[-1]))
            else:
                separate_layers[p].append(Dense(s, activation=inner_activation,
                                                kernel_regularizer=regularizers.l2(k_reg),
                                                kernel_initializer='he_uniform',
                                                name="%s_dense_%i" % (p, i))(separate_layers[p][-1]))
            separate_layers[p].append(Dropout(dropout, name="%s_dropout_%i" % (p, i))(separate_layers[p][-1]))
        separate_layers[p].append(Dense(1, activation='linear', name='%s_out' % p)(separate_layers[p][-1]))

    final_model = Model(inputs=[main_inputs], outputs=[separate_layers[p][-1] for p in phenotypes])

    return final_model


def get_trainable_model(prediction_model, input_size):
    inp = Input(shape=(input_size,), name='main_inputs')
    y1_pred, y2_pred, y3_pred, y4_pred = prediction_model([inp])
    y1_true = Input(shape=(1,), name="y1_true")
    y2_true = Input(shape=(1,), name="y2_true")
    y3_true = Input(shape=(1,), name="y3_true")
    y4_true = Input(shape=(1,), name="y4_true")
    out = CustomMultiLossLayer(nb_outputs=4)([y1_true, y2_true, y3_true, y4_true, y1_pred, y2_pred, y3_pred, y4_pred])

    train_model = Model(inputs=[inp, y1_true, y2_true, y3_true, y4_true], outputs=out)
    return train_model
