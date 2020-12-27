from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
import tensorflow as tf


class PolicyModel(Model):
    def __init__(self, outputs):
        super(PolicyModel, self).__init__()
        self.d1 = Dense(10, activation='relu', name='128-layer')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(outputs)
        print(len(self.d1.trainable_variables))

    def call(self, x) -> tf.Tensor:
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return x


class ValueModel(Model):
    def __init__(self):
        super(ValueModel, self).__init__()
        self.d1 = Dense(10, activation='relu', name='128-layer')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1)
        print(len(self.d1.trainable_variables))

    def call(self, x) -> tf.Tensor:
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x


def create_q_model():
    d0 = Input(shape=(5,))
    d1 = Dense(64, activation='relu', name='input')(d0)
    d2 = Dense(64, activation='relu')(d1)
    d3 = Dense(64, activation='relu')(d2)
    d5 = Dense(1)(d3)
    return tf.keras.Model(outputs=d5, inputs=d0)

