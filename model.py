from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
import tensorflow as tf

class PolicyModel(Model):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.d1 = Dense(10, activation='relu', name='128-layer')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(2)
        self.sm = tf.keras.layers.Softmax()
        print(len(self.d1.trainable_variables))

    def call(self, x) -> tf.Tensor:
        # x = self.input_layer(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        # print('before softmax:' + str(x))
        x = self.sm(x)

        return x


class ValueModel(Model):
    def __init__(self):
        super(ValueModel, self).__init__()
        self.d1 = Dense(10, activation='relu', name='128-layer')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1)
        print(len(self.d1.trainable_variables))

    def call(self, x) -> tf.Tensor:
        # x = self.input_layer(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x