from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
import tensorflow as tf

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(128, activation='relu', name='128-layer')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(2, activation='softmax')
        print(len(self.d1.trainable_variables))

    def call(self, x) -> tf.Tensor:
        # x = self.input_layer(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

