import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPool1D, Dense, Flatten
from tensorflow.keras import Model


class ResidualUnit(Model):
  def __init__(self, filter_in, filter_out, kernel_size):
    super(ResidualUnit, self).__init__()
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv1 = tf.keras.layers.Conv1D(filter_out, kernel_size, padding="same")

    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv1D(filter_out, kernel_size, padding="same")

    if filter_in == filter_out:
      self.identity = lambda x: x
    else:
      self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding="same")

  def call(self, x, training=False, mask=None):
    h = self.bn1(x, training=training)
    h = tf.nn.relu(h)
    h = self.conv1(h)

    h = self.bn2(h, training=training)
    h = tf.nn.relu(h)
    h = self.conv2(h)
    return self.identity(x) + h


class ResnetLayer(Model):
  def __init__(self, filter_in, filters, kernel_size):
    super(ResnetLayer, self).__init__()
    self.sequnce = list()
    for f_in, f_out in zip([filter_in] + list(filters), filters):
      self.sequnce.append(ResidualUnit(f_in, f_out, kernel_size))

  def call(self, x, training=False, mask=None):
    for unit in self.sequnce:
      x = unit(x, training=training)
    return x


class ResNet18(tf.keras.Model):
  def __init__(self):
    super(ResNet18, self).__init__()
    self.conv1 = Conv1D(filters=64, kernel_size=7, strides=2, padding=3, use_bias=False)
    self.bn1 = BatchNormalization()
    self.maxpool = MaxPool1D(pool_size=3, strides=2, padding=1)

    self.res1 = ResnetLayer(8, (16, 16), (3, 3))
    self.pool1 = MaxPool1D((2,2))

    self.res2 = ResnetLayer(16, (32, 32), (3, 3))
    self.pool2 = MaxPool1D((2,2))

    self.res3 = ResnetLayer(32, (64, 64), (3, 3))

    self.flatten = Flatten()
    self.dense1 = Dense(128, activation="relu")
    self.dense2 = Dense(10, activation="softmax")

  def call(self, x, training=False, mask=None):
    x = self.conv1(x)
    x = self.bn1(x)
    x = tf.nn.relu(x)
    x = self.maxpool(x)

    x = self.res1(x, training=training)
    x = self.pool1(x)
    x = self.res2(x, training=training)
    x = self.pool2(x)
    x = self.res3(x, training=training)

    x = self.flatten(x)
    x = self.dense1(x)
    return self.dense2(x)