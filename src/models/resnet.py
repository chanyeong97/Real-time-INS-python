import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPool1D, Dense, Flatten, Dropout
from tensorflow.keras import Model


class ResidualUnit(Model):
  def __init__(self, filter_in, filter_out, kernel_size, strides):
    super(ResidualUnit, self).__init__()
    if filter_in == filter_out:
      strides = 1
      self.identity = lambda x: x
    else:
      self.identity = tf.keras.Sequential()
      self.identity.add(Conv1D(filter_out, kernel_size=1, strides=strides, use_bias=False))
      self.identity.add(BatchNormalization())

    self.conv1 = Conv1D(filter_out, kernel_size, strides=strides, padding='valid')
    self.bn1 = BatchNormalization()

    self.conv2 = Conv1D(filter_out, kernel_size, strides=1, padding='valid')
    self.bn2 = BatchNormalization()

  def call(self, x, training=False, mask=None):
    h = tf.keras.layers.ZeroPadding1D(padding=1)(x)
    h = self.conv1(h)
    h = self.bn1(h, training=training)
    h = tf.nn.relu(h)
    
    h = tf.keras.layers.ZeroPadding1D(padding=1)(h)
    h = self.conv2(h)
    h = self.bn2(h, training=training)

    h += self.identity(x)
    out = tf.nn.relu(h)
    return out


class ResnetLayer(Model):
  def __init__(self, filter_in, filters, kernel_size, strides):
    super(ResnetLayer, self).__init__()
    self.sequnce = list()
    for f_in, f_out in zip([filter_in] + list(filters), filters):
      self.sequnce.append(ResidualUnit(f_in, f_out, kernel_size, strides))

  def call(self, x, training=False, mask=None):
    for unit in self.sequnce:
      x = unit(x, training=training)
    return x


class ResNet18(Model):
  def __init__(self):
    super(ResNet18, self).__init__()
    self.conv1 = Conv1D(filters=64, kernel_size=7, strides=2, padding='valid', use_bias=False)
    self.bn1 = BatchNormalization()
    self.maxpool = MaxPool1D(pool_size=3, strides=2, padding='valid')

    self.res1 = ResnetLayer(64, (64, 64), 3, 1)
    self.res2 = ResnetLayer(64, (128, 128), 3, 2)
    self.res3 = ResnetLayer(128, (256, 256), 3, 2)
    self.res4 = ResnetLayer(256, (512, 512), 3, 2)

    self.flatten = Flatten()
    self.dense1 = Dense(1024, activation="relu")
    self.dropout1 = Dropout(0.5)
    self.dense2 = Dense(1024, activation='relu')
    self.dropout2 = Dropout(0.5)
    self.dense3 = Dense(2)

  def call(self, x, training=False, mask=None):
    x = tf.keras.layers.ZeroPadding1D(padding=3)(x)
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
    x = self.maxpool(x)

    x = self.res1(x, training=training)
    x = self.res2(x, training=training)
    x = self.res3(x, training=training)
    x = self.res4(x, training=training)

    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dropout1(x, training=training)
    x = self.dense2(x)
    x = self.dropout2(x, training=training)
    return self.dense3(x)
