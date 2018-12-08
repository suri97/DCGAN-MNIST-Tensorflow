import numpy as np
import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, img_shape):
        self.img_rows, self.img_cols, self.channels = img_shape
        super(Discriminator, self).__init__()
        print ("Initializing Discriminator")

        self.conv1 = tf.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv2 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv3 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.fc = tf.layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.2))

    def forward(self, X, momentum=0.5):
        X = tf.reshape(X, [-1, self.img_rows, self.img_cols, self.channels])

        z = self.conv1(X)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = self.conv2(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = self.conv3(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = tf.layers.flatten(z)
        logits = self.fc(z)
        return logits


