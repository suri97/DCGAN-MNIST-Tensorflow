import numpy as np
import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, img_shape):
        self.img_rows, self.img_cols, self.channels = img_shape
        super(Discriminator, self).__init__()
        print ("Initializing Discriminator")

        self.conv1 = tf.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv2 = tf.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv3 = tf.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv4 = tf.layers.Conv2D(filters=1024, kernel_size=(4, 4), strides=[2, 2], padding="SAME",
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        self.conv5 = tf.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=[1, 1], padding="valid",
                                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.2))

    def forward(self, X, momentum=0.5):
        z = self.conv1(X)
        z = tf.nn.leaky_relu(z)

        z = self.conv2(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = self.conv3(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = self.conv4(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        logits = self.conv5(z)
        return logits


