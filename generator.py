import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_row, self.img_col, self.channels = img_shape
        print("Initializing Generator Weights")

        self.conv1 = tf.layers.Conv2DTranspose( filters=1024, kernel_size=(4,4), strides=(1,1),
                                                               padding="valid", kernel_initializer=tf.random_normal_initializer(stddev=0.02) )
        self.conv2 = tf.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = tf.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = tf.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = tf.layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    def forward(self, X, momentum=0.5):
        z = self.conv1(X)
        z = tf.layers.batch_normalization(z, momentum=momentum)
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

        z = self.conv5(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)

        return tf.nn.tanh(z)




