import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_row, self.img_col, self.channels = img_shape
        print("Initializing Generator Weights")

        self.fc = tf.layers.Dense(units=4 * 4 * 1024)
        self.conv1 = tf.layers.Conv2DTranspose(filters=512, kernel_size=(5, 5), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = tf.layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = tf.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = tf.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2),
                                                   padding="SAME",
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    def forward(self, X, momentum=0.5):
        z = self.fc(X)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.relu(z)

        z = tf.reshape(z, [-1, 4, 4, 1024])
        z = self.conv1(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.relu(z)

        z = self.conv2(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.relu(z)

        z = self.conv3(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)
        z = tf.nn.relu(z)

        z = self.conv4(z)
        z = tf.layers.batch_normalization(z, momentum=momentum)

        return tf.nn.tanh(z)


