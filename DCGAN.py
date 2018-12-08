import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from generator import Generator
from discriminator import Discriminator

fixed_z = np.random.uniform(-1, 1, (25, 1, 1, 100))


class DCGAN:
    def __init__(self, img_shape, epochs=50000,
                 lr_gen=0.0002, lr_dc=0.0002, z_shape=100, batch_size=100,
                 beta1=0.5, epochs_for_sample=50):

        self.rows, self.cols, self.channels = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_shape = z_shape
        self.epochs_for_sample = epochs_for_sample
        self.generator = Generator(img_shape)
        self.discriminator = Discriminator(img_shape)

        mnist = tf.keras.datasets.mnist

        (x_train, _), (x_test, _) = mnist.load_data()

        X = np.concatenate([x_train, x_test])
        X = np.reshape(X, (-1, 28, 28, 1))
        X = tf.image.resize_images(X, [64, 64])
        self.X = (X / 127.5) - 1  # Scale between -1 and 1

        self.phX = tf.placeholder(dtype=tf.float32, shape=[None, self.rows, self.cols, self.channels])
        self.phZ = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, self.z_shape])
        self.loss_plot = tf.placeholder(dtype=tf.float32, shape=[])

        self.gen_out = self.generator.forward(self.phZ)

        disc_logits_fake = self.discriminator.forward(self.gen_out)
        disc_logits_real = self.discriminator.forward(self.phX)

        disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake, labels=tf.zeros_like(disc_logits_fake)))
        disc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_real, labels=tf.ones_like(disc_logits_real)))

        self.disc_loss = tf.add(disc_loss_fake, disc_loss_real)

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,
                                                                               labels=tf.ones_like(disc_logits_fake)))

        self.disc_train = tf.train.AdamOptimizer(lr_dc, beta1=beta1).minimize(self.disc_loss,
                                                                              var_list=self.discriminator.variables)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1=beta1).minimize(self.gen_loss,
                                                                              var_list=self.generator.variables)

    def train(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        train_writer = tf.summary.FileWriter('./logs')
        train_writer.add_graph(tf.get_default_graph())

        dc_plot = tf.summary.scalar('Discriminator', self.loss_plot)
        gen_plot = tf.summary.scalar('Generator', self.loss_plot)

        cnt = 0

        for i in range(self.epochs):
            X_numpy = self.sess.run(self.X)
            idx = np.random.randint(0, len(X_numpy), self.batch_size)
            batch_X = X_numpy[idx]

            batch_Z = np.random.uniform(-1, 1, (self.batch_size, 1, 1, self.z_shape))
            _, d_loss = self.sess.run([self.disc_train, self.disc_loss],
                                      feed_dict={self.phX: batch_X, self.phZ: batch_Z})

            batch_Z = np.random.uniform(-1, 1, (self.batch_size, 1, 1, self.z_shape))
            _, g_loss = self.sess.run([self.gen_train, self.gen_loss], feed_dict={self.phZ: batch_Z})

            if i % self.epochs_for_sample == 0:
                self.generate_sample(i)
                print("Epoch: " + str(i) + " Discriminator loss: " + str(d_loss) + " Generator loss: " + str(g_loss))
                train_writer.add_summary(self.sess.run(dc_plot, feed_dict={self.loss_plot: d_loss}),
                                         i / self.epochs_for_sample)
                train_writer.add_summary(self.sess.run(gen_plot, feed_dict={self.loss_plot: g_loss}),
                                         i / self.epochs_for_sample)

    def generate_sample(self, epoch):
        c = 5
        r = 5
        imgs = self.sess.run(self.gen_out, feed_dict={self.phZ: fixed_z})
        imgs = imgs * 0.5 + 0.5  # scale between 0, 1
        fig, axs = plt.subplots(c, r)
        cnt = 0
        for i in range(c):
            for j in range(r):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
                fig.savefig("samples/%05d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    img_shape = (64, 64, 1)
    epochs = 50000
    dcgan = DCGAN(img_shape, epochs)

    if not os.path.exists('samples/'):
        os.makedirs('samples/')

    dcgan.train()



