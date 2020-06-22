import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return criterion(preds[:, -1, :, :], labels[:, -1, :, :])

    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #     logits=preds[:, -1, :, :], labels=labels[:, -1, :, :]))


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

        self.softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        input_images = tf.reshape(input_images, [-1, self.samples_per_class * self.num_classes, 784])
        input_labels = tf.reshape(input_labels, [-1, self.samples_per_class * self.num_classes, self.num_classes])

        data_tensor = tf.concat([input_images, input_labels], 2)

        h = self.layer1(data_tensor)

        out = self.layer2(h)

        out = self.softmax(out)

        out = tf.reshape(out, [-1, self.samples_per_class, self.num_classes, self.num_classes])

        return out


def fill_last_zeros(inp_labels, num_classes):
    # inp_labels[:, -num_classes:] = 0.
    inp_labels[:, -1, :, :] = 0.
    return inp_labels


ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))

labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

labels_zeros = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels_zeros)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        l_zeros = fill_last_zeros(l.copy(), FLAGS.num_classes)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32), labels_zeros: l_zeros.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            l_zeros = fill_last_zeros(l.copy(), FLAGS.num_classes)
            feed = {ims: i.astype(np.float32), labels: l.astype(np.float32), labels_zeros: l_zeros.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
