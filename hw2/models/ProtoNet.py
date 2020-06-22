import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import utils

class ProtoNet(tf.keras.Model):

	def __init__(self, num_filters, latent_dim):
		super(ProtoNet, self).__init__()
		self.num_filters = num_filters
		self.latent_dim = latent_dim
		num_filter_list = self.num_filters + [latent_dim]
		self.convs = []
		for i, num_filter in enumerate(num_filter_list):
			block_parts = [
				layers.Conv2D(
					filters=num_filter,
					kernel_size=3,
					padding='SAME',
					activation='linear'),
			]

			block_parts += [layers.BatchNormalization()]
			block_parts += [layers.Activation('relu')]
			block_parts += [layers.MaxPool2D()]
			block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
			self.__setattr__("conv%d" % i, block)
			self.convs.append(block)
		self.flatten = tf.keras.layers.Flatten()

	def call(self, inp):
		out = inp
		for conv in self.convs:
			out = conv(out)
		out = self.flatten(out)
		return out


def pairwise_dist(A, B):
	"""
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
	with tf.variable_scope('pairwise_dist'):
		# squared norms of each row in A and B
		na = tf.reduce_sum(tf.square(A), 1)
		nb = tf.reduce_sum(tf.square(B), 1)

		# na as a row and nb as a co"lumn vectors
		na = tf.reshape(na, [-1, 1])
		nb = tf.reshape(nb, [1, -1])

		# return pairwise euclidead difference matrix
		# D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
		D = na - 2 * tf.matmul(A, B, False, True) + nb
	return D


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
	"""
		calculates the prototype network loss using the latent representation of x
		and the latent representation of the query set
		Args:
			x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
			q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
			labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
			num_classes: number of classes (N) for classification
			num_support: number of examples (S) in the support set
			num_queries: number of examples (Q) in the query set
		Returns:
			ce_loss: the cross entropy loss between the predicted labels and true labels
			acc: the accuracy of classification on the queries
	"""
	#############################
    #### YOUR CODE GOES HERE ####

    # compute the prototypes

	x_reshape = tf.reshape(x_latent, [num_classes, num_support, -1])

	proto = tf.math.reduce_mean(x_reshape, axis=1)

	dist_matrix = pairwise_dist(q_latent, proto)
	dist_matrix = tf.math.scalar_mul(-1, dist_matrix)

	# compute cross entropy loss

	labels_reshape = tf.reshape(labels_onehot, [num_classes * num_queries, -1])

	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	ce_loss = loss_fn(labels_reshape, dist_matrix)

	# ce_loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_reshape, logits=dist_matrix))

	indices = tf.argmax(dist_matrix, 1)
	depth = num_classes
	pred = tf.one_hot(indices, depth)

	acc = utils.accuracy(pred, labels_reshape)

    # note - additional steps are needed!

    # return the cross-entropy loss and accuracy
    # ce_loss, acc = None, None
    #############################
	return ce_loss, acc

