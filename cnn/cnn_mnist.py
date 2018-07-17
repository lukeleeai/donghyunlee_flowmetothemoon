import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data/', one_hot=True, reshape=False)

num_classes = 10
num_epoch = 10
batch_size = 128
num_batch = mnist.train.num_examples // batch_size
dropout = 0.7
validation_size = 256

xavier = tf.contrib.layers.xavier_initializer()
zero = tf.zeros_initializer()

global_step = tf.Variable(0, trainable=False)
start_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=False)

X = tf.placeholder(tf.float32, [None, 784])
X = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, num_classes])

keep_prob = tf.placeholder(tf.float32)

K, L, M, N = 32, 64, 128, 1024
W1 = tf.get_variable('W1', [3, 3, 1, K], initializer=xavier)
W2 = tf.get_variable('W2', [3, 3, K, L], initializer=xavier)
W3 = tf.get_variable('W3', [3, 3, L, M], initializer=xavier)
Wc = tf.get_variable('Wc', [4 * 4 * M, N], initializer=xavier)
Wout = tf.get_variable('Wout', [N, num_classes], initializer=xavier)

b1 = tf.get_variable('b1', [K], initializer=zero)
b2 = tf.get_variable('b2', [L], initializer=zero)
b3 = tf.get_variable('b3', [M], initializer=zero)
bc = tf.get_variable('bc', [N], initializer=zero)
bout = tf.get_variable('bout', [num_classes], initializer=zero)


def conv2d(x, W, b, strides=1):
	conv = tf.nn.conv2d(x, W, [1, strides, strides, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, b)
	return tf.nn.relu(conv)


def max_pool(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, keep_prob):
	conv1 = conv2d(x, W1, b1)
	conv1 = max_pool(conv1)

	conv2 = conv2d(conv1, W2, b2)
	conv2 = max_pool(conv2)

	conv3 = conv2d(conv2, W3, b3)
	conv3 = max_pool(conv3)

	fc = tf.reshape(conv3, [-1, Wc.shape[0]])
	fc = tf.nn.relu(tf.matmul(fc, Wc) + bc)
	fc = tf.nn.dropout(fc, keep_prob)

	out = tf.matmul(fc, Wout) + bout
	return out


def cross_entropy(logits):
	xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
	return tf.reduce_mean(xent)


def train(cost):
	tf.summary.scalar('cost', cost)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(cost, global_step=global_step)
	return train_op


def accuracy(logits):
	correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) * 100
	tf.summary.scalar('accuracy', accuracy)
	return accuracy


with tf.Session() as sess:

	logits = conv_net(X, keep_prob)

	cost = cross_entropy(logits)
	train_op = train(cost)
	accuracy = accuracy(logits)

	summary_op = tf.summary.merge_all()
	saver = tf.train.Saver()

	summary_writer = tf.summary.FileWriter('cnn_mnist_log/', graph_def=sess.graph_def)

	sess.run(tf.global_variables_initializer())

	for epoch in range(num_epoch):
		avg_cost = 0
		for batch in range(num_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(train_op, feed_dict={X:batch_x, Y:batch_y, keep_prob:dropout})

			batch_cost = sess.run(cost, feed_dict={X:batch_x, Y:batch_y, keep_prob:1.0})
			avg_cost += batch_cost / num_batch
			print('Epoch: {:>2}, Batch: {:>3}, Cost: {:.9f}'.format(epoch+1, batch+1, batch_cost))

			if batch % 100 == 0:
				val_x = mnist.validation.images[:validation_size]
				val_y = mnist.validation.labels[:validation_size]
				val_feed_dict = {X:val_x, Y:val_y, keep_prob:1.0}

				accuracy_ = sess.run(accuracy, feed_dict=val_feed_dict)
				print('Epoch: {:>2}, Batch: {:>3}, Val Accuracy: {:.9f}'.format(epoch+1, batch+1, accuracy_))

				summary = sess.run(summary_op, feed_dict=val_feed_dict)
				summary_writer.add_summary(summary, sess.run(global_step))

				saver.save(sess, 'cnn_mnist_log/checkpoint', global_step=global_step)

	test_x = mnist.test.images[:validation_size]
	test_y = mnist.test.labels[:validation_size]
	test_acc = sess.run(accuracy, feed_dict={X: test_x, Y: test_y, keep_prob:1.0})
	print('Test Accuracy: %s' %  (test_acc))