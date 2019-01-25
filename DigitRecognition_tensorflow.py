{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TensorFlow version: 1.12.0\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "print (\"TensorFlow version: \" + tf.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting MNIST_data/train-images-idx3-ubyte.gz\n",
      "Extracting MNIST_data/train-labels-idx1-ubyte.gz\n",
      "Extracting MNIST_data/t10k-images-idx3-ubyte.gz\n",
      "Extracting MNIST_data/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"MNIST_data/\", one_hot=True) # y labels are oh-encoded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_train = mnist.train.num_examples # 55,000\n",
    "n_validation = mnist.validation.num_examples # 5000\n",
    "n_test = mnist.test.num_examples # 10,000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_input = 784   # input layer (28x28 pixels)\n",
    "n_hidden1 = 512 # 1st hidden layer\n",
    "n_hidden2 = 256 # 2nd hidden layer\n",
    "n_hidden3 = 128 # 3rd hidden layer\n",
    "n_output = 10   # output layer (0-9 digits)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rate = 1e-4\n",
    "n_iterations = 1000\n",
    "batch_size = 128\n",
    "dropout = 0.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(\"float\", [None, n_input])\n",
    "Y = tf.placeholder(\"float\", [None, n_output])\n",
    "keep_prob = tf.placeholder(tf.float32) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "weights = {\n",
    "    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),\n",
    "    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),\n",
    "    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),\n",
    "    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "biases = {\n",
    "    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),\n",
    "    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),\n",
    "    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),\n",
    "    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])\n",
    "layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])\n",
    "layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])\n",
    "layer_drop = tf.nn.dropout(layer_3, keep_prob)\n",
    "output_layer = tf.matmul(layer_3, weights['out']) + biases['out']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-12-812430f7e244>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "\n",
      "Future major versions of TensorFlow will allow gradients to flow\n",
      "into the labels input on backprop by default.\n",
      "\n",
      "See `tf.nn.softmax_cross_entropy_with_logits_v2`.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))\n",
    "train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 0 \t| Loss = 4.2574983 \t| Accuracy = 0.140625\n",
      "Iteration 100 \t| Loss = 0.5458355 \t| Accuracy = 0.859375\n",
      "Iteration 200 \t| Loss = 0.43445873 \t| Accuracy = 0.859375\n",
      "Iteration 300 \t| Loss = 0.2428877 \t| Accuracy = 0.9296875\n",
      "Iteration 400 \t| Loss = 0.29975304 \t| Accuracy = 0.9375\n",
      "Iteration 500 \t| Loss = 0.26157486 \t| Accuracy = 0.9140625\n",
      "Iteration 600 \t| Loss = 0.35185328 \t| Accuracy = 0.8828125\n",
      "Iteration 700 \t| Loss = 0.34758186 \t| Accuracy = 0.90625\n",
      "Iteration 800 \t| Loss = 0.39502913 \t| Accuracy = 0.890625\n",
      "Iteration 900 \t| Loss = 0.46541688 \t| Accuracy = 0.875\n"
     ]
    }
   ],
   "source": [
    "# train on mini batches\n",
    "for i in range(n_iterations):\n",
    "    batch_x, batch_y = mnist.train.next_batch(batch_size)\n",
    "    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})\n",
    "\n",
    "    # print loss and accuracy (per minibatch)\n",
    "    if i%100==0:\n",
    "        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})\n",
    "        print(\"Iteration\", str(i), \"\\t| Loss =\", str(minibatch_loss), \"\\t| Accuracy =\", str(minibatch_accuracy))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Accuracy on test set: 0.9168\n"
     ]
    }
   ],
   "source": [
    "test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})\n",
    "print(\"\\nAccuracy on test set:\", test_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from PIL import Image\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "img = np.invert(Image.open(\"test_img.png\").convert('L')).ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction for test image: 2\n"
     ]
    }
   ],
   "source": [
    "prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})\n",
    "print (\"Prediction for test image:\", np.squeeze(prediction))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
