import glob
import os
import random

import numpy as np
import tensorflow as tf
import numpy


class FFModel():
    def __init__(self, input_size):

        tf.reset_default_graph()

        n_hidden_1 = 100  # 1st layer number of features
        n_hidden_2 = 150  # 2nd layer number of features
        n_input = 10
        n_classes = 2

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.3)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.3)),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.3))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.3)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.3)),
            'out': tf.Variable(tf.random_normal([n_classes], stddev=0.3))
        }
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        self.model = self.multilayer_perceptron(self.X, weights, biases)
        #self.pred = tf.nn.softmax(self.multilayer_perceptron(self.X, weights, biases))
        self.pred = self.model
        # self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.Y))
        #self.loss_function = tf.reduce_mean(tf.pow(self.Y - self.pred, 2))
        self.loss_function = tf.reduce_mean(tf.pow(self.Y - self.model, 2))
        #self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_function)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss_function)

        self.session = tf.Session()
        self.saver = tf.train.Saver()

        tf.initialize_all_variables().run(session=self.session)

        files_path = os.path.join("model", 'model*.ckpt*.index')
        labels_files = sorted(
            glob.iglob(files_path), key=os.path.getctime, reverse=True)

        if labels_files:
            print "loading from ", labels_files[0]
            self.saver.restore(self.session, labels_files[0].replace(".index", ""))
    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    # def model(self, X):
    #
    #     rng = numpy.random
    #
    #     # Set model weights
    #     W = tf.Variable(rng.randn(), name="weight")
    #     b = tf.Variable(rng.randn(), name="bias")
    #
    #     # Construct a linear model
    #     pred = tf.add(tf.mul(X, W), b)
    #     return pred
        # w_h = self.init_weights([input_size, 10])  # create symbolic variables
        # w_o = self.init_weights([10, 1])
        #
        # h1 = (tf.matmul(X, w_h))  # this is a basic mlp, think 2 stacked logistic regressions
        # #h1 = tf.nn.sigmoid(tf.matmul(X, w_h))  # this is a basic mlp, think 2 stacked logistic regressions
        # h2 = tf.matmul(h1, w_o)
        # return h2  # note that we dont take the softmax at the end because our cost fn does that for us

    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        #layer_1 = tf.Print(layer_1, [layer_1], "layer1: ")
        layer_1 = tf.nn.relu(layer_1)
        #layer_1 = tf.sigmoid(layer_1)
        #layer_1 = tf.Print(layer_1, [layer_1], "relu1: ")

        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        #layer_2 = tf.sigmoid(layer_2)
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        #out_layer = tf.Print(out_layer, [out_layer], "out_layer: ")
        out_layer = tf.nn.softplus(out_layer)
        #out_layer = tf.sigmoid(out_layer)
        return out_layer

    def training(self, loss, learning_rate):
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer()
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def loss(self, logits, labels):

        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        #   logits, labels, name='xentropy')
        # loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        # loss = tf.reduce_sum(tf.pow((logits - labels), 2))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        return loss

    def train(self, train_data, labels):
        for i in range(10):
            _, loss_value = self.session.run([self.optimizer, self.loss_function],
                                             feed_dict={self.X: train_data, self.Y: labels})
            if (i % 100) == 0:
                print "loss value: ", loss_value
        print "loss value: ", loss_value

            #self.saver.save(self.session, 'model.ckpt')

    def save(self):
        self.saver.save(self.session, 'model/model' + str(random.randint(0, 1000000)) + '.ckpt')

    def predict(self, input):

        run = self.session.run([self.pred],
                               feed_dict={self.X: input})

        if random.random() < 0.00002:
            print "output : ", run

        return run[0][0]
