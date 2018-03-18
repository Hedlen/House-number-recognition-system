import tensorflow as tf
# model for SVHN


class Model(object):

    @classmethod
    def batch_norm(cls, data):
        mean, var = tf.nn.moments(data, axes=[0])
        return tf.nn.batch_normalization(data, mean, var, None, None, 0.001)

    @classmethod
    def conv2d(cls, data, dropout, kernel_shape, bias_shape, pooling_stride):
        weight = tf.get_variable('weight', kernel_shape)
        bias = tf.get_variable('bias', bias_shape)
        conv_out = tf.nn.conv2d(data, weight, [1, 1, 1, 1], 'SAME')
        add_bias = tf.nn.bias_add(conv_out, bias)
        norm = cls.batch_norm(add_bias)
        activation = tf.nn.relu(norm)
        pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [
                              1, pooling_stride, pooling_stride, 1], 'SAME')
        return tf.nn.dropout(pool, dropout)

    @classmethod
    def full_connect(cls, data, kernel_shape, bias_shape, use_bias=True):
        weight = tf.get_variable('weight', kernel_shape)
        if use_bias:
            bias = tf.get_variable('bias', bias_shape)
        else:
            bias = tf.zeros(bias_shape, name='bias')
        full = tf.nn.bias_add(tf.matmul(data, weight), bias)
        activation = tf.nn.relu(full)
        return activation

    @classmethod
    def forward(cls, data, dropout=0.0):
        # the first hidden layer should contain maxout units instead of rectifier units
        with tf.variable_scope('conv1'):
            conv1 = cls.conv2d(data, dropout, [5, 5, 3, 48], [48], 2)

        with tf.variable_scope('conv2'):
            conv2 = cls.conv2d(conv1, dropout, [5, 5, 48, 64], [64], 1)
        with tf.variable_scope('conv3'):
            conv3 = cls.conv2d(conv2, dropout, [5, 5, 64, 128], [128], 2)
        with tf.variable_scope('conv4'):
            conv4 = cls.conv2d(conv3, dropout, [5, 5, 128, 160], [160], 1)
        with tf.variable_scope('conv5'):
            conv5 = cls.conv2d(conv4, dropout, [5, 5, 160, 192], [192], 2)
        with tf.variable_scope('conv6'):
            conv6 = cls.conv2d(conv5, dropout, [5, 5, 192, 192], [192], 1)
        with tf.variable_scope('conv7'):
            conv7 = cls.conv2d(conv6, dropout, [5, 5, 192, 192], [192], 2)
        with tf.variable_scope('conv8'):
            conv8 = cls.conv2d(conv7, dropout, [5, 5, 192, 192], [192], 1)

        shape = conv8.get_shape().as_list()
        #fc_size = shape[1] * shape[2] * shape[3]
        reshape = tf.reshape(conv8, [-1, shape[1] * shape[2] * shape[3]])

        with tf.variable_scope('fc1'):
            fc1 = cls.full_connect(reshape, [3072, 3072], [3072])
        with tf.variable_scope('fc2'):
            fc2 = cls.full_connect(fc1, [3072, 3072], [3072])

        with tf.variable_scope('digits_length'):
            length = cls.full_connect(fc2, [3072, 7], [7], False)
        with tf.variable_scope('digit1'):
            digit1 = cls.full_connect(fc2, [3072, 11], [11], False)
        with tf.variable_scope('digit2'):
            digit2 = cls.full_connect(fc2, [3072, 11], [11], False)
        with tf.variable_scope('digit3'):
            digit3 = cls.full_connect(fc2, [3072, 11], [11], False)
        with tf.variable_scope('digit4'):
            digit4 = cls.full_connect(fc2, [3072, 11], [11], False)
        with tf.variable_scope('digit5'):
            digit5 = cls.full_connect(fc2, [3072, 11], [11], False)
        digits = tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)

        return length, digits

    @classmethod
    def loss(clc, length_logits, digits_logits, length_labels, digits_labels):
        length = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=length_labels, logits=length_logits))
        digit1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length + digit1 + digit2 + digit3 + digit4 + digit5
        return loss
