import tensorflow as tf
import numpy as np

from tf_utils import conv2d
import config as c


class ConvGru(object):
    def __init__(self, num_filter=c.NUM_FILTER, device="/device:GPU:1", dtype=tf.float32):
        self.i2h_shape = (3, 3, num_filter, num_filter * 3)
        self.h2h_shape = (5, 5, num_filter, num_filter * 3)
        self.device = device
        if c.COLD_START:
            self.time_seq = c.TIME_SEQ
        else:
            self.time_seq = c.TIME_SEQ + c.IN_SEQ
        self.batch = c.BATCH_SIZE
        self.dtype = dtype

    def init_state(self, shape):
        state = tf.zeros(shape, dtype=self.dtype, name="init_state")
        return state

    def __call__(self, input_data, states):
        """

        :param input_data: shape (batch, time, H, W, channel)
        :param states: shape (batch, H, W, channel)

        :return: outputs: shape(batch, time, H, W, channel)
        """
        assert input_data is not None
        assert states is not None
        with tf.device(self.device):
            outputs = []
            for i in range(self.time_seq):
                in_data = input_data[:,i-1,:,:,:]
                i2h = conv2d(in_data, name="i2h", kshape=self.i2h_shape, dtype=self.dtype)
                i2h = tf.split(i2h, 3, axis=3)

                h2h = conv2d(states, name="h2h", kshape=self.h2h_shape, dtype=self.dtype)
                h2h = tf.split(h2h, 3, axis=3)

                reset_gate = tf.nn.sigmoid(i2h[0] + h2h[0], name="reset")
                update_gate = tf.nn.sigmoid(i2h[1] + h2h[1], name="update")

                new_mem = tf.nn.leaky_relu(i2h[2] + reset_gate * h2h[2], alpha=0.2, name="leaky")

                next_h = update_gate * states + (1 - update_gate) * new_mem

                states = next_h
                outputs.append(next_h)
            outputs = tf.stack(outputs, axis=1)

        return outputs
