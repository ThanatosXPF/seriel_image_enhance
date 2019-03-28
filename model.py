import tensorflow as tf

from tf_utils import conv2d, deconv2d, weighted_l2
from conv_gru import ConvGru
from my_gru import ConvGRUCell
import config as c


class Model:
    def __init__(self, restore_path=None):
        self.input = None
        self.batch = c.BATCH_SIZE

        if c.COLD_START:
            self.time_seq = c.TIME_SEQ
        else:
            self.time_seq = c.TIME_SEQ + c.IN_SEQ

        if c.DATA_PRECISION == "HALF":
            self.dtype = tf.float16
        elif c.DATA_PRECISION == "SINGLE":
            self.dtype = tf.float32

        self._lr = c.LR
        self.in_data = None
        self.gt_data = None

        self.define_convgru_graph()

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        self.saver = tf.train.Saver(max_to_keep=0)

        if restore_path is not None:
            self.saver.restore(self.sess, restore_path)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def define_convgru_graph(self):
        with tf.name_scope("Graph"):
            with tf.variable_scope("encoder") and tf.device('/device:GPU:0'):
                self.in_data = tf.placeholder(self.dtype,
                                              name="input",
                                              shape=[None, self.time_seq, c.H, c.W, 1])
                self.gt_data = tf.placeholder(self.dtype,
                                              name="gt",
                                              shape=[None, self.time_seq, c.H, c.W, 1])
                conv_in = tf.reshape(self.in_data, shape=[-1, c.H, c.W, 1])

                first_conv = conv2d(conv_in, name="conv1",
                                    kshape=(7, 7, 1, c.NUM_FILTER), dtype=self.dtype)

            with tf.variable_scope("Conv_Gru", reuse=tf.AUTO_REUSE):
                # gru = ConvGru(dtype=self.dtype)
                gru = ConvGRUCell(num_filter=c.NUM_FILTER,
                                  b_h_w=(self.batch, c.H, c.W),
                                  h2h_kernel=3, i2h_kernel=3,
                                  name="gru_cell", chanel=c.NUM_FILTER)
                gru_input = tf.reshape(first_conv, shape=[self.batch, self.time_seq, c.H, c.W, c.NUM_FILTER])
                # states = gru.init_state(shape=[self.batch, c.H, c.W, c.NUM_FILTER])
                # outputs = gru(gru_input, states)
                outputs, _ = gru.unroll(length=self.time_seq, inputs=gru_input, begin_state=None)

            with tf.variable_scope("decoder") and tf.device('/device:GPU:0'):
                dec_in = tf.reshape(outputs, shape=[-1, c.H, c.W, c.NUM_FILTER])
                dec = deconv2d(dec_in, name="dec2_", kshape=(7, 7), n_outputs=1)
                # dec = tf.cast(dec, dtype=tf.float16)
                out = tf.reshape(dec, shape=[-1, self.time_seq, c.H, c.W, 1])
                print(out)
            with tf.variable_scope("loss"):
                self.result = out
                if c.USE_BALANCED_LOSS:
                    self.loss = weighted_l2(out, self.gt_data)
                else:
                    self.loss = tf.reduce_mean(tf.square(tf.subtract(out, self.gt_data)))
                self.mse = tf.reduce_mean(tf.square(tf.subtract(out, self.gt_data)))
                self.rmse = tf.sqrt(self.mse)
                self.mae = tf.reduce_mean(tf.abs(tf.subtract(out, self.gt_data)))

                self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    def define_convgru_graph_2(self):
        with tf.name_scope("Graph"):
            with tf.variable_scope("encoder") and tf.device('/device:GPU:0'):
                self.in_data = tf.placeholder(self.dtype,
                                              name="input",
                                              shape=[None, self.time_seq, c.H, c.W, 1])
                self.gt_data = tf.placeholder(self.dtype,
                                              name="gt",
                                              shape=[None, self.time_seq, c.H, c.W, 1])
                conv_in = tf.reshape(self.in_data, shape=[-1, c.H, c.W, 1])

                first_conv = conv2d(conv_in, name="conv1",
                                    kshape=(7, 7, 1, c.NUM_FILTER), dtype=self.dtype)

            with tf.variable_scope("Conv_Gru", reuse=tf.AUTO_REUSE):
                gru = ConvGru(dtype=self.dtype)
                gru_input = tf.reshape(first_conv, shape=[self.batch, self.time_seq, c.H, c.W, c.NUM_FILTER])
                states = gru.init_state(shape=[self.batch, c.H, c.W, c.NUM_FILTER])
                outputs = gru(gru_input, states)

            with tf.variable_scope("decoder") and tf.device('/device:GPU:0'):
                dec_in = tf.reshape(outputs, shape=[-1, c.H, c.W, c.NUM_FILTER])
                # dec = deconv2d(dec_in, name="dec2_", kshape=(7, 7), n_outputs=1)
                dec = conv2d(dec_in, name="conv2",
                                    kshape=(7, 7, c.NUM_FILTER, 1), dtype=self.dtype)
                # dec = tf.cast(dec, dtype=tf.float16)
                out = tf.reshape(dec, shape=[-1, self.time_seq, c.H, c.W, 1])
                print(out)
            with tf.variable_scope("loss"):
                self.result = out
                if c.USE_BALANCED_LOSS:
                    self.loss = weighted_l2(out, self.gt_data)
                else:
                    self.loss = tf.reduce_mean(tf.square(tf.subtract(out, self.gt_data)))
                self.mse = tf.reduce_mean(tf.square(tf.subtract(out, self.gt_data)))
                self.rmse = tf.sqrt(self.mse)
                self.mae = tf.reduce_mean(tf.abs(tf.subtract(out, self.gt_data)))

                self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    def train_step(self, pred, gt):
        _, l2, mse, rmse, mae, res = self.sess.run([self.optimizer,self.loss, self.mse,
                                           self.rmse, self.mae, self.result],
                                          feed_dict={self.in_data: pred,
                                                     self.gt_data: gt}
                                          )
        if len(res[res !=0]) ==0:
            print("All Result 0!!!!!!!!!!!!!")
        else:
            print("in: ", pred.min(), pred.max())
            print("gt: ", gt.min(), gt.max())
            print("result: ", res.min(), res.max())
        return l2, mse, rmse, mae

    def save_model(self, iter):
        save_path = self.saver.save(self.sess, c.SAVE_DIR+"model.ckpt", iter)
        print("Model saved in path: %s" % save_path)

    def valid(self, pred, gt):
        l2, mse, rmse, mae, result = self.sess.run([self.loss, self.mse, self.rmse,
                                                self.mae, self.result],
                                          feed_dict={self.in_data: pred,
                                                     self.gt_data: gt}
                                          )
        print("in: ", pred.min(), pred.max())
        print("gt: ", gt.min(), gt.max())
        print("result: ", result.min(), result.max())
        return l2, mse, rmse, mae, result


