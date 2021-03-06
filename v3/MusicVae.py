import tensorflow as tf
import numpy as np
import os
import json
from DanceDataset import DanceDataset


class MusicVae:
    def __init__(self ,
                 model_save_dir='',
                 log_dir='',
                 train_file_list=[]):
        self.time_step=120
        self.acoustic_dim=16
        self.dim_z=8
        self.ADD_NOISE = False
        self.n_hidden = 500
        self.epoch_size = 1000
        self.batch_size = 10
        self.learning_rate = 1e-5
        self.train_file_list=train_file_list
        self.keep_prob=0.9


        self.model_save_dir=model_save_dir
        self.log_dir=log_dir

    def init_dataset(self):
        self.train_dataset = DanceDataset(train_file_list=self.train_file_list,
                                          acoustic_dim=self.acoustic_dim,
                                          temporal_dim=3,
                                          motion_dim=63,
                                          time_step=self.time_step,
                                          overlap=True,
                                          overlap_interval=10,
                                          batch_size=self.batch_size)

    def music_encoder(self,input_acoustic, n_hidden,n_output):
        # input ?*16  n_hidden 500 keep_prob 1 n_output 20
        with tf.variable_scope("music_encoder"):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [input_acoustic.get_shape()[1], n_hidden], initializer=w_init)#15*16*500
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(input_acoustic, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            gaussian_params = tf.matmul(h1, wo) + bo

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

        return mean, stddev

    # Bernoulli MLP as decoder
    def music_decoder(self,z, n_hidden, n_output, reuse=False):

        with tf.variable_scope("music_decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer-mean
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h1, wo) + bo)

        return y

    # Gateway
    def music_vae(self,input_acoustic, target, n_hidden,dim_z, reuse=False):
        with tf.variable_scope("music_autoencoder", reuse=reuse):
        # encoding
            mu, sigma = self.music_encoder(input_acoustic, n_hidden,dim_z)

            # sampling by re-parameterization technique
            z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

            # decoding
            y = self.music_decoder(z, n_hidden, self.acoustic_dim)
            y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
            # smoothness loss
            loss_smo = tf.losses.mean_squared_error(y[1:], y[:-1])

            marginal_likelihood = tf.losses.mean_squared_error(target, y)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

            marginal_likelihood = tf.reduce_mean(marginal_likelihood)
            KL_divergence = tf.reduce_mean(KL_divergence)

            ELBO = marginal_likelihood - KL_divergence

            loss = marginal_likelihood+loss_smo

            return y, z, loss, marginal_likelihood, KL_divergence

    def train(self,resume=False):
        acoustic_input = tf.placeholder(tf.float32, shape=[None, self.acoustic_dim])
        acoustic_target = tf.placeholder(tf.float32, shape=[None,  self.acoustic_dim])


        y, z, loss, neg_marginal_likelihood, KL_divergence = self.music_vae(acoustic_input, acoustic_target, self.n_hidden,self.dim_z)

        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


        iterator = self.train_dataset.train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        step_num=self.train_dataset.train_size // self.batch_size
        print("step_size: %d" % (step_num))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            if resume:
                ckpt = tf.train.get_checkpoint_state(self.model_save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("restore weight from %s ..."%self.model_save_dir)
                    saver.restore(sess, ckpt.model_checkpoint_path)
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            writer.add_graph(sess.graph)
            summ = tf.summary.merge_all()
            for i in range(self.epoch_size):
                print("epoch:%d" % i)
                # 每次进行训练的12时候，每个batch训练batch_size个样本
                loss_avg = 0
                sess.run(iterator.initializer)
                for step in range(step_num):
                    batch_data = sess.run(next_element)
                    acoustic_in = batch_data[:, :, :self.acoustic_dim]
                    acoustic_in=acoustic_in.reshape([-1,self.acoustic_dim])
                    acoustic_ta=acoustic_in

                    if self.ADD_NOISE:
                        acoustic_in = acoustic_in * np.random.randint(2, size=acoustic_in.shape)
                        acoustic_in += np.random.randint(2, size=acoustic_in.shape)

                    test_y,test_z,_, tot_loss, loss_likelihood, loss_divergence = sess.run(
                            (y, z,train_op, loss, neg_marginal_likelihood, KL_divergence),
                            feed_dict={acoustic_input: acoustic_in,acoustic_target: acoustic_ta})
                    if step % 10 == 0:
                        print("epoch: %d step: %d, L_tot %03.6f L_likelihood %03.6f L_divergence %03.6f " % (
                            i, step,tot_loss, loss_likelihood, loss_divergence))
                print("epoch %d: L_tot %03.6f L_likelihood %03.6f L_divergence %03.6f" % (i, tot_loss, loss_likelihood, loss_divergence))
                if (i + 1) % 10 == 0:
                    print("保存模型：", saver.save(sess,os.path.join(self.model_save_dir,'stock2.model'), global_step=i))

            writer.close()
    def predict(self,test_file,result_save_dir):
        acoustic_input = tf.placeholder(tf.float32, shape=[None, self.acoustic_dim])
        acoustic_target = tf.placeholder(tf.float32, shape=[None,  self.acoustic_dim])

        test_dataset,train_motion_scaler,test_size = self.train_dataset.load_test_data(test_file)





        y, z, loss, neg_marginal_likelihood, KL_divergence = self.music_vae(acoustic_input, acoustic_target, self.n_hidden,self.dim_z)


       
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(self.model_save_dir)
            saver.restore(sess, module_file)
            file_name = os.path.basename(test_file) + '.json'
            print("test the file %s" % file_name)
            iterator = test_dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            test_predict = np.empty([0, self.acoustic_dim])
            music_test = np.empty([0, self.acoustic_dim])

            sess.run(iterator.initializer)
            loss_sum = 0
            if (test_size % self.batch_size == 0):
                test_size = test_size // self.batch_size
            else:
                test_size = test_size // self.batch_size + 1
            for step in range(test_size):
                batch_data = sess.run(next_element)
                acoustic_in = batch_data[:, :, :self.acoustic_dim]
                acoustic_in=acoustic_in.reshape([-1,self.acoustic_dim])
                acoustic_ta=acoustic_in




                test_y,test_z, tot_loss, loss_likelihood, loss_divergence = sess.run(
                            (y, z, loss, neg_marginal_likelihood, KL_divergence),
                            feed_dict={acoustic_input: acoustic_in,acoustic_target: acoustic_ta})

                predict = test_y.reshape((-1,  self.acoustic_dim))
                acoustic_ta = acoustic_ta.reshape((-1,  self.acoustic_dim))

                test_predict = np.append(test_predict, predict, axis=0)
                music_test = np.append(music_test, acoustic_ta, axis=0)

                loss_sum += tot_loss

            test_predict = train_motion_scaler.inverse_transform(test_predict)
            music_test = train_motion_scaler.inverse_transform(music_test)
            acc = np.average(np.abs(test_predict - music_test))

            test_predict = np.reshape(test_predict, [-1, self.acoustic_dim])
            length = test_predict.shape[0]
            test_predict = test_predict.tolist()
            data = {"length": length, "music": test_predict}
            with open(os.path.join(result_save_dir,file_name), 'w') as file_object:
                json.dump(data, file_object)
            print(loss_sum, acc)