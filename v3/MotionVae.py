import tensorflow as tf
import numpy as np
import os
import json
from DanceDataset import DanceDataset


class MotionVae:
    def __init__(self ,
                 model_save_dir='',
                 log_dir='',
                 train_file_list=[]):
        self.time_step=120
        self.motion_dim=63
        self.dim_z=16
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
                                          acoustic_dim=16,
                                          temporal_dim=3,
                                          motion_dim=63,
                                          time_step=self.time_step,
                                          overlap=True,
                                          overlap_interval=10,
                                          batch_size=self.batch_size)

    def motion_encoder(self ,input_motion, n_hidden,n_output):
    # input ?*16  n_hidden 500 keep_prob 1 n_output 20
        with tf.variable_scope("motion_encoder"):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [input_motion.get_shape()[1], n_hidden], initializer=w_init)#15*16*500
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(input_motion, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
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
    def motion_decoder(self ,z, n_hidden, n_output,  reuse=False):

        with tf.variable_scope("motion_decoder", reuse=reuse):
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
    def motion_vae(self ,input_motion, target, n_hidden,dim_z):

        # encoding
        mu, sigma =self.motion_encoder(input_motion, n_hidden,dim_z)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self.motion_decoder(z, n_hidden, self.motion_dim)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        #smoothness loss
        loss_smo=tf.losses.mean_squared_error(y[1:],y[:-1])

        #tf.reduce_mean(x, 0)
        # loss
        marginal_likelihood = tf.losses.mean_squared_error(target ,y)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)


        loss = marginal_likelihood+loss_smo


        return y, z, loss, marginal_likelihood, KL_divergence,loss_smo

    def train(self,resume=False):
        motion_input = tf.placeholder(tf.float32, shape=[None, self.motion_dim])
        motion_target = tf.placeholder(tf.float32, shape=[None,  self.motion_dim])



        y, z, loss, neg_marginal_likelihood, KL_divergence,loss_smoothness = self.motion_vae(motion_input, motion_target, self.n_hidden,self.dim_z)

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
                    motion_in = batch_data[:, :, -self.motion_dim:]
                    motion_in=motion_in.reshape([-1,self.motion_dim])
                    motion_ta=motion_in

                    if self.ADD_NOISE:
                        motion_in = motion_in * np.random.randint(2, size=motion_in.shape)
                        motion_in += np.random.randint(2, size=motion_in.shape)

                    test_y, test_z, _, tot_loss, loss_likelihood, loss_divergence,loss_smo = sess.run(
                        [y, z, train_op, loss, neg_marginal_likelihood, KL_divergence,loss_smoothness],
                        feed_dict={motion_input: motion_in, motion_target: motion_ta})
                    if step % 10 == 0:
                        print("epoch: %d step: %d, L_tot %03.6f L_likelihood %03.6f L_divergence %03.6f  loss_smoothness %03.6f " % (
                            i, step,tot_loss, loss_likelihood, loss_divergence,loss_smo))
                print("epoch %d: L_tot %03.2f L_likelihood %03.6f L_divergence %03.6f" % (i, tot_loss, loss_likelihood, loss_divergence))
                if (i + 1) % 10 == 0:
                    print("保存模型：", saver.save(sess,os.path.join(self.model_save_dir,'stock2.model'), global_step=i))

            writer.close()
    def predict(self,test_file,result_save_dir):
        motion_input = tf.placeholder(tf.float32, shape=[None, self.motion_dim])
        motion_target = tf.placeholder(tf.float32, shape=[None,  self.motion_dim])

        test_dataset,train_motion_scaler,test_size,center = self.train_dataset.load_test_data(test_file,0)





        y, z, loss, neg_marginal_likelihood, KL_divergence,loss_smoothness = self.motion_vae(motion_input, motion_target, self.n_hidden, self.dim_z)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(self.model_save_dir)
            saver.restore(sess, module_file)
            file_name = os.path.basename(test_file) + '.json'
            print("test the file %s" % file_name)
            iterator = test_dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            test_predict = np.empty([0, self.motion_dim])
            motion_test = np.empty([0, self.motion_dim])

            sess.run(iterator.initializer)
            loss_sum = 0
            if (test_size % self.batch_size == 0):
                test_size = test_size // self.batch_size
            else:
                test_size = test_size // self.batch_size + 1
            for step in range(test_size):
                batch_data = sess.run(next_element)
                motion_in = batch_data[:, :, -self.motion_dim:]
                motion_in = motion_in.reshape([-1, self.motion_dim])
                motion_ta = motion_in

                test_y,test_z, tot_loss, loss_likelihood, loss_divergence = sess.run(
                            (y, z, loss, neg_marginal_likelihood, KL_divergence),
                            feed_dict={motion_input: motion_in,motion_target: motion_ta})

                predict = test_y.reshape((-1,  self.motion_dim))
                motion_ta = motion_ta.reshape((-1,  self.motion_dim))

                test_predict = np.append(test_predict, predict, axis=0)
                motion_test = np.append(motion_test, motion_ta, axis=0)

                loss_sum += tot_loss

            test_predict = train_motion_scaler.inverse_transform(test_predict)
            motion_test = train_motion_scaler.inverse_transform(motion_test)
            acc = np.average(np.abs(test_predict - motion_test))

            test_predict = np.reshape(test_predict, [-1, self.motion_dim//3, 3])
            length = test_predict.shape[0]
            test_predict = test_predict.tolist()
            center=center.tolist()
            data = {"length": length, "skeletons": test_predict,"center":center}
            with open(os.path.join(result_save_dir,file_name), 'w') as file_object:
                json.dump(data, file_object)
            print(loss_sum, acc)
