import tensorflow as tf
import numpy as np
import os
import json
from DanceDataset import DanceDataset
from MotionVae import MotionVae
from MusicVae import MusicVae
from tensorflow.python import pywrap_tensorflow


class VAE_LSTM_model:
    def __init__(self,
                 train_file_list,
                 model_save_dir,
                 log_dir,
                 rnn_unit_size=32,
                 acoustic_dim=16,
                 temporal_dim=3,
                 motion_dim=63,
                 time_step=120,
                 batch_size=10,
                 learning_rate=1e-3,
                 extr_loss_threshold=0.045,
                 overlap=True,
                 epoch_size=1500,
                 use_mask=True,
                 ):
        # lstm
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.dense_dim = 8
        self.lstm1_input_dim = 16
        self.lstm1_output_dim = 16
        self.rnn_unit_size = rnn_unit_size
        self.acoustic_dim = acoustic_dim
        self.temporal_dim = temporal_dim
        self.motion_dim = motion_dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.extr_loss_threshold = extr_loss_threshold
        self.train_file_list = train_file_list
        self.overlap = overlap
        self.epoch_size = epoch_size
        self.use_mask = use_mask

        # vae
        self.n_hidden = 500
        self.music_latent_dim = 10
        self.motion_latent_dim = 16
        self.rnn_input_dim=32
        self.lstm_output_dim=32

        self.train_dataset = DanceDataset(train_file_list=self.train_file_list,
                                          acoustic_dim=self.acoustic_dim,
                                          temporal_dim=self.temporal_dim,
                                          motion_dim=self.motion_dim,
                                          time_step=self.time_step,
                                          overlap=self.overlap,
                                          overlap_interval=10,
                                          batch_size=self.batch_size)

        self.dense1_dim = 8

        self.musicVae = MusicVae()
        self.motionVae = MotionVae()

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_unit_size)

    def lstm_block(self,input,time_step,input_dim,rnn_input_dim,rnn_unit_size,output_dim,attn_cell,batch_size,name=''):
        with tf.variable_scope(name):
            input = tf.nn.bias_add(tf.matmul(input, tf.Variable(
                tf.truncated_normal([input_dim,rnn_input_dim]))),bias=tf.Variable(tf.zeros(shape=[rnn_input_dim])))
            input = tf.reshape(input, [-1, time_step,rnn_input_dim])
            cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(3)])
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input,
                                                           initial_state=init_state,
                                                           dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, rnn_unit_size])
            lstm = tf.nn.bias_add(
                tf.matmul(output, tf.Variable(tf.truncated_normal([rnn_unit_size, output_dim]))),
                bias=tf.Variable(tf.zeros(shape=[output_dim])))
            lstm = tf.reshape(lstm, [-1, time_step, output_dim])
        return lstm

    def acoustic_features_extractor(self, acoustic_input, acoustic_target, temporal_input, mask_input, trainable):
        batch_size = tf.shape(acoustic_input)[0]
        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )
        # ----------------------------------dense 1-------------------------------------
        with tf.variable_scope("dense1"):
            dense1=tf.layers.dense(acoustic_input,self.dense_dim,activation=tf.nn.relu,trainable=trainable)

            lstm2_input=tf.concat([dense1, temporal_input], 2)
            lstm2_input=tf.reshape(lstm2_input, [-1, self.dense_dim + self.temporal_dim])
        # ----------------------------------lstm2-------------------------------------
        lstm2=self.lstm_block(input=lstm2_input,
                              time_step=self.time_step,
                              input_dim=self.dense_dim + self.temporal_dim,
                              rnn_input_dim=self.rnn_input_dim,
                              rnn_unit_size=self.rnn_unit_size,
                              output_dim=self.lstm1_output_dim,
                              attn_cell=attn_cell,
                              batch_size=batch_size,
                              name='lstm2')
        # ----------------------------------dense 3  and sample-------------------------------------
        with tf.variable_scope("dense_3"):
            music_mean = tf.layers.dense(lstm2,
                                     self.music_latent_dim,
                                     activation=None,
                                     trainable=trainable)
            music_sigma= tf.layers.dense(lstm2,
                                     self.music_latent_dim,
                                     activation=None,
                                     trainable=trainable)
            music_sigma=tf.add(1e-6 ,music_sigma)

            music_latent= music_mean + music_sigma * tf.truncated_normal(tf.shape(music_mean))

        # ----------------------------------mask 4-------------------------------------
        with tf.variable_scope("mask_4"):
            if self.use_mask:
                reduced_music_latent = music_latent
            else:
                mask = mask_input
                reduced_music_latent = tf.multiply(music_latent, mask)

        # ----------------------------------dense 5-------------------------------------
        with tf.variable_scope("dense_5"):
            dense5 = tf.layers.dense(reduced_music_latent,
                                         self.dense_dim,
                                         activation=tf.nn.relu,
                                         trainable=trainable)
            lstm6_input = tf.concat([dense5, temporal_input], 2)
            lstm6_input = tf.reshape(lstm6_input, [-1, self.dense_dim + self.temporal_dim])
            motion_latent_lstm_input=lstm6_input

        lstm6=self.lstm_block(input=lstm6_input,
                                time_step=self.time_step,
                                input_dim=self.dense_dim + self.temporal_dim,
                                rnn_input_dim=self.rnn_input_dim,
                                rnn_unit_size=self.rnn_unit_size,
                                output_dim=self.lstm1_output_dim,
                                attn_cell=attn_cell,
                                batch_size=batch_size,
                                name='lstm6')
        # ----------------------------------dense 8-------------------------------------
        with tf.variable_scope("dense_7"):
            decoded_acoustic_features = tf.layers.dense(lstm6,
                                                        self.acoustic_dim,
                                                        activation=None,
                                                        trainable=trainable)

        motion_latent = self.lstm_block(input=motion_latent_lstm_input,
                                time_step=self.time_step,
                                input_dim=self.dense_dim + self.temporal_dim,
                                rnn_input_dim=self.rnn_input_dim,
                                rnn_unit_size=self.rnn_unit_size,
                                output_dim=self.motion_latent_dim,
                                attn_cell=attn_cell,
                                batch_size=batch_size,
                                name='lstm_motion_latent')

        return motion_latent, decoded_acoustic_features

    def motion_predictor(self, motion_latent, temporal_input, trainable=True):
        batch_size = tf.shape(motion_latent)[0]

        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )
        # ----------------------------------dense 9-------------------------------------
        with tf.variable_scope("dense_9"):
            dense9 = tf.layers.dense(motion_latent,
                                     self.dense_dim,
                                     activation=tf.nn.relu,
                                     trainable=trainable)
            lstm_10_input = tf.concat([dense9, temporal_input], 2)
            lstm_10_input = tf.reshape(lstm_10_input, [-1, self.dense_dim + self.temporal_dim])

        # ----------------------------------lstm 10-------------------------------------
        lstm_10 = self.lstm_block(input=lstm_10_input,
                                  time_step=self.time_step,
                                  input_dim=self.dense_dim + self.temporal_dim,
                                  rnn_input_dim=self.rnn_input_dim,
                                  rnn_unit_size=self.rnn_unit_size,
                                  output_dim=48,
                                  attn_cell=attn_cell,
                                  batch_size=batch_size,
                                  name='lstm_10')




        # ----------------------------------dense 11-------------------------------------
        with tf.variable_scope("dense_11"):
            predicted_motion_features = tf.layers.dense(lstm_10,
                                                        self.motion_dim,
                                                        activation=None,
                                                        trainable=trainable)
        return predicted_motion_features

    def train(self, resume=False):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])

        motion_latent, decoded_acoustic_features = self.acoustic_features_extractor(acoustic, acoustic, temporal, mask, trainable=True)
        predicted_motion_features = self.motion_predictor(motion_latent, temporal,trainable=True)

        # loss
        loss_extr = tf.losses.mean_squared_error(decoded_acoustic_features, acoustic)

        loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
        loss = tf.maximum(self.extr_loss_threshold, loss_extr) + loss_pred
        tf.summary.scalar("loss_extr", loss_extr)
        tf.summary.scalar("loss_pred", loss_pred)
        tf.summary.scalar("loss", loss)

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)  # 自行选择优化器

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        iterator = self.train_dataset.train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        step_num = self.train_dataset.train_size // self.batch_size
        print("step_size: %d" % (step_num))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume:
                ckpt = tf.train.get_checkpoint_state(self.model_save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("restore weight from %s ..." % self.model_save_dir)
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
                    temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]
                    motion_in = batch_data[:, :,
                                self.acoustic_dim + self.temporal_dim:self.acoustic_dim + self.temporal_dim + self.motion_dim]
                    mask_in = temporal_in[:, :, 1]
                    mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                    _, loss_, loss_e, loss_p, sum = sess.run(
                        [train_op, loss, loss_extr, loss_pred, summ],
                        feed_dict={
                            acoustic: acoustic_in,
                            temporal: temporal_in,
                            motion: motion_in,
                            mask: mask_in
                        })
                    loss_avg += loss_
                    if step % 10 == 0:
                        print(
                            "epoch: %d step: %d, total loss: %.9f, extr loss: %.9f, predict loss: %.9f " % (
                                i, step, loss_, loss_e, loss_p))
                writer.add_summary(sum, i)
                print("epoch: %d loss_avg: %f, " % (i, loss_avg / step))
                if (i + 1) % 10 == 0:
                    print("保存模型：", saver.save(sess, os.path.join(self.model_save_dir, 'stock2.model'), global_step=i))

            writer.close()

    def predict(self, test_file, result_save_dir):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])
        test_dataset, train_motion_scaler, test_size, center = self.train_dataset.load_test_data(test_file)

        motion_latent, decoded_acoustic_features = self.acoustic_features_extractor(acoustic, acoustic, temporal, mask,
                                                                                    trainable=True)
        predicted_motion_features = self.motion_predictor(motion_latent, temporal, trainable=True)

        loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)

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
            loss = 0
            if (test_size % self.batch_size == 0):
                test_size = test_size // self.batch_size
            else:
                test_size = test_size // self.batch_size + 1
            for step in range(test_size):
                batch_data = sess.run(next_element)
                batch_data.reshape([-1, self.time_step, self.acoustic_dim + self.temporal_dim + self.motion_dim])
                acoustic_in = batch_data[:, :, :self.acoustic_dim]
                temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]
                motion_in = batch_data[:, :,
                            self.acoustic_dim + self.temporal_dim:self.acoustic_dim + self.temporal_dim + self.motion_dim]
                mask_in = temporal_in[:, :, 1]
                mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                prob, loss_ = sess.run([predicted_motion_features, loss_pred], feed_dict={acoustic: acoustic_in,
                                                                                          temporal: temporal_in,
                                                                                          motion: motion_in,
                                                                                          mask: mask_in})

                predict = prob.reshape((-1, self.motion_dim))
                motion_in = motion_in.reshape((-1, self.motion_dim))
                test_predict = np.append(test_predict, predict, axis=0)
                motion_test = np.append(motion_test, motion_in, axis=0)
                loss += loss_

            test_predict = train_motion_scaler.inverse_transform(test_predict)
            motion_test = train_motion_scaler.inverse_transform(motion_test)
            acc = np.average(np.abs(test_predict - motion_test))

            test_predict = np.reshape(test_predict, [-1, self.motion_dim // 3, 3])
            length = test_predict.shape[0]
            test_predict = test_predict.tolist()
            center = center.tolist()
            data = {"length": length, "skeletons": test_predict, "center": center}
            with open(os.path.join(result_save_dir, file_name), 'w') as file_object:
                json.dump(data, file_object)
            print(loss, acc)