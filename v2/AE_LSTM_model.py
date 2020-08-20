import tensorflow as tf
import numpy as np
import os
import json
from DanceDataset import DanceDataset


class AE_LSTM_model:
    def __init__(self,
                 train_file_list,
                 model_save_dir,
                 log_dir,
                 rnn_input_dim=32,
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

                 dense_dim=24,
                 lstm_output_dim=32,
                 reduced_size=10,


                 ):
        self.model_save_dir=model_save_dir
        self.log_dir=log_dir
        self.rnn_input_dim = rnn_input_dim
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

        self.train_dataset = DanceDataset(train_file_list=self.train_file_list,
                                          acoustic_dim=self.acoustic_dim,
                                          temporal_dim=self.temporal_dim,
                                          motion_dim=self.motion_dim,
                                          time_step=self.time_step,
                                          overlap=self.overlap,
                                          overlap_interval=10,
                                          batch_size=self.batch_size)

        self.dense_dim = dense_dim
        self.lstm_output_dim=lstm_output_dim
        self.reduced_size=reduced_size

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_unit_size)

    def acoustic_features_extractor(self, acoustic_input, temporal_input, mask_input, trainable=True,use_mask=True):
        batch_size = tf.shape(acoustic_input)[0]
        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )

        # ----------------------------------dense 1-------------------------------------
        with tf.variable_scope("dense_1"):
            dense_1 = tf.layers.dense(acoustic_input,
                                      self.dense_dim,
                                      activation=tf.nn.relu,
                                      trainable=trainable)

        # ----------------------------------lstm 2-------------------------------------
        with tf.variable_scope("lstm_2"):

            concat2 = tf.concat([dense_1, temporal_input], 2)
            concat2 = tf.reshape(concat2, [-1,self.dense_dim + self.temporal_dim])
            concat_rnn2 = tf.nn.bias_add(tf.matmul(concat2,tf.Variable(tf.truncated_normal([self.dense_dim + self.temporal_dim, self.rnn_input_dim]))),
                                         bias=tf.Variable(tf.zeros(shape=[self.rnn_input_dim])))
            concat_rnn2 = tf.reshape(concat_rnn2, [-1, self.time_step, self.rnn_input_dim])
            cell2 = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(3)] )
            init_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
            output_rnn2, final_states2 = tf.nn.dynamic_rnn(cell2, concat_rnn2,
                                                           initial_state=init_state2,
                                                           dtype=tf.float32)
            output2 = tf.reshape(output_rnn2, [-1, self.rnn_unit_size])
            pred2 = tf.nn.bias_add(tf.matmul(output2, tf.Variable(tf.truncated_normal([self.rnn_unit_size, self.lstm_output_dim]))),
                                   bias=tf.Variable(tf.zeros(shape=[self.lstm_output_dim])))
            pred2 = tf.reshape(pred2, [-1, self.time_step, self.lstm_output_dim])

        # ----------------------------------dense 3-------------------------------------
        with tf.variable_scope("dense_3"):
            dense3 = tf.layers.dense(pred2,
                                     self.reduced_size,
                                     activation=None,
                                     trainable=trainable)

        # ----------------------------------mask 4-------------------------------------
        with tf.variable_scope("mask_4"):
            if self.use_mask:
                reduced_acoustic_features = dense3
            else:
                mask = mask_input
                reduced_acoustic_features = tf.multiply(dense3, mask)

        # ----------------------------------dense 5-------------------------------------
        with tf.variable_scope("dense_5"):
            dense5 = tf.layers.dense(reduced_acoustic_features,
                                     self.dense_dim,
                                     activation=tf.nn.relu,
                                     trainable=trainable)

        # ----------------------------------lstm 7-------------------------------------
        with tf.variable_scope("lstm_6"):
            concat7 = tf.concat([dense5, temporal_input], 2)
            concat7 = tf.reshape(concat7, [-1, self.dense_dim+self.temporal_dim])
            concat_rnn7 = tf.nn.bias_add(tf.matmul(concat7, tf.Variable(tf.truncated_normal([27, self.rnn_input_dim]))),
                                         bias=tf.Variable(tf.zeros(shape=[self.rnn_input_dim])))
            concat_rnn7 = tf.reshape(concat_rnn7, [-1, self.time_step, self.rnn_input_dim])
            cell7 = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(3)])
            init_state7 = cell7.zero_state(batch_size, dtype=tf.float32)
            output_rnn7, final_states7 = tf.nn.dynamic_rnn(cell7, concat_rnn7, initial_state=init_state7,
                                                           dtype=tf.float32)
            output7 = tf.reshape(output_rnn7, [-1, self.rnn_unit_size])
            pred7 = tf.nn.bias_add(tf.matmul(output7, tf.Variable(tf.truncated_normal([self.rnn_unit_size, self.lstm_output_dim]))),
                                   bias=tf.Variable(tf.zeros(shape=[self.lstm_output_dim])))
            pred7 = tf.reshape(pred7, [-1, self.time_step, self.lstm_output_dim])

        # ----------------------------------dense 8-------------------------------------
        with tf.variable_scope("dense_8"):
            decoded_acoustic_features = tf.layers.dense(pred7,
                                                        self.acoustic_dim,
                                                        activation=None,
                                                        trainable=trainable)

        return reduced_acoustic_features, decoded_acoustic_features

    def motion_predictor(self, reduced_acoustic_features, temporal_input, trainable=True):
        batch_size = tf.shape(reduced_acoustic_features)[0]

        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )
        # ----------------------------------dense 9-------------------------------------
        with tf.variable_scope("dense_9"):
            dense9 = tf.layers.dense(reduced_acoustic_features,
                                     self.dense_dim,
                                     activation=tf.nn.relu,
                                     trainable=trainable)

        # ----------------------------------lstm 10-------------------------------------
        with tf.variable_scope("lstm_10"):
            concat10 = tf.concat([dense9, temporal_input], 2)
            concat10 = tf.reshape(concat10, [-1, self.dense_dim+self.temporal_dim])
            concat_rnn10 = tf.nn.bias_add(
                tf.matmul(concat10, tf.Variable(tf.truncated_normal([self.dense_dim+self.temporal_dim, self.rnn_input_dim]))),
                bias=tf.Variable(tf.zeros(shape=[self.rnn_input_dim])))
            concat_rnn10 = tf.reshape(concat_rnn10,
                                      [-1, self.time_step, self.rnn_input_dim])
            cell10 = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(3)])
            init_state10 = cell10.zero_state(batch_size, dtype=tf.float32)
            output_rnn10, final_states10 = tf.nn.dynamic_rnn(cell10,
                                                             concat_rnn10,
                                                             initial_state=init_state10,
                                                             dtype=tf.float32)
            output10 = tf.reshape(output_rnn10, [-1, self.rnn_unit_size])
            pred10 = tf.nn.bias_add(tf.matmul(output10,tf.Variable(tf.truncated_normal([self.rnn_unit_size, self.lstm_output_dim]))),
                                    bias=tf.Variable(tf.zeros(shape=[self.lstm_output_dim])))

            pred10 = tf.reshape(pred10, [-1, self.time_step, self.lstm_output_dim])

        # ----------------------------------dense 11-------------------------------------
        with tf.variable_scope("dense_11"):
            predicted_motion_features = tf.layers.dense(pred10,
                                                        self.motion_dim,
                                                        activation=None,
                                                        trainable=trainable)
        return predicted_motion_features

    def train(self,resume=False):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])

        reduced_acoustic_features, decoded_acoustic_features = self.acoustic_features_extractor(acoustic,
                                                                                                temporal,
                                                                                                mask,
                                                                                                trainable=True,
                                                                                                use_mask=self.use_mask)
        predicted_motion_features = self.motion_predictor(reduced_acoustic_features,
                                                          temporal,
                                                          trainable=True)

        # loss
        loss_extr = tf.losses.mean_squared_error(decoded_acoustic_features, acoustic)
        loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
        loss = tf.maximum(self.extr_loss_threshold, loss_extr) + loss_pred
        tf.summary.scalar("loss_extr", loss_extr)
        tf.summary.scalar("loss_pred", loss_pred)
        tf.summary.scalar("loss", loss)

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
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
                    temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]
                    motion_in = batch_data[:, :,
                                self.acoustic_dim + self.temporal_dim:self.acoustic_dim + self.temporal_dim + self.motion_dim]
                    mask_in = temporal_in[:, :, 1]
                    mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                    _, loss_, loss_e, loss_p, sum = sess.run([train_op, loss, loss_extr, loss_pred, summ],
                                                             feed_dict={
                                                                        acoustic: acoustic_in,
                                                                        temporal: temporal_in,
                                                                        motion: motion_in,
                                                                        mask: mask_in
                                                             })
                    loss_avg += loss_
                    if step % 10 == 0:
                        print("epoch: %d step: %d, total loss: %.9f, extr loss: %.9f predict loss: %.9f " % (
                            i, step, loss_, loss_e, loss_p))
                writer.add_summary(sum, i)
                print("epoch: %d loss_avg: %f, " % (i, loss_avg / step))
                if (i + 1) % 10 == 0:
                    print("保存模型：", saver.save(sess,os.path.join(self.model_save_dir,'stock2.model'), global_step=i))

            writer.close()

    def predict(self,test_file,result_save_dir):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])

        test_dataset,train_motion_scaler,test_size,center = self.train_dataset.load_test_data(test_file)



        reduced_acoustic_features, decoded_acoustic_features = self.acoustic_features_extractor(acoustic,
                                                                                                temporal,
                                                                                                mask,
                                                                                                trainable=False)
        predicted_motion_features = self.motion_predictor(reduced_acoustic_features,
                                                          temporal,
                                                          trainable=False)
        loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
        saver = tf.train.Saver(tf.global_variables())
        if (test_size % self.batch_size == 0):
            test_size = test_size // self.batch_size
        else:
            test_size = test_size // self.batch_size + 1
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
            for step in range(test_size):
                batch_data = sess.run(next_element)
                batch_data.reshape([-1, self.time_step,self.acoustic_dim + self.temporal_dim + self.motion_dim])
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

                predict = prob.reshape((-1,  self.motion_dim))
                motion_in = motion_in.reshape((-1,  self.motion_dim))
                test_predict = np.append(test_predict, predict, axis=0)
                motion_test = np.append(motion_test, motion_in, axis=0)
                loss += loss_

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
            print(loss, acc)