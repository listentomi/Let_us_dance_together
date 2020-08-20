import tensorflow as tf
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('..')
from data_prepare.feature_extract import audio_feature_extract, motion_feature_extract

rnn_unit = 32
rnn_size = 32
# 特征的维度
input_acoustic_size = 16
input_temporal_size = 3
input_mask_size = 1
input_motion_size = 63
# lstm输出
lstm_output_size = 16

time_step = 120
batch_size = 10

output_motion_size = 63
# learning rate
lr = 1e-3
# 计算loss时的threshold
Eth = 0.045

is_overlap = True

def load_features_from_dir(data_dir,over_write=False):
    acoustic_features,temporal_indexes = audio_feature_extract(data_dir,over_write=over_write)  # [n_samples, n_acoustic_features]
    motion_features,center = motion_feature_extract(data_dir, with_rotate=True, with_centering=False)  # [n_samples, n_motion_features]]
    return acoustic_features,temporal_indexes, motion_features[:acoustic_features.shape[0],:],center

def load_train_features_and_scaler(train_dirs, acoustic_size, temporal_size, output_size, acoustic_scaler=MinMaxScaler(), motion_scaler=MinMaxScaler() ):

    train_acoustic_features = np.empty([0,input_acoustic_size])
    train_temporal_features = np.empty([0,temporal_size])
    train_motion_features = np.empty([0,output_motion_size])
    train_acoustic_scaler = acoustic_scaler
    train_motion_scaler = motion_scaler
    train_temporal_scaler= MinMaxScaler()

    for one_dir in train_dirs:
        acoustic_features,temporal_indexes, motion_features,_ = load_features_from_dir(one_dir)
        #  [n_samples, n_acoustic_features]

        train_acoustic_features = np.append(train_acoustic_features,acoustic_features,axis=0)
        train_motion_features = np.append(train_motion_features,motion_features[:,:output_motion_size],axis=0)
        train_temporal_features = np.append(train_temporal_features, temporal_indexes, axis=0)


    train_acoustic_features = train_acoustic_scaler.fit_transform(train_acoustic_features)
    train_motion_features = train_motion_scaler.fit_transform(train_motion_features)
    train_temporal_features=train_temporal_scaler.fit_transform(train_temporal_features)

    assert (len(train_acoustic_features) == len(train_motion_features) == len(train_temporal_features))
    num_train_seq = int(len(train_acoustic_features) / time_step)
    train_acoustic_features = train_acoustic_features[:num_train_seq * time_step, :]
    train_motion_features = train_motion_features[:num_train_seq * time_step, :]
    train_temporal_features = train_temporal_features[:num_train_seq * time_step, :]

    if is_overlap:
        temp_acoustic_features = train_acoustic_features
        temp_motion_features = train_motion_features
        temp_temporal_features = train_temporal_features
        for i in range(1, time_step // 10 - 1):
            temp_acoustic_features = np.concatenate(
                (temp_acoustic_features, train_acoustic_features[10 * i:(num_train_seq - 1) * time_step + 10 * i, :]), axis=0)
            temp_motion_features = np.concatenate(
                (temp_motion_features, train_motion_features[10 * i:(num_train_seq - 1) * time_step + 10 * i, :]), axis=0)
            temp_temporal_features = np.concatenate(
                (temp_temporal_features, train_temporal_features[10 * i:(num_train_seq - 1) * time_step + 10 * i, :]), axis=0)
        train_acoustic_features=temp_acoustic_features
        train_motion_features=temp_motion_features
        train_temporal_features=temp_temporal_features
    num_train_seq = int(len(train_acoustic_features) / time_step)
    normalized_acoustic_data = train_acoustic_features.reshape(num_train_seq, time_step, -1)
    normalized_temporal_data = train_temporal_features.reshape(num_train_seq, time_step, -1)
    normalized_motion_data = train_motion_features.reshape(num_train_seq, time_step, -1)
    print("train size: %d" % (len(train_acoustic_features)))
    all_data = np.concatenate(
        (normalized_acoustic_data, normalized_temporal_data, normalized_motion_data), axis=2)

    train_dataset = tf.data.Dataset.from_tensor_slices(all_data)

    return train_dataset, len(normalized_acoustic_data), train_acoustic_scaler,  train_motion_scaler,train_temporal_scaler




# 获取测试数据





def lstm_cell():  # 定义默认的LSTM单元
    return tf.nn.rnn_cell.LSTMCell(rnn_size)  # 代表接受和返回的state将是2-tuple的形式


# acoustic features extractor
def acoustic_features_extractor(acoustic_input, temporal_input, mask_input, trainable=True,use_mask=True):
    batch_size = tf.shape(acoustic_input)[0]
    attn_cell = lstm_cell
    if trainable:
        def attn_cell():
            return tf.contrib.rnn.DropoutWrapper(
                lstm_cell()
            )

    # ----------------------------------dense 1-------------------------------------
    with tf.variable_scope("dense_1") as scope:
        # acoustic_input 20 24
        dense_1 = tf.layers.dense(acoustic_input, 24, activation=tf.nn.relu, trainable=trainable)

    # ----------------------------------lstm 2-------------------------------------
    with tf.variable_scope("lstm_2") as scope:

        concat2 = tf.concat([dense_1, temporal_input], 2)  # concat:20 27
        concat2 = tf.reshape(concat2, [-1, 27])
        concat_rnn2 = tf.nn.bias_add(tf.matmul(concat2, tf.Variable(tf.truncated_normal([27, rnn_unit]))),
                                     bias=tf.Variable(tf.zeros(shape=[rnn_unit])))
        # concat_rnn2=tf.matmul(concat2,tf.Variable(tf.random_normal([27, rnn_unit])))+ tf.Variable(tf.constant(0.01, shape=[rnn_unit, ])) #300*3
        concat_rnn2 = tf.reshape(concat_rnn2, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入  20*15*3

        cell2 = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(3)])
        init_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
        output_rnn2, final_states2 = tf.nn.dynamic_rnn(cell2, concat_rnn2, initial_state=init_state2,
                                                       dtype=tf.float32)
        output2 = tf.reshape(output_rnn2, [-1, rnn_size])  # 300*3
        pred2 = tf.nn.bias_add(tf.matmul(output2, tf.Variable(tf.truncated_normal([rnn_size, 32]))),
                               bias=tf.Variable(tf.zeros(shape=[32])))
        pred2 = tf.reshape(pred2, [-1, time_step, 32])
        # pred2= tf.matmul(output2, tf.Variable(tf.random_normal([rnn_unit, 32]))) + tf.Variable(tf.constant(0.01, shape=[32, ]))  # 300*16
    # ----------------------------------dense 3-------------------------------------
    with tf.variable_scope("dense_3") as scope:
        dense3 = tf.layers.dense(pred2, 10, activation=None, trainable=trainable)  # 300*16

    # ----------------------------------mask 4-------------------------------------
    with tf.variable_scope("mask_4") as scope:
        # mask 300 1
        if use_mask:
            reduced_acoustic_features = dense3
        else:
            mask = mask_input

            # mask 300 16
            reduced_acoustic_features = tf.multiply(dense3, mask)
    # ----------------------------------dense 5-------------------------------------
    with tf.variable_scope("dense_5") as scope:
        dense5 = tf.layers.dense(reduced_acoustic_features, 24, activation=tf.nn.relu, trainable=trainable)  # 300*16

    # ----------------------------------lstm 7-------------------------------------
    with tf.variable_scope("lstm_6") as scope:
        concat7 = tf.concat([dense5, temporal_input], 2)
        # concat:300 16+3=19
        concat7 = tf.reshape(concat7, [-1, 27])
        concat_rnn7 = tf.nn.bias_add(tf.matmul(concat7, tf.Variable(tf.truncated_normal([27, rnn_unit]))),
                                     bias=tf.Variable(tf.zeros(shape=[rnn_unit])))

        # concat_rnn7 = tf.matmul(concat7, tf.Variable(tf.random_normal([27, rnn_unit]))) + tf.Variable(tf.constant(0.01, shape=[rnn_unit, ]))  # 300*3
        concat_rnn7 = tf.reshape(concat_rnn7, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入  20*15*3
        cell7 = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(3)])
        init_state7 = cell7.zero_state(batch_size, dtype=tf.float32)
        output_rnn7, final_states7 = tf.nn.dynamic_rnn(cell7, concat_rnn7, initial_state=init_state7,
                                                       dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        # output_rnn 20，15，3
        output7 = tf.reshape(output_rnn7, [-1, rnn_size])  # 300*3

        pred7 = tf.nn.bias_add(tf.matmul(output7, tf.Variable(tf.truncated_normal([rnn_size, 32]))),
                               bias=tf.Variable(tf.zeros(shape=[32])))
        pred7 = tf.reshape(pred7, [-1, time_step, 32])
        # pred7 = tf.matmul(output7, tf.Variable(tf.random_normal([rnn_unit, 32]))) + tf.Variable(tf.constant(0.01, shape=[32, ]))  # 300*16

    # ----------------------------------dense 8-------------------------------------
    with tf.variable_scope("dense_8") as scope:
        decoded_acoustic_features = tf.layers.dense(pred7, input_acoustic_size, activation=None, trainable=trainable)

        # 300 16               # 300 16
    return reduced_acoustic_features, decoded_acoustic_features


# motion predictor

def motion_predictor(reduced_acoustic_features, decoded_acoustic_features, temporal_input, trainable=True):
    batch_size = tf.shape(reduced_acoustic_features)[0]

    attn_cell = lstm_cell
    if trainable:
        def attn_cell():
            return tf.contrib.rnn.DropoutWrapper(
                lstm_cell()
            )

    with tf.variable_scope("dense_9") as scope:
        dense9 = tf.layers.dense(reduced_acoustic_features, 24, activation=tf.nn.relu, trainable=trainable)
    with tf.variable_scope("lstm_10") as scope:
        concat10 = tf.concat([dense9, temporal_input], 2)  # 300*35
        concat10 = tf.reshape(concat10, [-1, 27])
        concat_rnn10 = tf.nn.bias_add(tf.matmul(concat10, tf.Variable(tf.truncated_normal([27, rnn_unit]))),
                                      bias=tf.Variable(tf.zeros(shape=[rnn_unit])))

        # concat_rnn10 = tf.matmul(concat10, tf.Variable(tf.random_normal([27, rnn_unit]))) + tf.Variable(tf.constant(0.01, shape=[rnn_unit, ]))  # 300*3
        concat_rnn10 = tf.reshape(concat_rnn10, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入  20*15*3
        cell10 = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(3)])
        init_state10 = cell10.zero_state(batch_size, dtype=tf.float32)
        output_rnn10, final_states10 = tf.nn.dynamic_rnn(cell10, concat_rnn10, initial_state=init_state10,
                                                         dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        # output_rnn 20，15，3
        output10 = tf.reshape(output_rnn10, [-1, rnn_size])  # 300*3
        pred10 = tf.nn.bias_add(tf.matmul(output10, tf.Variable(tf.truncated_normal([rnn_size, 32]))),
                                bias=tf.Variable(tf.zeros(shape=[32])))

        pred10 = tf.reshape(pred10, [-1, time_step, 32])
        # pred10 = tf.matmul(output10, tf.Variable(tf.random_normal([rnn_unit, 32]))) +  tf.Variable(tf.constant(0.01, shape=[32, ]))  # 300*63
    with tf.variable_scope("dense_11") as scope:
        predicted_motion_features = tf.layers.dense(pred10, output_motion_size, activation=None, trainable=trainable)

    return predicted_motion_features


def train_lstm(resume,use_mask,train_dirs):
    acoustic = tf.placeholder(tf.float32, shape=[None, time_step, input_acoustic_size])
    temporal = tf.placeholder(tf.float32, shape=[None, time_step, input_temporal_size])
    motion = tf.placeholder(tf.float32, shape=[None, time_step, input_motion_size])
    mask = tf.placeholder(tf.float32, shape=[None, time_step, 1])

    train_dataset, data_size, train_acoustic_scaler,  train_motion_scaler,train_temporal_scaler=load_train_features_and_scaler(train_dirs=train_dirs,
                                       acoustic_size=input_acoustic_size,temporal_size=3,output_size=output_motion_size,
                                       acoustic_scaler=MinMaxScaler(), motion_scaler=MinMaxScaler())
    epoch_size = 1500

    reduced_acoustic_features, decoded_acoustic_features = acoustic_features_extractor(acoustic, temporal, mask,
                                                                                       trainable=True,use_mask=use_mask)
    predicted_motion_features = motion_predictor(reduced_acoustic_features, decoded_acoustic_features, temporal,
                                                 trainable=True)

    # 损失函数
    loss_extr = tf.losses.mean_squared_error(decoded_acoustic_features, acoustic)
    loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
    # loss_pred=tf.norm(predicted_motion_features-motion, ord='euclidean')
    loss = tf.maximum(Eth, loss_extr) + loss_pred
    tf.summary.scalar("loss_extr", loss_extr)
    tf.summary.scalar("loss_pred", loss_pred)
    tf.summary.scalar("loss", loss)

    # =loss_pred
    train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=1000000)
    iterator = train_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    print("step_size: %d" % (data_size // batch_size))
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        if resume:
            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and ckpt.model_checkpoint_path:
                print("restore weight from ./model ...")
                saver.restore(sess, ckpt.model_checkpoint_path)
        writer = tf.summary.FileWriter("./train_nn_log", sess.graph)
        writer.add_graph(sess.graph)
        summ = tf.summary.merge_all()
        # 重复训练200次
        for i in range(epoch_size):
            print("epoch:%d" % i)
            # 每次进行训练的12时候，每个batch训练batch_size个样本
            loss_avg = 0
            sess.run(iterator.initializer)
            for step in range(data_size // batch_size):
                batch_data = sess.run(next_element)
                acoustic_in = batch_data[:, :, :input_acoustic_size]
                temporal_in = batch_data[:, :, input_acoustic_size:input_acoustic_size + input_temporal_size]
                motion_in = batch_data[:, :,
                            input_acoustic_size + input_temporal_size:input_acoustic_size + input_temporal_size + input_motion_size]
                mask_in = temporal_in[:, :, 1]
                mask_in = np.reshape(mask_in, [-1, time_step, 1])

                _, loss_, loss_e, loss_p, sum = sess.run([train_op, loss, loss_extr, loss_pred, summ], feed_dict={
                    acoustic: acoustic_in,
                    temporal: temporal_in,
                    motion: motion_in,
                    mask: mask_in})
                loss_avg += loss_
                if step % 10 == 0:
                    print("epoch: %d step: %d, total loss: %.9f, extr loss: %.9f predict loss: %.9f " % (
                    i, step, loss_, loss_e, loss_p))
            writer.add_summary(sum, i)
            print("epoch: %d loss_avg: %f, " % (i, loss_avg / (data_size // batch_size)))
            if (i + 1) % 10 == 0:
                print("保存模型：", saver.save(sess, 'model/stock2.model', global_step=i))

        writer.close()


# ————————————————预测模型————————————————————
def prediction(test_file,train_acoustic_scaler, train_motion_scaler, train_temporal_scaler):
    global motion_features_test
    acoustic = tf.placeholder(tf.float32, shape=[None, time_step, input_acoustic_size])
    temporal = tf.placeholder(tf.float32, shape=[None, time_step, input_temporal_size])
    motion = tf.placeholder(tf.float32, shape=[None, time_step, input_motion_size])
    mask = tf.placeholder(tf.float32, shape=[None, time_step, input_mask_size])



    test_acoustic_features, temporal_indexes, test_motion_features,center = load_features_from_dir(test_file,over_write=True)

    test_motion_features=test_motion_features[:,:63]


    temporal_indexes = train_temporal_scaler.transform(temporal_indexes)
    test_acoustic_features = train_acoustic_scaler.transform(test_acoustic_features)
    test_motion_features = train_motion_scaler.transform(test_motion_features)



    assert (len(temporal_indexes) == len(test_acoustic_features) == len(test_motion_features))
    num_train_seq = int(len(temporal_indexes) / time_step)

    test_temporal_indexes = temporal_indexes[:num_train_seq * time_step, :]
    test_acoustic_features = test_acoustic_features[:num_train_seq * time_step, :]
    test_motion_features = test_motion_features[:num_train_seq * time_step, :]

    normalized_acoustic_data = test_acoustic_features.reshape(num_train_seq, time_step, -1)
    normalized_temporal_data = test_temporal_indexes.reshape(num_train_seq, time_step, -1)
    normalized_motion_data = test_motion_features.reshape(num_train_seq, time_step, -1)
    print("test size: %d" % (len(normalized_acoustic_data)))
    all_data = np.concatenate(
        (normalized_acoustic_data, normalized_temporal_data, normalized_motion_data), axis=2)

    test_dataset = tf.data.Dataset.from_tensor_slices(all_data)
    test_dataset= test_dataset.batch(batch_size)

    reduced_acoustic_features, decoded_acoustic_features = acoustic_features_extractor(acoustic, temporal, mask,
                                                                                       trainable=False)
    predicted_motion_features = motion_predictor(reduced_acoustic_features, decoded_acoustic_features, temporal,
                                                 trainable=False)
    loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
    saver = tf.train.Saver(tf.global_variables())
    test_step_size=len(normalized_acoustic_data) // batch_size
    if(len(normalized_acoustic_data) % batch_size!=0):
        test_step_size+=1
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        file_name = os.path.basename(test_file)+'.json'
        print("test the file %s" % file_name)
        iterator = test_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        test_predict =  np.empty([0, input_motion_size])
        motion_test = np.empty([0, input_motion_size])
        sess.run(iterator.initializer)
        loss=0
        for step in range(test_step_size):
            batch_data = sess.run(next_element)
            batch_data.reshape([-1, time_step, input_acoustic_size + input_temporal_size + input_motion_size])
            acoustic_in = batch_data[:, :, :input_acoustic_size]
            temporal_in = batch_data[:, :, input_acoustic_size:input_acoustic_size + input_temporal_size]
            motion_in = batch_data[:, :,
                        input_acoustic_size + input_temporal_size:input_acoustic_size + input_temporal_size + input_motion_size]

            mask_in = batch_data[:, :, -1]
            mask_in = np.reshape(mask_in, [-1, time_step, 1])

            prob,loss_ = sess.run([predicted_motion_features,loss_pred], feed_dict={acoustic: acoustic_in,
                                                                  temporal: temporal_in,
                                                                  motion: motion_in,
                                                                  mask: mask_in})

            predict = prob.reshape((-1,input_motion_size))
            motion_in = motion_in.reshape((-1,input_motion_size))
            test_predict=np.append(test_predict, predict,axis=0)
            motion_test=np.append(motion_test, motion_in,axis=0)
            loss+=loss_

        test_predict = train_motion_scaler.inverse_transform(test_predict)
        motion_test=train_motion_scaler.inverse_transform(motion_test)
        acc = np.average(np.abs(test_predict- motion_test))

        test_predict = np.reshape(test_predict, [-1, 21, 3])
        length = test_predict.shape[0]
        test_predict = test_predict.tolist()
        center=center.tolist()
        data = {"length": length, "skeletons": test_predict,"center": center}
        with open('../result/' + file_name, 'w') as file_object:
            json.dump(data, file_object)
        print( loss,acc)







if __name__=='__main__':
    danceType = 'C'
    dataSize = 9
    trainSize = 8
    # validSize = dataSize - trainSize

    testIndex = 9
    train_dirs = []
    valid_dirs = []
    test_dirs = []
    is_train=False
    # for i in range(1, dataSize+1):
    #     if i <= trainSize:
    #         train_dirs.append("../../../Music-to-Dance-Motion-Synthesis/DANCE_" + danceType + "_" + str(i))
    #     else:
    #         test_dirs.append("../../../Music-to-Dance-Motion-Synthesis/DANCE_" + danceType + "_" + str(i))
    train_dirs = [
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_1",
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_1",
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_2",
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_3",
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_4",
            "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_6",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_7",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_C_8",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_1",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_2",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_3",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_4",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_6",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_7",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_8",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_9",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_1",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_2",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_3",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_4",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_6",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_7",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_8",

        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_1",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_2",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_3",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_4",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_6",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_7",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_8",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_9",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_10",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_11",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_12",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_13",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_14",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_16",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_17",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_18",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_19",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_20",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_21",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_22",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_23",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_24",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_26",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_27",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_28",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_29",
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_30",
        ]

    print('test_dirs:', test_dirs)
    print('train_dirs:', train_dirs)
    print('valid_dirs:', valid_dirs)
    test_dirs=[
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_10",
        # "../../../Music-to-Dance-Motion-Synthesis/DANCE_T_9",
        # "../../../Music-to-Dance-Motion-Synthesis/DANCE_W_31",
         #"../../../Music-to-Dance-Motion-Synthesis/DANCE_W_32",
    ]
    if(is_train):
        train_lstm(False,use_mask=True,train_dirs=train_dirs)

    else:
        train_dataset, data_size, train_acoustic_scaler, train_motion_scaler, train_temporal_scaler= load_train_features_and_scaler(
            train_dirs=train_dirs,
            acoustic_size=input_acoustic_size, temporal_size=3, output_size=output_motion_size,
            acoustic_scaler=MinMaxScaler(), motion_scaler=MinMaxScaler())
        for test_file in test_dirs:
            prediction( test_file,train_acoustic_scaler, train_motion_scaler, train_temporal_scaler)