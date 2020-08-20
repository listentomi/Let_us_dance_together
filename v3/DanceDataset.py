from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append('..')
import numpy as np
import os
import json
from data_prepare.feature_extract import audio_feature_extract, motion_feature_extract
import tensorflow as tf
import pickle

class DanceDataset:
    def __init__(self,
                 train_file_list,
                 acoustic_dim=16,
                 temporal_dim=3,
                 motion_dim=63,
                 time_step=120,
                 overlap=True,
                 overlap_interval=10,
                 batch_size=10
                 ):
        print("\n...loading training data...\n")
        self.train_file_list = train_file_list
        self.acoustic_dim = acoustic_dim
        self.temporal_dim = temporal_dim
        self.motion_dim = motion_dim
        self.overlap = overlap
        self.time_step = time_step
        self.overlap_interval = overlap_interval
        self.batch_size=batch_size
        self.load_train_data_and_scaler()

    def load_features_from_dir(self, data_dir,over_write=False):
        # load data from data_dir
        acoustic_features, temporal_features = audio_feature_extract(data_dir,over_write=over_write)
        motion_features,center  = motion_feature_extract(data_dir, with_rotate=True, with_centering=False)

        return acoustic_features, temporal_features, motion_features[:acoustic_features.shape[0], :],center

    def load_train_data_and_scaler(self):
        # load data from train_file_list, return train dataset and scaler

        print("\nloading training data, please wait....\n")

        train_acoustic_features = np.empty([0, self.acoustic_dim])
        train_temporal_features = np.empty([0, self.temporal_dim])
        train_motion_features = np.empty([0, self.motion_dim])

        self.train_acoustic_scaler = MinMaxScaler()
        self.train_motion_scaler = MinMaxScaler()
        self.train_temporal_scaler = MinMaxScaler()

        for file_dir in self.train_file_list:

            acoustic_features, temporal_indexes, motion_features,_ = self.load_features_from_dir(file_dir)
            train_acoustic_features = np.append(train_acoustic_features, acoustic_features, axis=0)
            train_motion_features = np.append(train_motion_features, motion_features[:, :self.motion_dim], axis=0)
            train_temporal_features = np.append(train_temporal_features, temporal_indexes, axis=0)

        # normalization
        train_acoustic_features = self.train_acoustic_scaler.fit_transform(train_acoustic_features)
        train_motion_features = self.train_motion_scaler.fit_transform(train_motion_features)
        train_temporal_features = self.train_temporal_scaler.fit_transform(train_temporal_features)

        # resize
        assert (len(train_acoustic_features) == len(train_motion_features) == len(train_temporal_features))
        num_train_time_step = int(len(train_acoustic_features) / self.time_step)
        train_acoustic_features = train_acoustic_features[:num_train_time_step * self.time_step, :]
        train_motion_features = train_motion_features[:num_train_time_step * self.time_step, :]
        train_temporal_features = train_temporal_features[:num_train_time_step * self.time_step, :]

        # overlap
        if self.overlap:
            temp_acoustic_features = train_acoustic_features
            temp_motion_features = train_motion_features
            temp_temporal_features = train_temporal_features
            for i in range(1, self.time_step // self.overlap_interval - 1):
                temp_acoustic_features = np.concatenate(
                    (temp_acoustic_features,
                     train_acoustic_features[10 * i:(num_train_time_step - 1) * self.time_step + 10 * i, :]), axis=0)
                temp_motion_features = np.concatenate(
                    (temp_motion_features,
                     train_motion_features[10 * i:(num_train_time_step - 1) * self.time_step + 10 * i, :]),
                    axis=0)
                temp_temporal_features = np.concatenate(
                    (temp_temporal_features,
                     train_temporal_features[10 * i:(num_train_time_step - 1) * self.time_step + 10 * i, :]), axis=0)
            train_acoustic_features = temp_acoustic_features
            train_motion_features = temp_motion_features
            train_temporal_features = temp_temporal_features

        # reshape
        num_train_time_step = int(len(train_acoustic_features) / self.time_step)
        normalized_acoustic_data = train_acoustic_features.reshape(num_train_time_step, self.time_step, -1)
        normalized_temporal_data = train_temporal_features.reshape(num_train_time_step, self.time_step, -1)
        normalized_motion_data = train_motion_features.reshape(num_train_time_step, self.time_step, -1)
        print("train size: %d" % (len(train_acoustic_features)))

        all_data = np.concatenate(
            (normalized_acoustic_data, normalized_temporal_data, normalized_motion_data), axis=2)

        train_dataset = tf.data.Dataset.from_tensor_slices(all_data)

        self.train_size = len(normalized_acoustic_data)

        train_dataset = train_dataset.batch(self.batch_size).shuffle(buffer_size=1000000)

        self.train_dataset = train_dataset

    def load_test_data(self, test_file,start):


        test_acoustic_features, test_temporal_indexes, test_motion_features,center = self.load_features_from_dir(test_file,over_write=True)
        test_motion_features = test_motion_features[:, :self.motion_dim]

        test_temporal_indexes = self.train_temporal_scaler.transform(test_temporal_indexes)
        test_acoustic_features = self.train_acoustic_scaler.transform(test_acoustic_features)
        test_motion_features = self.train_motion_scaler.transform(test_motion_features)

        assert (len(test_temporal_indexes) == len(test_acoustic_features) == len(test_motion_features))
        num_test_time_step = int(len(test_temporal_indexes) / self.time_step)
        if start!=0:
            test_temporal_indexes = test_temporal_indexes[start:(num_test_time_step-1) * self.time_step+start, :]
            test_acoustic_features = test_acoustic_features[start:(num_test_time_step-1) * self.time_step+start, :]
            test_motion_features = test_motion_features[start:(num_test_time_step-1) * self.time_step+start, :]
            normalized_acoustic_data = test_acoustic_features.reshape(num_test_time_step-1, self.time_step, -1)
            normalized_temporal_data = test_temporal_indexes.reshape(num_test_time_step-1, self.time_step, -1)
            normalized_motion_data = test_motion_features.reshape(num_test_time_step-1, self.time_step, -1)
        else:
            test_temporal_indexes = test_temporal_indexes[:num_test_time_step * self.time_step, :]
            test_acoustic_features = test_acoustic_features[:num_test_time_step * self.time_step, :]
            test_motion_features = test_motion_features[:num_test_time_step * self.time_step, :]

            normalized_acoustic_data = test_acoustic_features.reshape(num_test_time_step, self.time_step, -1)
            normalized_temporal_data = test_temporal_indexes.reshape(num_test_time_step, self.time_step, -1)
            normalized_motion_data = test_motion_features.reshape(num_test_time_step, self.time_step, -1)


        print("test size: %d" % (len(normalized_acoustic_data)))

        all_data = np.concatenate(
            (normalized_acoustic_data, normalized_temporal_data, normalized_motion_data), axis=2)

        test_dataset = tf.data.Dataset.from_tensor_slices(all_data)
        test_dataset = test_dataset.batch(self.batch_size)
        return test_dataset,self.train_motion_scaler,len(normalized_acoustic_data),center

    def generate_test_data(self, acoustic_features, temporal_indexes, start):
        test_acoustic_features, test_temporal_indexes = acoustic_features, temporal_indexes
        center = [69.42719559087985, 89.64205651632169, 62.5705835854135]

        test_temporal_indexes = self.train_temporal_scaler.transform(test_temporal_indexes)
        test_acoustic_features = self.train_acoustic_scaler.transform(test_acoustic_features)

        assert (len(test_temporal_indexes) == len(test_acoustic_features))
        num_test_time_step = int(len(test_temporal_indexes) / self.time_step)
        if start != 0:
            test_temporal_indexes = test_temporal_indexes[start:(num_test_time_step - 1) * self.time_step + start, :]
            test_acoustic_features = test_acoustic_features[start:(num_test_time_step - 1) * self.time_step + start, :]
            normalized_acoustic_data = test_acoustic_features.reshape(num_test_time_step - 1, self.time_step, -1)
            normalized_temporal_data = test_temporal_indexes.reshape(num_test_time_step - 1, self.time_step, -1)
        else:
            test_temporal_indexes = test_temporal_indexes[:num_test_time_step * self.time_step, :]
            test_acoustic_features = test_acoustic_features[:num_test_time_step * self.time_step, :]

            normalized_acoustic_data = test_acoustic_features.reshape(num_test_time_step, self.time_step, -1)
            normalized_temporal_data = test_temporal_indexes.reshape(num_test_time_step, self.time_step, -1)

        print("test size: %d" % (len(normalized_acoustic_data)))

        all_data = np.concatenate(
            (normalized_acoustic_data, normalized_temporal_data), axis=2)

        test_dataset = tf.data.Dataset.from_tensor_slices(all_data)
        test_dataset = test_dataset.batch(self.batch_size)
        return test_dataset, self.train_motion_scaler, len(normalized_acoustic_data), center