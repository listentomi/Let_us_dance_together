from data_prepare.feature_extract import  Music
from VAE_LSTM_FIX_model import VAE_LSTM_FIX_model
import librosa
import json
import numpy as np
import os
from data_prepare.visualize import  draw_predict
from visualization.threeDPoints2Bvh import smartbody_skeleton
def getStandardFrames(frames):
    new_frames = np.zeros([len(frames), 19, 3])
    for i in range(len(frames)):
        # Hips
        new_frames[i][0][0] = frames[i][2][0] * -1
        new_frames[i][0][1] = frames[i][2][1]
        new_frames[i][0][2] = frames[i][2][2]
        # RightHip
        new_frames[i][1][0] = frames[i][16][0] * -1
        new_frames[i][1][1] = frames[i][16][1]
        new_frames[i][1][2] = frames[i][16][2]

        # RightKnee
        new_frames[i][2][0] = frames[i][17][0] * -1
        new_frames[i][2][1] = frames[i][17][1]
        new_frames[i][2][2] = frames[i][17][2]
        # RightAnkle
        new_frames[i][3][0] = frames[i][18][0] * -1
        new_frames[i][3][1] = frames[i][18][1]
        new_frames[i][3][2] = frames[i][18][2]

        # LeftHip
        new_frames[i][4][0] = frames[i][7][0] * -1
        new_frames[i][4][1] = frames[i][7][1]
        new_frames[i][4][2] = frames[i][7][2]
        # LeftKnee
        new_frames[i][5][0] = frames[i][8][0] * -1
        new_frames[i][5][1] = frames[i][8][1]
        new_frames[i][5][2] = frames[i][8][2]
        # LeftAnkle
        new_frames[i][6][0] = frames[i][9][0] * -1
        new_frames[i][6][1] = frames[i][9][1]
        new_frames[i][6][2] = frames[i][9][2]

        temp1 = [(frames[i][12][0] + frames[i][3][0]) / 2, (frames[i][12][1] + frames[i][3][1]) / 2,
                 (frames[i][12][2] + frames[i][3][2]) / 2]
        temp2 = [(frames[i][1][0] + frames[i][0][0]) / 2, (frames[i][1][1] + frames[i][0][1]) / 2,
                 (frames[i][1][2] + frames[i][0][2]) / 2]

        # Spine
        new_frames[i][7][0] = (temp1[0] + frames[i][2][0]) / 2 * -1
        new_frames[i][7][1] = (temp1[1] + frames[i][2][1]) / 2
        new_frames[i][7][2] = (temp1[2] + frames[i][2][2]) / 2
        # Thorax
        new_frames[i][8][0] = temp1[0] * -1
        new_frames[i][8][1] = temp1[1]
        new_frames[i][8][2] = temp1[2]

        # Neck
        new_frames[i][9][0] = (temp1[0] + (temp2[0] - temp1[0]) * 0.5) * -1
        new_frames[i][9][1] = (temp1[1] + (temp2[1] - temp1[1]) * 0.5)
        new_frames[i][9][2] = (temp1[2] + (temp2[2] - temp1[2]) * 0.5)
        # Head
        new_frames[i][10][0] = (temp1[0] + (temp2[0] - temp1[0]) * 1.3) * -1
        new_frames[i][10][1] = (temp1[1] + (temp2[1] - temp1[1]) * 1.3)
        new_frames[i][10][2] = (temp1[2] - (temp2[2] - temp1[2]) * 0.5)

        # LeftShoulder
        new_frames[i][11][0] = frames[i][3][0] * -1
        new_frames[i][11][1] = frames[i][3][1]
        new_frames[i][11][2] = frames[i][3][2]

        # LeftElbow
        new_frames[i][12][0] = frames[i][4][0] * -1
        new_frames[i][12][1] = frames[i][4][1]
        new_frames[i][12][2] = frames[i][4][2]
        # LeftWrist
        new_frames[i][13][0] = frames[i][5][0] * -1
        new_frames[i][13][1] = frames[i][5][1]
        new_frames[i][13][2] = frames[i][5][2]

        # RightShoulder
        new_frames[i][14][0] = frames[i][12][0] * -1
        new_frames[i][14][1] = frames[i][12][1]
        new_frames[i][14][2] = frames[i][12][2]
        # RightElbow
        new_frames[i][15][0] = frames[i][13][0] * -1
        new_frames[i][15][1] = frames[i][13][1]
        new_frames[i][15][2] = frames[i][13][2]

        # RightWrist
        new_frames[i][16][0] = frames[i][14][0] * -1
        new_frames[i][16][1] = frames[i][14][1]
        new_frames[i][16][2] = frames[i][14][2]

        # LeftWristEndSite
        new_frames[i][17][0] = frames[i][6][0] * -1
        new_frames[i][17][1] = frames[i][6][1]
        new_frames[i][17][2] = frames[i][6][2]

        # RightWristEndSite
        new_frames[i][18][0] = frames[i][15][0] * -1
        new_frames[i][18][1] = frames[i][15][1]
        new_frames[i][18][2] = frames[i][15][2]
    return new_frames
hop_length = 512
window_length = hop_length * 2
fps = 25
spf = 0.04  # 40 ms
sample_rate = 44100  #
resample_rate = hop_length * fps

music_dir= '../music/W'
music_name='风流寡妇圆舞曲'
music_path=os.path.join(music_dir,music_name+'.mp3')
duration =librosa.get_duration(filename=music_path)


music = Music(music_path, sr=resample_rate, start=0, duration=duration) # 25fps
acoustic_features, temporal_indexes = music.extract_features()  # 16 dim
acoustic_features_path = os.path.join(music_dir, music_name+"_acoustic_features.npy")
temporal_indexes_path = os.path.join(music_dir, music_name+"_temporal_features.npy")
np.save(acoustic_features_path, acoustic_features)
np.save(temporal_indexes_path, temporal_indexes)


train_dirs = []
with open('train_dirs.txt', 'r')as f:
    for line in f.readlines():
        train_dirs.append(line[:-1])

Model = VAE_LSTM_FIX_model(
        train_file_list=train_dirs,
        model_save_dir='./good_result/W/model',
        log_dir='./good_result/W/train_nn_log',
        motion_vae_ckpt_dir='./good_result/W/motion_vae_model/stock2.model-349',
        music_vae_ckpt_dir='./good_result/W/music_vae_model/stock2.model-269',
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
        use_mask=True)
result_save_dir= '../result/W'
Model.predict_from_music(acoustic_features, temporal_indexes,music_name,result_save_dir=result_save_dir)
motion_path=os.path.join(result_save_dir,music_name+'.json')

draw_predict(motion_path, result_save_dir,music_name,temporal_indexes_path,music_path)

bvh_path=music_path=os.path.join(result_save_dir,music_name+'.bvh')

with open(motion_path, 'r') as fin:
        data = json.load(fin)

frames = np.array(data['skeletons'])
frames = getStandardFrames(frames)
smartbody_skeleton = smartbody_skeleton.SmartBodySkeleton()
smartbody_skeleton.poses2bvh(frames, output_file=bvh_path)



