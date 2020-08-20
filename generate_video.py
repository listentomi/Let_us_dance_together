from data_prepare.visualize import draw_predict


motion_path='./result/Fairy Tale.json'
video_dir='./result'
music_name='Fairy Tale'
tempo_path='./music/Fairy Tale_temporal_features.npy'
draw_predict(motion_path, video_dir,music_name,tempo_path,'music/Fairy Tale.mp3')