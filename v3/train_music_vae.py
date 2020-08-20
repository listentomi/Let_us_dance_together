from MusicVae import  MusicVae

if __name__=="__main__":
    train_dirs = []
    with open('./train_dirs.txt', 'r')as f:
        for line in f.readlines():
            train_dirs.append(line[:-1])

    Model = MusicVae(model_save_dir='./good_result/W/music_vae_model',
                 log_dir='./good_result/W/music_log',
                 train_file_list=train_dirs)
    Model.init_dataset()
    Model.train(resume=True)