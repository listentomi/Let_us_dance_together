from MotionVae import  MotionVae

if __name__=="__main__":
    train_dirs = []
    with open('./train_dirs.txt', 'r')as f:
        for line in f.readlines():
            train_dirs.append(line[:-1])
    test_dirs = [
        "../../../Music-to-Dance-Motion-Synthesis/DANCE_R_10"
    ]
    Model = MotionVae(model_save_dir='./good_result/R/motion_vae_model',
                 log_dir='./good_result/R/motion_log',
                 train_file_list=train_dirs)
    Model.init_dataset()
    for test_file in test_dirs:
        Model.predict(test_file, './result')