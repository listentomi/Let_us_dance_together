# readme

## Dependencies

tensorflow-gpu           1.6.0

librosa                  0.6.0

Keras-Preprocessing      1.1.2

numpy                    1.19.1

opencv-contrib-python    4.3.0.36

opencv-python            4.3.0.36

sklearn                  0.0

scikit-video             1.1.11

scikit-image             0.17.2

## dataset

You can download the dataset from: 

https://github.com/mhy12345/Music-to-Dance-Motion-Synthesis

## Train

First of all, please modify the `train_dirs.txt` to prepare the train dataset.

for **v1,v2,v4**, you can run the following `.py` files for training 

`./v1/train.py`

`./v2/train.py`

`./v4/train.py`

for **v4**,  you must run the `./v3/train_motion_vae.py` and `./v3/train_music_vae.py` to get the pretrained VAE, then  run the `./v3/train.py` .

For the details of the network, please modify the model by yourself (e.g., the number of epoches, learning_rate, the model save dir)

## Test

for **v1,v2,v3,v4**, you can run the following `.py` files for testing

`./v1/test.py`

`./v2/test.py`

`./v4/test.py`

we provide `generate_dance_from_music.py` in `./v3` for you to generate a video that contains music and dance. 

You can use the tools I uploaded https://github.com/oneThousand1000/bvh_visualization to visualize the results, those tool can visualize .bvh files.  we will updata the project soon afterwards :)

## TODO

### v1 

论文复现

#### 更新进展

①首先调整了rnn_size=32, time_step=1，训练效果并不好

②rnn_size=640, time_step=120，训练效果依然不好

调整思路，之前的dataloader并没有对数据进行shuffle，并且存在重叠，重新写了一个dataloader用于shuffle batch（无重叠 ）

③重构dataloader以后效果依然不好

检查了loss函数，发现使用`tf.norm`有失偏颇，因为`tf.norm`仅仅是求出向量的范数，这里实际上需要使用`tf.losses.mean_squared_error`，MSEloss更为合适

进行了归一化方法的修改（详见v2）

加入了overlap

放弃训练所有数据，而是对舞蹈种类进行分类训练

### v2

1. minmax方法归一化    （MinMaxScaler）
   1. 改进：之前对于数据的归一化是取整体的minmax，会产生很大的问题。
   2. 比如temporal feature，第一列的数据范围在0-10^2左右，第二列只有{0,1}两种取值，第三列则非常大，从0-10000...，这样进行归一化，第一第二列几乎没有了，相当于输入了一个第一第二维度为0，第三维度均匀分布的矩阵，数据坍缩了。
   3. 又因为temporal的数值没有任何实际的比例意义，所以对每个维度单独进行归一化（这样是没有关系的，因为temporal只与beat有关）
2. 普通方法归一化（均值为0，方差为1）  （StandardScaler）
   1. 收敛效果没有minmax好，但是结果还可以接受

**最后决定使用minmax**

### v3

vae单独训练，motion_vae和music_vae参数提前训练好，并且在网络中固定。

实际上最后的模型训练的是dense和lstm部分

**需要注意的地方：**

- 使用了overlap
- 对不同种类的舞蹈分开训练（包括vae）
- 使用minmax进行归一化
- 使用MSEloss

### v4

vae和lstm一起训练。

收敛效果很差，几乎训练几个epoch以后loss就不下降了，训练非常困难，最后输出结果是一个几乎静止的动作

## Pretrained-model

You can get the pretrained model of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1VxklDyWodBikT-DM9W-pFQ 
提取码：pbso

## Result

You can get the final results of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1lAHzNf4dJj6PNh-ZhSG5XA 
提取码：sk7n

## Paper

我们的工作（v1，v2部分）基于以下论文

```
@inproceedings{tang2018dance,
	title={Dance with Melody: An LSTM-autoencoder Approach to Music-oriented Dance Synthesis},
	author={Tang, Taoran and Jia, Jia and Mao, Hanyang},
	booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
	pages={1598--1606},
	year={2018},
	organization={ACM}
}
@inproceedings{tang2018anidance,
	title={AniDance: Real-Time Dance Motion Synthesize to the Song},
	author={Tang, Taoran and Mao, Hanyang and Jia, Jia},
	booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
	pages={1237--1239},
	year={2018},
	organization={ACM}
}
```

