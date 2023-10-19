## Class-guided human motion prediction via multi-spatial-temporal supervision
This is the code for the paper

Jinkai Li, Honghu Pan, Lian Wu, Chao Huang, Xiaoling Luo, Yong Xu,
[_"Class-guided human motion prediction via multi-spatial-temporal supervision"_](https://link.springer.com/article/10.1007/s00521-023-08362-x). In Neural Computing and Applications (NCAA) 2023

### Dependencies

* cuda 11.4
* Python 3.8
* Pytorch 1.7.0

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
h3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.
Directory structure:
```shell script
cmu_mocap
|-- test
|-- train
```

Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 66 --epoch 100
```
```bash
python main_cmu_mocap.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 75  --epoch 100 --alpha 0.15  --beta 0.15 --gama 0.1
```
### Evaluation
To evaluate the pretrained model,
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 66 --ckpt ./checkpoint/pretrained/ckpt_Best_H36M_AverageError55.0655_err10.2935_err22.7684_err48.1131_err59.2875_err77.9491_err111.9817.pth
```
```bash
python main_cmu_mocap_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 75  --ckpt ./checkpoint/pretrained/ckpt_Best_CMU_AverageError39.3981_err8.9772_err16.1292_err31.6335_err40.0811_err55.6205_err83.9470.pth
```

### Citing

If you use our code, please cite our work

```
@article{li2023class,
  title={Class-guided human motion prediction via multi-spatial-temporal supervision},
  author={Li, Jinkai and Pan, Honghu and Wu, Lian and Huang, Chao and Luo, Xiaoling and Xu, Yong},
  journal={Neural Computing and Applications},
  volume={35},
  number={13},
  pages={9463--9479},
  year={2023},
  publisher={Springer}
}
```

### Acknowledgments
The overall code framework is adapted from [_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://github.com/wei-mao-2019/HisRepItself) 
