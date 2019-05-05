# octconv.pytorch
[PyTorch](pytorch.org) implementation of Octave Convolution in [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049)

## ResNet-50 on ImageNet
| Architecture             | LR decay strategy   | Parameters | GFLOPs | Top-1 / Top-5 Error (%) |
| ------------------------ | ------------------- | ---------- | ------ | ----------------------- |
| [ResNet-50](https://drive.google.com/open?id=1n7H6WNrvtf0eyWeWotbWD1kb95iVWaze)                | step (90 epochs)    | 25.557M    | 4.089  | 76.010 / 92.834         |
| [ResNet-50](https://drive.google.com/open?id=1_aconGn2oZB1Bvgq65g2tsqSI7CSPAEt)                | cosine (120 epochs) | 25.557M    | 4.089  | 77.150 / 93.468         |
| [OctResNet-50 (alpha=0.5)](https://drive.google.com/open?id=1F9esqmbIJmfTOsAZ6_6JEUnI83LVgF_S) | cosine (120 epochs) | 25.557M    | 2.367  | 77.640 / 93.662         |



## To be Done
- [ ] Support for MobileNet family (pending for architectural details from the author)

## Acknowledgement
[Official MXNet implmentation](https://github.com/facebookresearch/OctConv) by [@cypw](https://github.com/cypw)

## Citation
```bibtex
@article{chen2019drop,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  journal={arXiv preprint arXiv:1904.05049},
  year={2019}
}
```
