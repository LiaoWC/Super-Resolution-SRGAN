# Super-Resolution-SRGAN

## Prepare

### Data

Download data here: https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view

- Put training images into './datasets/training_hr_images/training_hr_images/'
- Put testing images into './datasets/testing_hr_images/testing_hr_images/'

### Model

Pretrained model: https://drive.google.com/file/d/1xMk8OI1eaHipv0IRq7Qc4QIyS2pWnOdz/view?usp=sharing  
Put it into './' => './generator_1052.pth'

## Train

```shell
python3 srgan.py
# P.S. add "-h" to get more infomation
```

## Test

```shell
python3 inference.py
```

The output will be in './results'.

## Reference

- SRGAN paper: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
  Network (https://arxiv.org/abs/1609.04802v5)
- Code is modified from: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan