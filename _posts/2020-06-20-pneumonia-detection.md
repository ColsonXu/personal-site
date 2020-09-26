---
title: 'Pneumonia Detection from Chest X-Ray'
date: 2020-06-20 00:00:00
description: This blog showcases my pneumonia detection neural network project and the thought process behind it.
---

__Note: You can find the jupyter notebook [here](https://jovian.ml/colsonxu/pneumonia-detection). I suggest opening the notebook so you can follow along.__

## PURPOSE

In the purpose of improving my understanding of convolutional neural network, I decided to do a project that detects pneumonia from chest x-ray images. Additionally, I want my model to be able to distinguish between bacteria and virus infections, which has rarely been done before. The reason for deploying a neural network model to detect pneumonia is that traditionally the diagnosis of disease has always been a challenge in the process of treatment. The X-Ray images differ in quality. Sometimes it is hard for human to detect minor abnormalities thus introducing a lot of human error. A neural network, on the other hand, can be trained to recognize patterns quickly and accurately. Being able to quickly identify pneumonia with confidence is crucial for early treatment of patients.

The data I am using can be found on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## FIRST MODEL
I have devided to use a custom CNN (Convolutional Neural Network) from a class I took which was used on the CIFAR-10 dataset. The model's architecture is shown below.

```python
nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Flatten(), 
      nn.Linear(4096, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 10)
)
```

