---
title: 'Pneumonia Detection from Chest X-Ray'
date: 2020-06-20 00:00:00
description: This blog showcases my pneumonia detection neural network project and the thought process behind it.
featured_image: '/images/pneumonia-detection/featured.png'
tags: ["featured", "ML", "AI"]
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
## INPUT PRE-PROCESSING
Something I immediately noticed after I took a look at the data is that the dimension of them differ vastly. the smallest image is 460x157 while the largest image is 2916x2583. (As you can see, not only the resolution differ by miles, but the difference in aspect ratio is also night and day) This raises a great challenge because I don’t know what is the best way to normalize these input images. If I resize all the images to a certain size, then I am afraid of large images loosing details and small images becomes pixelated. Also, because of the differences in aspect ratio, if I normalize all the images to be squared, those images that have a high width to height ratio is going to loose a lot of information because half of the lungs is going to be cropped out, as shown below.
![img](/images/pneumonia-detection/normalization.png)
After many times of trail and error, I discovered that downsizing the images to around 256 pixels won’t hurt the detection accuracy much and it helped speed up the training. Thus, I decided on a final resolution of 256x300 this way the inputs are not too large and the rectangular dimension can account for the small fraction of the data that has a high width to height ratio.

After taking care of the input normalization I found out that the performance of this model was not bad, but not ideal. I want to test out different configurations to see what can I do to improve the performance of this model, both the training time and accuracy.

The first thing I noticed is that the input images are black & white (grayscale), however, they actually have three channels (RGB). So the first thing I want to do is to transform the pictures to grayscale. This was easy to do with a custom transform function that will be called when creating the ImageFolder dataset from raw input.

```python
normalize = tf.Compose([
    tf.Grayscale(),
    tf.Resize(256),
    tf.CenterCrop((256,300)),
    tf.ToTensor()
])
```

## VRAM USAGE
With the image flattened to just one channel, the amount of parameters the model has to tweak got decreased dramatically. However, I still notice that when training, the RAM or vRAM usage is still unreasonably high (Over 16GB while the entire dataset is just 1GB) and it usually crashes the notebook kernel. I couldn’t understand why for a long time and it took me days of research to came across the paper on AlexNet. This paper is so well written that everyone with elementary machine learning knowledge should be able to comprehend. In that paper, I learned that I don’t have to have so many convolutional layers and and output channels to make my model robust and accurate. The AlexNet model was designed to classify 1.2 million pictures of real-world objects into 1000 categories. Even with a problem of this complexity, the model only contains five convolutional layers and three FC (Fully-Connected) layers.

Taking a look at my model at the time, (I did some changes to the original model used in class) I have 6 convolutional layers and four FC layers. Those convolutional layers also have quite a bit more output channels than AlexNet. This all resulted in a extremely high number of parameters in the first FC layer (I once got to around a million during an experiment!) which, I learned from that paper, is the most space expensive. I then modified my network to mimic AlexNet, I didn’t want to use a pretrained model because I want to improve my understanding of the basics and it seems it has paid off well so far. I really learned a LOT from doing this project.

This is my architecture now.

```python
nn.Sequential(
      nn.Conv2d(1, 48, kernel_size=11, stride=4, padding=5),
      nn.ReLU(),
      nn.Conv2d(48, 128, kernel_size=5, stride=3, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(2, 2),
      
      nn.Flatten(),
      nn.Linear(3840, 2048),
      nn.ReLU(),
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Linear(1024, 3),
)
```

Now the first FC layer only takes in 3840 channels as opposed to a million before. Training time also drastically improved as I now have the luxury to set `num_workers` to 4 because of all the freed up memory space. (p.s. Later I upgraded my workstation's CPU to 12-core Ryzen 3900XT, I can set `num_workers to a much higher number and achieve a much shorter training time.`)

## BATCH SIZE
Another interesting thing I found out is that when I tried to increase the batch size (from 35 to 300) after I reduced the memory usage, the model’s accuracy stuck at around 49% for a long time before slowly increasing. The same thing happened again when I tried to decrease the batch size to just 1. I did not do too much experiment here but I seemed like 35 is a sweet spot for this model and dataset. The reason behind this behavior is still unknown.

## INPUT AUGMENTATION
Usually with this dataset, we can achieve an accuracy of over 95%. However, because I am making it distinguish between different kinds of pneumonia on top of simply identifying it, it makes the problem much more challenging for the model. With this model architecture, the highest accuracy I can achieve now is around 80%. My guess is that the model can identify pneumonia cases, but sometimes fails to name the cause. Also, I found out that despite of having an accuracy of 80%, when I apply the model on the test set, I can only achieve 70% accuracy. It means that the model is overfitting the training set, which can happen because the size of the dataset is still relatively small (4232 training images, 1000 validation images.) After some research, I learned that one way of dealing with this problem is to artificially increase the dataset by augmenting the images (Like changing the brightness, rotating the picture, adding noise to the picture, etc.) It is a bit of a challenge because rotating the image by 90, 180, or 270 degrees, which is the most common and the simplest method, does not make sense for this dataset. When applying this model to real world data, the X-Ray imagery is always going to be in its upright position. The same thing applies for flipping the images vertically; no patient is going to be up-side-down when they take an X-Ray. However, what I can do is to rotate the images just a little bit to simulate patients who are not aligning with the machine perfectly. I can also flip the images horizontally because of the symmetry of human lungs. I augmented the input data using the code below.

```python
augmentation = tf.Compose([
    tf.Grayscale(),
    tf.Resize(256),
    tf.CenterCrop((256,300)),
    tf.ColorJitter(brightness=0.2),
    tf.RandomRotation(degrees=15),
    tf.RandomHorizontalFlip(p=0.5),
    tf.ToTensor()
])

augmented_dataset = ImageFolder(data_dir+'/train', transform=augmentation)

img_index = 0
for _ in range(20):
    for img, label in augmented_dataset:
        lb = dataset.classes[label]
        save_image(img, lb+str(img_index)+'.jpeg')
        img_index += 1
```

After the augmentation, the new data set has 109,872 images in total. I am now able to achieve an accuracy of 94% while training. However, very interestingly, the accuracy on the test set actually decreased to 65%. It means my model actually overfitted more. I suspect the cause is how I augmented the data. For example, after the augmentation, there are black margins on images that have been rotated and the model will never see that on real-world X-Rays. This then became even more challenging as X-Ray images are usually already post-processed and any popular input augmentation like noise, rotation, flipping, cropping wouldn’t make sense here.
