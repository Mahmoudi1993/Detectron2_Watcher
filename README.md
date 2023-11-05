[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmoudi1993/Detectron2_Watcher/blob/main/Project_Real_Time_People_Counting_Using_Detectron2.ipynb)

# Detectron2_Watcher
Object tracking implemented with Detectron2, Deep SORT and PyTorch. Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. This platform is implemented in PyTorch. Thanks to its modular design its a very flexible and extensible framework providing fast training. We can take the output of Detectron2 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

# Demo of Object Detector on Cars



![ezgif com-video-to-gif](https://github.com/Mahmoudi1993/Detectron2_Watcher/assets/74957886/c19d39b0-4403-4f15-b654-1cada2f057fe)


# Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.
Installing dependencies and libraries 
We can use both CPU and GPU for training and inference of the models.

Running on CPU:

```
!pip install -q -U torch torchvision -f https://download.pytorch.org/whl/torch_stable.html 
!pip install -q -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
```

Running on GPU:

```

# pip install detectron2 on PC
# Create an environment Anaconda
# conda activate detectron_evn

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pycocotools
pip install -U torch torchvision cython(pip install cython)
conda install -c anaconda git
git clone https://github.com/facebookresearch/detectron2.git


# runing codes project real time people counting on Visual Studio Code

```

# Downloading Official  Detectron2 Model Zoo Pre-trained Weights
Our object tracker uses Detectron2 to make the object detections, which deep sort then uses to track.Many pretrained models can be found back within the "modelzoo". This is a collection of models pretrained on a certain dataset that are ready to be used. Mostly people will use the pretrained weights of these model for initalization of there own custom model. This significantly shortens the training time and performance.For easy demo purposes we will use the pre-trained weights for our tracker. Download pre-trained Detectron2.weights file: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# Running the Object Detector with Detectron2
we need to do is run the Detector.py script to run our object tracker with Detectron2, DeepSort and PyTorch.

```
# Run Detectron2 deep sort Detector on video
python Detector.py --video /content/Detectron2_Watcher/example/Car_test_1.mp4 --output /content/Detectron2_Watcher/Results/Car_test_1_result.mp4 --model /content/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
```

Calculate Coordinates for a Polygon Zone:

Before we can start counting objects in a zone, we need to first define the zone in which we want to count objects. 
We need the coordinates of the zone. We’ll use these later to know whether an object is inside or outside of the zone.


To calculate coordinates inside a zone, we can use PolygonZone, 
an interactive web application that lets you draw polygons on an image and export their coordinates for use with supervision.

PolgyonZone needs a frame from the video with which you will be working. We can extract a frame from our video using the following code:

```
# PolgyonZone needs a frame from the video with which you will be working.
# We can extract a frame from our video using the following code:

import supervision as sv
import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

generator = sv.get_video_frames_generator("/content/Detectron2_Watcher/example/Car_test_1.mp4")
iterator = iter(generator)
frame = next(iterator)

OUTPUT_VIDEO_PATH = r'/content/Detectron2_Watcher/Results'
os.chdir(OUTPUT_VIDEO_PATH)

cv2.imwrite("frame.jpg", frame)
```

This code will retrieve the first frame from our video and save the frame as a file on our local machine.

We can now use this image to calculate the coordinates of the zone we want to draw on our image. First, open up PolygonZone and upload the frame:
https://roboflow.github.io/polygonzone/

```
# Run Detectron2 deep sort Detector on video with flag polygon
# --polygon
python Detector.py --video /content/Detectron2_Watcher/example/Car_test_1.mp4 --output /content/Detectron2_Watcher/Results/Car_test_1_result.mp4 --model /content/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --polygon [788, 286],[140, 314],[376, 170],[496, 166],[788, 282]
```

set video flag to:

0 = ( Webcam in laptop )

1 = ( The webcam is connected to the laptop through the USB port )

link = ( Camera )

If you don't have a high-quality camera, we can use  yourself phone's camera through the IP Webcam app.

About this app:
IP Webcam turns your phone into a network camera with multiple viewing options. 
View your camera on any platform with VLC player or web browser. Stream video inside WiFi network without internet access.

```
# Run Detectron2 deep sort Detector on webcam
python Detector.py --video 0 --output /content/Detectron2_Watcher/Results/Car_test_1_result.mp4 --model /content/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

```
