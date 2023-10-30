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
# Helper function, used these for debugging purposes
# detector2 build only succeeds if CUDA version is correct

!nvidia-smi
!nvcc --version

import torch
torch.__version__
import torchvision
torchvision.__version__

!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
```

# Downloading Official  Detectron2 Model Zoo Pre-trained Weights
Our object tracker uses Detectron2 to make the object detections, which deep sort then uses to track.Many pretrained models can be found back within the "modelzoo". This is a collection of models pretrained on a certain dataset that are ready to be used. Mostly people will use the pretrained weights of these model for initalization of there own custom model. This significantly shortens the training time and performance.For easy demo purposes we will use the pre-trained weights for our tracker. Download pre-trained Detectron2.weights file: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

```
!pip install
```
