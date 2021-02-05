FireNet

FireNet is an artificial intelligence project for real-time fire detection.



FireNet is a real-time fire detection project containing an annotated dataset, pre-trained models and inference codes, all created to ensure that machine learning systems can be trained to detect fires instantly and eliminate false alerts. This is part of DeepQuest AI's to train machine learning systems to perceive, understand and act accordingly in solving problems in any environment they are deployed.

This is the first release of the FireNet. It contains an annotated dataset of 502 images splitted into 412 images for training and 90 images for validation.


>>> DOWNLOAD, TRAINING AND DETECTION:

The FireNet dataset is provided for download in the release section of this repository. You can download the dataset via the link below.

https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/fire-dataset.zip


We have also provided a ImageAI codebase to train a YOLOv3 detection model on the images and perform detection in mages and videos using a pre-trained model (also using YOLOv3) provided in the release section of this repository. The python codebase is contained in the fire_net.py file and the detection configuration JSON file for detection is also provided the detection_config.json. The pretrained YOLOv3 model is available for download via the link below.

https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/detection_model-ex-33--loss-4.97.h5


Running the experiment or detection requires that you have Tensorflow, and Keras, OpenCV and ImageAI installed. You can install this dependencies via the commands below.


- Tensorflow 1.4.0 (and later versions) Install or install via pip

 pip3 install --upgrade tensorflow 
- OpenCV Install or install via pip

 pip3 install opencv-python 
- Keras 2.x Install or install via pip

 pip3 install keras 
- ImageAI 2.0.3

pip3 install imageai --upgrade 



>>> Video & Prediction Results

Click below to watch the video demonstration of the trained model at work.
















References
Joseph Redmon and Ali Farhadi, YOLOv3: An Incremental Improvement
https://arxiv.org/abs/1804.02767
