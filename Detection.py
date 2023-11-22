
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import pandas as pd
import supervision as sv


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# which objects it can recognize:
# classes_id
def class_name_id():
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    predictor = DefaultPredictor(cfg)
    modelclasses = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    df = pd.DataFrame(modelclasses,columns=['Model classes'])
    return print(df)

# the result of the pretrained model from the modelzoo. 
# For this we'll use the Visualizer class.
class predictor():
    def __init__(self, model, score): #frame):
        # self.frame = frame
        self.model = model
        self.score = score
    def get_frame(self):
        
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file(self.model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        predictor = DefaultPredictor(cfg)
        return predictor

# Object Detection with Detectron2
class detection(object):
    def predictor(self, model, input_size, video_path, class_id):
        
        self.model = model
        self.size = input_size
        self.video = video_path
        self.class_id = class_id
        
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file(self.model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        predictor = DefaultPredictor(cfg)
        
        # webcam laptop(# 0 & # 1)  
        # link("https://192.168.1.2:8080/vidoe")
        # vidoe 'C:\Users\ASUA\Desktop\Newfolder\Final_Test\' 

        # extract video frame
        
        generator = sv.get_video_frames_generator(video_path)  

        #generator = cv2.VideoCapture("http://192.168.1.2:8080/video")
        iterator = iter(generator)
        frame = next(iterator)

        # detect
        outputs = predictor(frame)
        detections = sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int)
        )

        detections = detections[detections.class_id == self.class_id]
        
        # annotate ( box, % detect)
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        frame = box_annotator.annotate(scene=frame, detections=detections)

        return  predictor 
    
if __name__ == '__main__':
    detection().predictor('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 416, 'E:/Didban/Detectron2/CompleteDetectron2/videos/mall.mp4', 0)

        
