# Detector
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# import some common libraries
import numpy as np
import sys
import cv2
import random
import pandas as pd
import supervision as sv
from absl import app, flags, logging
from absl.flags import FLAGS
from Detection import predictor
#from IPython import display

FLAGS = flags.FLAGS

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('video', 'E:/Didban/Detectron2/CompleteDetectron2/videos/mall.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_integer('class_id', 0, 'class_id number to')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_integer('polygon',[1632, 1654], [2920, 1622], 'polygon threshold')

def main(argv):
    # input_size = FLAGS.size
    source_path = FLAGS.video
    target_path = FLAGS.output
    class_id = FLAGS.class_id
    input_size = FLAGS.size
    model = FLAGS.model
    
    # initiate polygon zone Grand # ************* flags
    polygon = np.array([
             [1632, 1654],
             [2920, 1622]
    ])


    video_info = sv.VideoInfo.from_video_path(source_path)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    # initiate annotators
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)


    emp1 = predictor(model)
    Predictor = (emp1.get_frame()) 

    def process_frame(frame: np.ndarray, i: int) -> np.ndarray:
        print('frame', i)
        # detect
        
        outputs = Predictor(frame)
        detections = sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int)
        )
        detections = detections[detections.class_id == class_id]
        zone.trigger(detections=detections)

        # annotate
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        frame = box_annotator.annotate(scene=frame, detections=detections, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)

        return  frame

    ESCALATOR_VIDEO_PATH =  r'E:/Didban/Detectron2/CompleteDetectron2/videos/mall.mp4' 

    sv.process_video(source_path=ESCALATOR_VIDEO_PATH, target_path = target_path, callback=process_frame)

    
    # display.clear_output()
    
    return 

if __name__ == '__main__':
    app.run(main)
        