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


FLAGS = flags.FLAGS

flags.DEFINE_string('video', '/content/Detectron2_Watcher/example/Car_test_1.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '/content/Detectron2_Watcher/Results/Car_test_1_result.mp4', 'path to output video')
flags.DEFINE_integer('class_id', 2, 'class_id number to')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('polygon','[788, 286],[140, 314],[376, 170],[496, 166],[788, 282]' , 'polygon threshold')

def main(argv):
    source_path = FLAGS.video
    target_path = FLAGS.output
    class_id = FLAGS.class_id
    input_size = FLAGS.size
    model = FLAGS.model
    polygon = FLAGS.polygon
    
    # initiate polygon zone cars
    res = ast.literal_eval(polygon)
    res = list(res)
 
    polygon = np.array(res)


    video_info = sv.VideoInfo.from_video_path(source_path)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    # initiate annotators
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=2, text_thickness=2, text_scale=1)


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

   Cars_VIDEO_PATH =  r'/content/Detectron2_Watcher/example/Car_test_1.mp4'

    sv.process_video(source_path = cars_VIDEO_PATH, target_path = target_path, callback=process_frame)

    
    # display.clear_output()
    
    return 

if __name__ == '__main__':
    app.run(main)
        
