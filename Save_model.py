# Detector
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# import some common libraries
import numpy as np
import ast
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

flags.DEFINE_string('DatasetLabels', 'E:/Didban/Detectron2/CompleteDetectron2/videos/mall.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_integer('class_id', 0, 'class_id number to')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('polygon','[788, 286],[140, 314],[376, 170],[496, 166],[788, 282]' , 'polygon threshold')

def main(argv):
    DatasetLabels= FLAGS.DatasetLabels
    class_id = FLAGS.class_id
    input_size = FLAGS.size
    model = FLAGS.model
    polygon = FLAGS.polygon
    score = FLAGS.score
    output = FLAGS.output
    
    # initiate polygon zone Grand # ************* flags
    res = ast.literal_eval(polygon)
    res = list(res)
 
    polygon = np.array(res)


    video_info = sv.VideoInfo.from_video_path(DatasetLabels)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    # initiate annotators
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=2, text_thickness=2, text_scale=1)
    

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))

    cfg.DATASETS.TRAIN = (DatasetLabels,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_id  # Number of output classes

    cfg.OUTPUT_DIR = output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025#0.00025  # Learning Rate
    cfg.SOLVER.MAX_ITER = 10000  # 20000 MAx Iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Batch Size

    trainer = DefaultTrainer(cfg) 

    def process_frame(frame: np.ndarray, i: int) -> np.ndarray:
        print('frame', i)
        # detect
        
        outputs = trainer(frame)
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
    
    trainer.resume_or_load(resume=False)
    # callback=process_frame
    trainer.train()
    # Save the model
    from detectron2.checkpoint import DetectionCheckpointer, Checkpointer
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    # Alternatively:
    # torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodel.pth"))
    checkpointer.save("mymodel_0") 
    
    return 

if __name__ == '__main__':
    app.run(main)
    

    