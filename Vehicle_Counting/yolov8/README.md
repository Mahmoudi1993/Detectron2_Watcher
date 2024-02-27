# Estimate vehicle Speed with YOLOv8
We are going to deveolpe a little program which detects the estimate speed of vehicles which drive through a wide street.
To make it simple, we are only going to track the vehicles going away (i.e appearing from the topdown of the screen).
To do so, we are going to use the library YOLOv8, making use of the object box detection.

Running the Estimate vehicle Speed with YOLOv8:

```
python estimate_vehicle_speed.py --video /content/People-Counting-in-Real-Time-with-Detectron2/example/Car_test_0S.mp4 --output /content/People-Counting-in-Real-Time-with-Detectron2/results/Car_test_0S_result.mp4 

```
# Demo of Estimate vehicle Speed


Command Line Args Reference:

```
Command :
--video: path to input video (use 0 for webcam)
  (default: None)

--video:
  (default: None)

--output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
  (default: None)

--class_id:
  (default: 2)

--model:
  (default: 
   '/content/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

--output_format: codec used in VideoWriter when saving video to file
  (default: 'XVID')

--polygon: The lines sepearating the lanes are start, 
                from left to right at [50,235,410,550]
  (default: '50,235,410,550')

--score: confidence threshold
  (default: 0)

--confidence_threshold: confidence_threshold
     (default: 0.3)

--model_resolution: model_resolution
     (default: 1280)

--iou_threshold: iou_threshold
     (default: 0.5)
```

Reference:

https://www.youtube.com/watch?v=uWP6UjDeZvY&feature=youtu.be

https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation
