# vehicle speed detection with detectron2
We are going to deveolpe a little program which detects the number of vehicles which drive through a certain highway.
To make it simple, we are only going to track the vehicles going away (i.e appearing from the bottom of the screen).
To do so, we are going to use the library Detectron2, making use of the object box detection.

Running the people counting with Detectron2:

```
python vehicle_counting.py --video /content/People-Counting-in-Real-Time-with-Detectron2/example/People_test_1.mp4 --output /content/People-Counting-in-Real-Time-with-Detectron2/results/People_test_1_rsult.mp4 

```

# Demo of Object Detector on Cars


![Uploading Cars_test_1_result_gif.gif…]()



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
  (default: 0.50)

```
Reference:

https://github.com/facebookresearch/detectron2

https://drive.google.com/file/d/1-95nd8kSvjsmQLD4UlHMKm4g2X4AElcG/view
________________________________________________________________________________________________________________________________

