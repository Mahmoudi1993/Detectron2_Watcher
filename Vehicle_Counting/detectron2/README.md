# vehicle speed detection with detectron2
We are going to deveolpe a little program which detects the number of vehicles which drive through a certain highway.
To make it simple, we are only going to track the vehicles going away (i.e appearing from the bottom of the screen).
To do so, we are going to use the library Detectron2, making use of the object box detection.

# Running the people counting with Detectron2:
'''
    python vehicle_counting.py --video /content/People-Counting-in-Real-Time-with-Detectron2/example/People_test_1.mp4 --output /content/People-Counting-in-Real-Time-with-Detectron2/results/People_test_1_rsult.mp4 

'''
