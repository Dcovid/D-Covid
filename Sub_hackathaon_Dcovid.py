# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:35:44 2020

@author: aaddewala
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:28:20 2020

@author: aaddewala
"""
import numpy as np
import cv2
from Helper_Functions import Input_pre_processing, close_object, display, get_box, load_model,standardizing_image


pramaters = {'video': 'Input_Video.mp4', 
 'model': 'Social_Distancing.caffemodel',
 'prototxt': 'Prototext.txt', 
 'labels': 'object.txt', 
 'confidence': 0.2}


cummulative_violations = 0   # cummulative violation for a given time window. For our example it si 1 sec
frame_no = 0                 # frame index
violations = 0               # cummulative violations over sequece of frames
violations_sec = 0           # violationa at a given instant. For our example it is 1 sec
Time = 0                     # Time elapsed  
Focal_Length = 600           # Focal Length of the Camera used
threshold = 200              # Minimum distance in cm to be maintained for Scoial Distancing


# Detecing a person and drawing a bounding box 
box_color,labels = Input_pre_processing(pramaters['labels'])

# Load model
model = load_model(pramaters["prototxt"], pramaters["model"])

img_array = []
# Accessing the video 
if pramaters['video']:
    cap = cv2.VideoCapture(pramaters['video'])
else:
    cap = cv2.VideoCapture(0)


while cap.isOpened():

    frame_no = frame_no+1

    # Capture each frame in a sequence
    ret, frame = cap.read()

    if not ret:
        break

    (height, width) = frame.shape[:2]

    # standardizing the frame to 300X300 and performing a forward pass
    blob = standardizing_image(frame)
    model.setInput(blob)
    detections = model.forward()
    
    # Dictionaries used to store the positions
    pos_dict = dict()
    coordinates = dict()

    for i in range(detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]

        if confidence > pramaters["confidence"]:

            index = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x_start, y_start, x_end, y_end) = box.astype('int')

            # Filtering only persons detected in the frame. Class Id of 'person' is 15
            if index == 15.00:

                # Draw bounding box for the object
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), box_color[index], 2)
                label = "{}: {:.2f}%".format(labels[index], confidence * 100)
                coordinates[i] = (x_start, y_start, x_end, y_end)
                pos_dict[i] = get_box(x_start, y_start, x_end, y_end,Focal_Length)

    # Distance between every object detected in a frame
    violation_object = close_object(pos_dict,threshold)
    
    for i in pos_dict.keys():
        
        if i in violation_object:
            COLOR = (0,0,255)
            violations = violations + 1
        else:
            COLOR = (0,255,0)
            
        (x_start, y_start, x_end, y_end) = coordinates[i]

        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), COLOR, 2)
        y = y_start - 15 if y_start - 15 > 15 else y_start + 15
        

    if frame_no%30==0:
        
        violations_sec = round(violations/(frame_no))
        cummulative_violations = cummulative_violations + violations_sec
        Time = frame_no/30
        print(cummulative_violations,'----------',violations_sec,'------------',Time,'-----------',frame_no,'---',i)

    
    display(frame, Time, violations_sec, cummulative_violations)
    
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)
    
    height1, width1, layers1 = frame.shape
    size = (width1,height1)
    img_array.append(frame)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
out = cv2.VideoWriter('Output Prediction.avi',cv2.VideoWriter_fourcc(*'DIVX'), 29, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()        
cap.release()        

cap.release()
cv2.destroyAllWindows()

