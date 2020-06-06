# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:53:38 2020

@author: skesh
"""
from math import pow, sqrt
import numpy as np
import cv2


def Input_pre_processing(object):
    
    labels = [line.strip() for line in open(object)]
    bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))
    
    return bounding_box_color,labels

def load_model(prototext,model):
    
    network = cv2.dnn.readNetFromCaffe(prototext, model)
    
    return network

def standardizing_image(frame):
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    return blob


def get_box(x_start, y_start, x_end, y_end,Focal_Length):

    # Mid point of bounding box
    x_mid = round((x_start+x_end)/2,4)
    y_mid = round((y_start+y_end)/2,4)

    height = round(y_end-y_start,4)

    # Distance from camera based on triangle similarity
    distance = (165 * Focal_Length)/height


    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
    x_mid_cm = (x_mid * distance) / Focal_Length
    y_mid_cm = (y_mid * distance) / Focal_Length
    return (x_mid_cm,y_mid_cm,distance) 
    
def close_object(pos_dict,threshold):
    
    violation_object = set()
    for i in pos_dict.keys(): 
        for j in pos_dict.keys():
            if i < j:
                
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))
                                         
                # Check if distance less than 2 metres or 200 centimetres
                if dist < threshold:
                    violation_object.add(i)
                    violation_object.add(j)
    
    return violation_object

def display(frame, Time, violations_sec, cummulative_violations):
    
    text = " Time : {} sec, Violation  : {}".format(Time,violations_sec)
    cv2.rectangle(frame, (0,0), (360,30), (30,20,30), thickness= cv2.FILLED)
    cv2.putText(frame, text, (2, 10),
    cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
            
    text_violation = "Time : {} sec, Cummulative Violations: {}".format(Time,cummulative_violations)
    cv2.putText(frame, text_violation, (9, 24),
    cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 255, 0), 1)
         

