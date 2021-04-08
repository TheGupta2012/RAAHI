#!/usr/bin/env python
# coding: utf-8

 ## Predictions Class



import cv2 as cv
import numpy as np 
from tts import speak
class Predictions():
    '''Provides predictions for a given binary frame where 
       the noise in the image has been removed.
       PARAMETERS: basis: string -> "mean" or "median" 
                           how do you provide the output 
                           for the lane that you acquired
                   threshold: float(0,1) : how closely you 
                           want the lane to be detected relative 
                           to center of image '''
    def __init__(self,basis = "mean",
                threshold = 0.1):
        
        if(basis not in ["mean","median"]):
            raise ValueError("Basis should be either mean or median")
        self.basis = basis
        
        if(threshold <=0 or threshold>=1 ):
            raise ValueError("Invalid range for threshold")
        self.threshold = threshold 
            
    def get_lane_middle(self,img,X):
        '''RETURNS: middle x co-ordinate based on the 
                    basis defined in class parameters '''
        if(self.basis == "mean"):
            try:
                mid = int(np.mean(X))
            except:
                mid = img.shape[1]//2
        else:
            try:
                mid = int(np.median(X))
            except:
                mid = img.shape[1]//2
        return mid
    
    def shifted_lane(self,frame,deviation):
        '''Generates outputs for where to shift 
        given the deviation of the lane center 
        with the image center orientation 
        
        RETURNS: frame with shift outputs '''
        height,width = frame.shape[0],frame.shape[1]
        shift_left = ["Lane present on left","Shift left"]
        shift_right = ["Lane present on right","Shift right"]
        if(deviation < 0):
            # means person on the right and lane on the left 
            # need to shift left 
            cv.putText(frame,shift_left[0],(40,40),5,1.1,(100,10,255),2)
            cv.putText(frame,shift_left[1],(40,70),5,1.1,(100,10,255),2)

#             speak(shift_left)
        else:
            # person needs to shift right 
            cv.putText(frame,shift_right[0],(40,40),5,1.1,(100,255,10),2)
            cv.putText(frame,shift_right[1],(40,70),5,1.1,(100,255,10),2)

#             speak(shift_right)
        return frame

    def get_outputs(self,frame,points):
        '''Generates predictions for walking 
           on a lane 
           PARAMETERS: frame : original frame on which we draw
                             predicted outputs. This already has the 
                             lanes drawn on it 
                       points : list of 2-tuples : the list 
                              which contains the points of the lane 
                              which is drawn on the image 
           RETURNS : a frame with the relevant outputs 
           '''
        
        height,width = frame.shape[0], frame.shape[1]
        # get the center of frame 
        center_x = width//2 
        # get the distribution of points on 
        # left and right of image center 
        left_x,right_x = 0,0
        X = []
        for i in points:
            for k in i:
                x = k[0]
                if(x < center_x):
                    left_x+=1
                else:
                    right_x+=1
                X.append(k[0])
        # get the lane middle and draw 
        try:
            lane_mid = self.get_lane_middle(frame,X)
        except:
            lane_mid = center_x
        cv.line(frame,(lane_mid,height-1),(lane_mid,height - width//10),(0,0,0),2)
        # calculate shift
        shift_allowed = int(self.threshold*width)
        # calculate deviations and put on image 
        deviation = lane_mid - center_x
        deviation_text = "Deviation: "+str(np.round((deviation * 100/width),3)) + "%"
        cv.putText(frame,deviation_text,(int(lane_mid-60),int(height-width//(9.5))),1,1.3,(250,20,250),2)
#         speak(deviation_text)
        
        if(abs(deviation) >= shift_allowed):
            # large deviation : give shift outputs only 
            frame = self.shifted_lane(frame,deviation)
            return frame 
        else:
            # if deviation lesser then that means either correct path 
            # or a turn is approaching : text put at the center of the 
            # frame 
            
            total_points= left_x + right_x 
            correct = ["Good Lane Maintainance"," Continue straight"]
            left_turn = ["Left turn is approaching","Please start turning left"]
            right_turn = ["Right turn is approaching","Please start turning right"]
            # if relative change in percentage of points is < 10% then 
            # going fine 
            try:
                left_perc = left_x*100/(total_points) 
                right_perc = right_x*100/(total_points) 
            except:
                left_perc = 50
                right_perc = 50
            if(abs(left_perc - right_perc) < 25):
                cv.putText(frame,correct[0],(40,40),5,1.1,(100,255,10),2)
                cv.putText(frame,correct[1],(40,70),5,1.1,(100,255,10),2)

#                 speak(correct)
            else:
                if(left_perc > right_perc): # more than 25% relative change 
                    # means a approximately a right turn is approaching 
                    cv.putText(frame,right_turn[0],(40,40),5,1.1,(100,10,255),2)
                    cv.putText(frame,right_turn[1],(40,70),5,1.1,(100,10,255),2)

#                     speak(right_turn)
                else:
                    cv.putText(frame,left_turn[0],(40,40),5,1.1,(100,10,255),2)
                    cv.putText(frame,left_turn[1],(40,70),5,1.1,(100,10,255),2)

#                     speak(left_turn)
            # return the frame with the outputs 
            # to-do : output with sound 
            return frame 





