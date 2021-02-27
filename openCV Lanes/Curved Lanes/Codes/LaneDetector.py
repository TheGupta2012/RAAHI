#!/usr/bin/env python
# coding: utf-8

# ## Making Lane Detector

# In[1]:


import cv2 as cv
import numpy as np
from ImagePrep import *
from CurveDetector import * 
from Plotter import *


# In[2]:


def rescale_frame(frame,percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width,height)
        return cv.resize(frame,dim,interpolation = cv.INTER_AREA)


# In[3]:


prep = PrepareImage((11,11),(3,3),auto_canny=True, segment_y = 0.5)
CurveMake = Curve(0.12)
plotter = Plotter((200,255,100),10,3)


# In[ ]:


# working fine...
cap = cv.VideoCapture("E:\InnerveHackathon\pathvalild_Trim.mp4")
count = 0
while (cap.isOpened()):
    # ret = a boolean return value from getting the frame,
    # frame = the current frame being projected in video
    ret, frame = cap.read()
    try:
        frame = rescale_frame(frame,percent = 57)
    except:
        break
    width , height = frame.shape[1], frame.shape[0]
    cv.imshow("Original",frame)
    frame = prep.get_binary_image(frame)
    points = prep.get_poly_maskpoints(frame)
    cv.imshow("Canny",frame)

    left_lane = points[0] # two tuple 
    right_lane = points[1] # two tuple
    # pass these co-ordinates and the frame in a new class which 
    # actually gets the polynomial fitted
    left_coords = CurveMake.get_left_curve(frame,left_lane)
    right_coords = CurveMake.get_right_curve(frame,right_lane)
    
#     print("Left parabola :",left_coords)
#     print("Right parabola :",right_coords)
    # to incorporate the fact that the curve has not been detected
    # we return None in the left and right coords if nothing detected.

    #define limits -> only in this range I shall plot my curve
    limit_left_x = (int(0.15*width),int(0.40*width)) # left to right
    limit_right_x = (int(0.85*width),int(0.55*width)) # right to left
    new_frame = np.zeros_like(frame)
    # plot on image
    new_frame = plotter.plot_curve_left(new_frame,limit_left_x,left_coords)
    new_frame = plotter.plot_curve_right(new_frame,limit_right_x,right_coords)
    cv.line(new_frame,(int(0.5*width),height),(int(0.5*width),0),(0,255,0),4)
    #show image
    cv.imshow("Parabolas",new_frame)
# cv.line(images[1][0],(images[1][0].shape[1]//2,0),(images[1][0].shape[1]//2,images[1][0].shape[0]),(200,200,0),5)
    count+=1    
    if(count>1000):
        break
    if cv.waitKey(13) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# In[ ]:




