#!/usr/bin/env python
# coding: utf-8

# ### Objective 
# - I have a segmented binary image 
# - Traverse the image patch by patch and get the percentage of points in the patch
# - If the patch is containing a lot of points,  it is probable that the non zero points are corresponding to a lane 
# - If not , simply discard those points and move ahead 
# - Loop till you cover the whole segment and then return the list of all the non zero points that you classify as <b>lanes</b>

# In[5]:


import cv2 as cv
import numpy as np


# def get_lane_points(frame,left,right):
#     '''PARAMETERS: frame : segmented binary image 
#                    left : left lane of the trapezium segment 
#                    right : right lane of the trapezium segment 
                   
#     RETURNS : points : a list of 2 tuples of proposed lane coords'''
#     points = []
#     left_x = left[0][0]
#     left_y = left[0][1]
#     width,height = frame.shape[1],frame.shape[0]
#     height-=1 # corner case 
#     # side of the square
#     side = 50
#     right_x = right[0][0] - side
#     # generate square : bottom left,bottom right, top right, top left 
#     square = [(left_x,height),(left_x+side,height),(left_x+side,height - side),(left_x,height - side)]
#     # start at the bottom 
#     # while the base of the square is inside the trapezium : assuming trapezium top at half of the height
#     while (square[0][1] > int(0.3*height)):
#         # while top left corner under the right lane, continue
#         while(square[2][0] < right_x):
#             # traverse left to right 
#             left_end= square[0][0]
#             right_end = left_end + side
#             bottom_end = square[0][1]
#             upper_end = bottom_end - side 
# #             print("Bottom :",bottom_end,"Top :",upper_end)
#             # start moving bottom to top 
#             sq_points = []
#             count = 0
#             for i in range(bottom_end,upper_end,-1):
#                 # start moving left to right
#                 for j in range(left_end,right_end):
#                     if(frame[i][j] > 0):
#                         count+=1
#                         sq_points.append((i,j))
#             # if the percentage of points is greater than 15% , keep else reject 
#             perc = count/25 # x% of 2500 
# #             print(count)
#             if(perc < 10):
#                 points.append(sq_points)
#             # update the square 
#             x,y = square[1][0],square[1][1]
#             square = [(x,y),(x+side,y),(x+side,y-side),(x,y-side)]
            
#         # update the base of square
#         left_x += side # shift right 
#         left_y -= side # shift up 
#         right_x-= side # shift the right point left
#         square = [(left_x,left_y),(left_x+side,left_y),(left_x+side,left_y-side),(left_x,left_y-side)]
        
#     return points

def get_lane_points(frame, left, right, pxx=10, pxy=30):
    '''
    input: frame, left -> points of left boundary, right -> points of right boundary
    pxx -> pixel size of x 
    pxy -> pixel size in y
    points : a list of 2 tuples of proposed lane coords'''
    x_start = left[0][0]
    x_end = right[0][0]
    y_start = left[1][1]
    y_end = left[0][1]
    x = np.array([], dtype=np.uint32)
    y = np.array([], dtype=np.uint32)
    for i in range(y_start, y_end, pxy):
        for j in range(x_start, x_end, pxx):
            if((pxx*pxy)/40 < np.count_nonzero(frame[i:i+pxx, j:j+pxy]) < (pxx*pxy)/15):
                nz = np.nonzero(frame[i:i+pxx, j:j+pxy])
                x = np.hstack((x, nz[0]+i))
                y = np.hstack((y, nz[1]+j))

    img2 = np.zeros_like(frame)
    for i in range(len(x)):
        img2[x[i], y[i]] = 255
    return np.transpose((x, y))


# In[7]:


def do_canny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7,7), 0)
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv.Canny(blur,lower,upper)
    return canny


# In[8]:


def rescale_frame(frame,percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width,height)
    return cv.resize(frame,dim,interpolation = cv.INTER_AREA)


# In[9]:


def do_segment(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    poly = np.array([
        [(0,height),(width,height),(int(frame.shape[1]/2),int(frame.shape[0]/2))] # making a triangular mask for the segment.
    ])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask,poly,255) # filling the frame's pentagon with ones
    
    segment = cv.bitwise_and(frame,mask)
    
    return segment


# In[20]:


cap = cv.VideoCapture(r"E:\InnerveHackathon\pathvalild_Trim.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = rescale_frame(frame,percent = 57)
    except:
        break
    canny = do_canny(frame)
    im = do_segment(canny)
    width,height = im.shape[1],im.shape[0]
    left = ((0,height),(width//2,int(height*0.3)))
    right = ((width,height),(width//2,int(0.3*height)))
    points = get_lane_points(im,left,right)
    cv.imshow("Orig",im)
    f = np.zeros_like(im) 
    for k in points:
        for i in k:
            x,y = i[0],i[1]
            f[x][y] = 180
    cv.imshow("De noised",f)
    if cv.waitKey(13) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# In[ ]:




