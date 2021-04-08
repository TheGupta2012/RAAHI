import cv2 as cv 
import numpy as np
class Curve():
    '''PARAMETERS: 
                     window_size -> float (0,1): how wide you want the 
                                    window to be 
      METHODS: // to-do
    '''
    def __init__(self,draw = False):
        self.non_zero = []
        self.prev_left = None
        self.prev_right = None
        self.draw = draw
    def get_lane_points(self,frame, left, right, pxx=10, pxy=30):
        '''
        PARAMETERS: frame, left -> points of left boundary, right -> points of right boundary
                    pxx -> pixel size of x 
                    pxy -> pixel size in y
        RETURNS: points : a list of 2 tuples of proposed lane coords'''
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
        
        return np.transpose((x, y))


    def detect_curve(self,img, x, y, left, right):
        ''' PARAMETRS: Frame, x-> X coordinates of white points , y-> Y coordinates of white points
                       left -> Points of left boundary
                       right -> Points of right boundary

            RETURNS: -> Image with the single curve traced
        '''
        img2 = np.zeros_like(img)
        
#         y = -y
        a, b, c = np.polyfit(x, y, 2)
        x_start = left[0][0]
        x_end = right[0][0]
        y_start = left[1][1]
        y_end = left[0][1]
        for i in range(min(x), max(x)):
            y_ = int(a*i*i+b*i+c)
            try:
                if(y_ < img2.shape[0] and y_>0):
                    img2[i, y_] = 255
            except:
                pass
        return img2


    def curveTrace(self,frame, left, right):
        '''
        PARAMETERS:  frame,left - coordinates of left boundary, right - coordinates of right boundary
        '''
        height, width = frame.shape
        self.non_zero = []
        # splitting the image to two parts
        left_img = frame[:, :width//2]
        right_img = frame[:, width//2+1:]


        # Working on the left curve 
        try:   
            curr_points = self.get_lane_points(left_img, left, right, 10, 30)
         # what if very less points?
            if(self.prev_left is None):
                self.prev_left = curr_points
                self.non_zero.append(curr_points)
                x,y = np.transpose(curr_points)
            else:
                if(len(curr_points) < int(0.6*len(self.prev_left)) or curr_points is None):
                    x,y = np.transpose(self.prev_left)
                    self.non_zero.append(self.prev_left)
                else:
                    x,y = np.transpose(curr_points)
                    self.prev_left = curr_points
                    self.non_zero.append(curr_points)

            left_curve = self.detect_curve(left_img, x, y, left, right)
        except:
            left_curve = left_img
        
        # Working on the right curve
        try:
            flipped_right_img = cv.flip(right_img, 1)
            curr_points = self.get_lane_points(flipped_right_img, left, right, 10, 30)
        
        # what if very less points?    
            if(self.prev_right is None ):
                self.prev_right = curr_points
                x,y = np.transpose(curr_points)
                self.non_zero.append(curr_points)
            else:
                if(len(curr_points) < int(0.6*len(self.prev_right)) or curr_points is None): #30 % 
                    x,y = np.transpose(self.prev_right)
                    self.non_zero.append(self.prev_right)
                else:
                    self.prev_right = curr_points
                    x,y = np.transpose(curr_points)
                    self.non_zero.append(curr_points)
            
            right_curve = self.detect_curve(flipped_right_img, x, y, left, right)
            flipped_right_curve = cv.flip(right_curve,1)
        except:
            flipped_right_curve=right_img
        
        
        img2 = np.hstack((left_curve, flipped_right_curve))
        return img2

    def drawCurve(self,image, curve,color=(255,255,0),thickness=3):
        '''
        PARAMETERS:  image: Original image colored
                     curve -> Curve to draw on the image
                     color -> color of the curve
                     thickness -> Thickness of the curve '''
        height, width, col = image.shape
        if(self.draw == True):
            start = curve.shape[0]//3
        else:
            start = curve.shape[0]
        for i in range(start, curve.shape[0]):
            for j in range(curve.shape[1]):
                if(curve[i, j] != 0):
                    for x in range(thickness):
                        try:
                            image[i, j+x] = color
                        except:
                            pass
        return image


