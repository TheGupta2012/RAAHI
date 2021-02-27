import cv2 as cv 
import numpy as np
class PrepareImage():
    '''ATTRIBUTES:
    gauss_size : kernel size for the gaussian blur 
                type-> tuple of size 2 with odd and equal 
                       entries > 1
    gauss_deviation : x and y axis standard deviations for 
               gaussian blur 
               type -> list-like of size = 2
    auto_canny : If auto canny is True use median of blurred 
                image to calculate thresholds 
                type-> boolean
    canny_low : the lower threshold of the canny filter 
                type -> int 
    canny_high : the higher threshold of the canny filter 
                type -> int 
    segment_x : the width of segment peak( the triangular 
               segment head). Given as the fraction of the width 
               of the image
               type -> float in (0,1) 0 and 1 exclusive
    segment_y : the height segment peak
                Given as the fraction of the height from the 
                top 
                type -> float in (0,1) 0 and 1 exclusive
                
    METHODS:
    do_canny : does gaussian blurring and canny thresholding of image 
    segment_image : segments the lane to reduce computation cost 
    get_poly_mask_points : returns the lane area to analyse for curve fitting
    get_binary_image: 
    '''
    def __init__(self,
                gauss_size = None,
                gauss_deviation = None,
                auto_canny = False,
                canny_low = 50,
                canny_high = 175,
                segment_x = 0.5,
                segment_y = 0.5):
        
        # setting gaussian kernel parameters.
        if(gauss_size is not None):
            if(len(gauss_size) != 2):
                raise Exception("Wrong size for the Gaussian Kernel")
            elif(type(gauss_size) is not tuple):
                raise Exception("Kernel type should be a tuple")
            elif(gauss_size[0]%2 == 0 or gauss_size[1]%2 == 0):
                raise Exception("Even entries found in Gaussian Kernel")    
        self.gauss_kernel = gauss_size
            
        if(gauss_deviation is not None):
            if(len(gauss_deviation)!=2):
                raise Exception("Wrong length of gauss deviation")
            else:
                self.gauss_deviation = gauss_deviation
            
        if(type(auto_canny) is not bool):
            raise TypeError("Incorrect Type mentioned for auto canny")
            
        # setting canny parameters
        if(auto_canny is False):
            self.auto_canny = False
            if(type(canny_high) is int and type(canny_low) is int):
                self.canny_low = canny_low 
                self.canny_high = canny_high 
            else:
                raise TypeError("Incorrect type specified for canny thresholds")
        else:
            self.auto_canny = True
            
        # setting segment parameters
        if segment_x >=1 or segment_x<=0:
            raise Exception("Fraction specified is out of range (0,1)")
        else:
            self.segment_x = segment_x
        if segment_y >=1 or segment_y<=0:
            raise Exception("Fraction specified is out of range (0,1)")
        else:
            self.segment_y = segment_y 
    def do_canny(self,frame):
        '''PARAMETERS: frame: the frame of the image on which we want to apply the 
                      canny filter 
          RETURNS : a canny filtered frame '''
        # gray the image 
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) 
        # apply blur 
        if(self.gauss_kernel is None):
            self.gauss_kernel = (9,9) # using a default kernel size 
        if(self.gauss_deviation is None):
            self.gauss_deviation = [3,3]
        
        blur = cv.GaussianBlur(gray, self.gauss_kernel, self.gauss_deviation[0], self.gauss_deviation[1])
        
        #apply canny filter 
        if self.auto_canny is False:
            canny = cv.Canny(blur,self.canny_low,self.canny_high)
        else:
            # Auto canny trumps specified parameters 
            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            canny = cv.Canny(blur,lower,upper)
        
        return canny 
    
    def segment_image(self,frame):
        '''PARAMETERS: frame : the frame of the image on which we want to apply the 
                      segementation filter 
        RETURNS : a segmented canny filtered frame '''
        height = frame.shape[0]
        width = frame.shape[1]
        shift = int(0.08 * width)
        points = np.array([
            [(0,height),(width,height),(int(width*self.segment_x)+shift,int(height*self.segment_y)),
             (int(width*self.segment_x)-shift,int(height*self.segment_y))]
        ])
        # create an image with zero intensity with same dimensions as frame.
        mask = np.zeros_like(frame)
        
        cv.fillPoly(mask,points,255) # filling the frame's triangle with white pixels
        # do a bitwise and on the canny filtered black and white image and the 
        # segment you just created to get a triangular area for lane detection 
        segment = cv.bitwise_and(frame, mask)
        
        # boundary lines...
        # cv.line(segment,(0,height),(int(width*self.segment_x)-shift,int(height*self.segment_y)),(250,0,0),1)
        # cv.line(segment,(width,height),(int(width*self.segment_x)+shift,int(height*self.segment_y)),(250,0,0),1)
        # cv.line(segment,(int(width*self.segment_x)+shift,int(height*self.segment_y)),
        #      (int(width*self.segment_x)-shift,int(height*self.segment_y)),(250,0,0),1)
        
        return segment 
    # this needs to be less tilted 
    def get_poly_maskpoints(self,frame):
        height = frame.shape[0]
        width = frame.shape[1]
        shift = int(0.08 * width)
        points = np.array([
                [(2*shift,height),(width-2*shift,height),(int(width*self.segment_x)+2*shift,int(height*self.segment_y)),
                 (int(width*self.segment_x)-2*shift,int(height*self.segment_y))]
            ])
        left = (points[0][0],points[0][3])
        right = (points[0][1],points[0][2])
        return (left,right)
    
    def get_binary_image(self,frame):
        can = self.do_canny(frame)
#         cv.imshow(can)
        seg = self.segment_image(can)
        return seg
    
