import cv2 as cv 
import numpy as np
class Plotter():
    '''A classs to actually plot the detected curves on the frame 
    of the video 
    PARAMETERS:  color - (3-tuple) specifying the values of the (R,G,B) channels
                 step - (int) It defines the pixel step required in plotting the 
                        curve. Specifies how detailed potting is required.
                        Smaller step means smoother curve 
                 thickness - (int) how thick you want the line to be 
                 '''
    def __init__(self,color = (200,255,100),
                step = 5,
                thickness = 3):
        self.color = color
        
        if(type(thickness) is not int or thickness < 1):
            raise Exception("Invalid line thickness given for plotting")
        self.thickness = thickness
        
        if(type(step) is not int or step < 1):
            raise Exception("Invalid step size given for plotting")
        self.step = step 
        
    def get_y(self,x,parabola):
        '''Returns the value of ax^2 + bx + c'''
        A = parabola[0]
        B = parabola[1]
        C = parabola[2]
        return int(A*(x**2) + B*x + C)
    
    def plot_buffer(self,frame,limits,orientation):
        '''Plot the buffer lane on the curve if no curve has 
           been detected.
           RETURNS: frame with the buffer lane drawn '''
        height = frame.shape[0]
        end_height = int(0.5 * height) # end point of buffer
        if orientation == 'L':
            # draw left buffer 
            # left limit starts from far left point
            left = limits[0]
            right = limits[1]
            #define points
            p1 = (left,height-1)
            p2 = (right,end_height)
            cv.line(frame,p1,p2,self.color,self.thickness)
            return frame
        else:
            # draw right buffer 
            # right limit starts from far right point
            right = limits[0]
            left = limits[1]
            #define points 
            
            p1 = (right,height-1)
            p2 = (left,end_height)
            cv.line(frame,p1,p2,self.color,self.thickness)
            return frame 
            
    def plot_curve_left(self,frame,limits,parabola):
        '''Method to plot the left parabola on the given 
           frame. If no parabola detected, buffer lane is 
           plotted on the image '''
        height = frame.shape[0]
        if parabola is None:
            plotted = self.plot_buffer(frame,limits,'L')
            return plotted
        else:
            x_start = limits[0] # left end
            x_end = limits[1] # right end 
            
            # get the first point on your image 
            left = x_start
            right = left + self.step 
            while right < x_end :
                # now generate co-ordinates of the points 
                # according to the parabola 
                y1 = self.get_y(left,parabola)
                y2 = self.get_y(right,parabola)
                
                # need to check if coordinates are actually inside the image 
                if y1 >= height or y2>= height:
                    left = right 
                    right = left + self.step 
                    continue
                # if okay 
                p1 = (left,y1)
                p2 = (right,y2)
                
                #plot this line on the frame
                cv.line(frame,p1,p2,self.color,self.thickness)
                left = right 
                right = left + self.step 
        
            return frame
        
    def plot_curve_right(self,frame,limits,parabola):
        '''Method to plot the right parabola on the given 
           frame. If no parabola detected, buffer lane is 
           plotted on the image '''
        height = frame.shape[0]
        if parabola is None:
            plotted = self.plot_buffer(frame,limits,'R')
            return plotted
        else:
            x_start = limits[0] # right end
            x_end = limits[1] # left end
            
            # get the first point on your image 
            right = x_start
            left = right - self.step 
            while left > x_end :
                # now generate co-ordinates of the points 
                # according to the parabola 
                y1 = self.get_y(left,parabola)
                y2 = self.get_y(right,parabola)
                # need to check that y co-ordinates lie in the image first 
                if y1 >= height or y2>= height:
                    right = left 
                    left = right - self.step 
                    continue 
                # if okay, 
                p1 = (left,y1)
                p2 = (right,y2)
                
                #plot this line on the frame
                cv.line(frame,p1,p2,self.color,self.thickness)
                right = left 
                left = right - self.step
                
            return frame
        
        