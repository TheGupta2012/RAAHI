import cv2 as cv 
import numpy as np
class Curve():
    '''PARAMETERS: 
                     window_size -> float (0,1): how wide you want the 
                                    window to be 
      METHODS: // to-do
    '''
    def __init__(self,
                window_size = 0.2):
        
        self.left_coords = []
        self.right_coords = []
        if(window_size >=1 or window_size<=0):
            raise Exception("Invalid window size given") 
        self.window_size = window_size
        
        
    def find_non_zero(self,frame):
        '''
        Finds all the non zero points inside the Trapezium boundary
        Faster than cv.FindNonZero...
        PARAMETERS: frame : binary image for which we want non zero x,y coords
        RETURNS: a list of 2-tuples of the non zero points , sorted by x 
        '''
        # this code is written manually to speed up computational cost
        half = int(frame.shape[0]/2) # height / 2
        row = frame.shape[0] - 2
        left, right = 0,frame.shape[1] 
        width = frame.shape[1]
        points =[]
        while row > half and left + int(0.08*width) < right - int(0.08*width):
            for i in range(left,right):
                if(frame[row][i] != 0):
                    points.append((i,row))
            left+=3
            right-=3
            row-=4
        return sorted(points)
    
    def isInside(self,point,left_lane,right_lane):
        '''Function to check whether the point given 
        lies inside the parallelogram or not 
        Cross product of all pairwise triangles must add 
        up to the area of parallelogram. 
        PARAMETERS: point ,> a 2,tuple which contains the 
                            co,ordinates of the point 
                    left_lane ,>2,tuple which consists of  the 
                            left side of the parallelogram
                    right_lane ,>2 ,tuple which consists of the 
                            right side of the parallelogram 
        RETURNS : Boolean specifying whether the point lies 
                  inside or not '''
        bottom_left = left_lane[0]
        bottom_right = right_lane[0]
        top_left = left_lane[1]
        top_right = right_lane[1]
        
        # start from top 
        A1B = np.subtract(top_left,point)
        A1C = np.subtract(top_right,point )
        a1 = 0.5 * abs(np.cross(A1B,A1C))
        
        # right 
        A2B = np.subtract(top_right,point )
        A2C = np.subtract(bottom_right,point)
        a2 = 0.5 * abs(np.cross(A2B,A2C))
        
        #bottom 
        A3B = np.subtract(bottom_right,point) 
        A3C = np.subtract(bottom_left,point)
        a3 = 0.5 * abs(np.cross(A3B,A3C))
        
        #left 
        A4B = np.subtract(bottom_left,point) 
        A4C = np.subtract(top_left,point) 
        a4 = 0.5 * abs(np.cross(A4B,A4C))
        
        AB = np.subtract(top_left,bottom_left)
        AC = np.subtract(bottom_right,bottom_left) 
        A = abs(np.cross(AB,AC))
        
        if(a1+a2+a3+a4 == A):
            return True
        else:
            return False
        
    def get_left_curve(self,frame,left_lane):
        '''Method to calculate the left fit curve for the 
        given frame 
        PARAMETERS: frame : the image frame 
                    left_lane : the boundary left lane of segmented image
        RETURNS: None : if no curve detected 
                 3-tuple: if curve detected (A,B,C) : coefficients of 
                    x^2, x and the constant factor in Ax^2 + Bx + C'''
        if(left_lane is None):
            raise Exception ("No left lane given")
            return 
        
        width = frame.shape[1]
        shift = self.window_size * width
        # start drawing windows and fitting curves 
        xy = self.find_non_zero(np.array(frame))
        left = left_lane  #-> like this - '/'
        # get the right lane -> '/'
        right = ((left[0][0] + shift,left[0][1]),(left[1][0]+shift,left[1][1]))
        
        while right[0][0] < int(0.5 * width):
            # get all the points that are non zero inside the parallelogram 
            # and get there x-y coords 
            X, Y =[],[]
            for k in xy:
                if(self.isInside(k,left,right) == True): 
                    # to - do
                    X.append(k[0])
                    Y.append(k[1])
            '''Only calculate the parabola if you actually 
            have more than 100 points to fit the curve to.
            This is because the points may just be noise if they are 
            very less'''
            if(len(X) <= 40):
                # shift window
                left = right 
                right = ((left[0][0] + shift,left[0][1]),(left[1][0]+shift,left[1][1]))
                continue 
            # polyfit returns a 3 tuple with the 
            # x^2 coefficient as 1st element, x as 2nd and constant as 3rd
            parabola = np.polyfit(X,Y,2)
            # save polynomial coords 
            left = right 
            right = ((left[0][0] + shift,left[0][1]),(left[1][0]+shift,left[1][1]))
            
            self.left_coords.append(parabola)
            
        A,B,C = [],[],[]
        for k in self.left_coords:
            A.append(k[0])
            B.append(k[1])
            C.append(k[2])
        #average out 
        try:
            A = sum(A)/len(A)
            B = sum(B)/len(B)
            C = sum(C)/len(C)
        except:
            return None
        return (A,B,C)
    
    def get_right_curve(self,frame,right_lane):
        '''Method to calculate the left fit curve for the 
        given frame 
        PARAMETERS: frame : the image frame 
                    left_lane : the boundary left lane of segmented image
        RETURNS: None : if no curve detected 
                 3-tuple: if curve detected (A,B,C) - coefficients of 
                    x^2, x and the constant factor in Ax^2 + Bx + C'''
        if(right_lane is None):
            raise Exception ("No right lane given")
            return 
        width = frame.shape[1]
        shift = self.window_size * width
        # get all non-zero x,y coordinates sorted by x 
        xy = self.find_non_zero(np.array(frame))

        # start drawing windows and fitting curves 
        right = right_lane  #-> like this - '\'
        # get the left lane - '\'
        left = ((right[0][0] - shift,right[0][1]),(right[1][0]-shift,right[1][1]))
        while left[0][0] > int(0.5 * width):
            
            # get all the non -zero points that are inside the parallelogram 
            # and get there x-y coords 
            X, Y =[],[]
            for k in xy:
                if(self.isInside(k,left,right) == True): 
                    # to - do
                    X.append(k[0])
                    Y.append(k[1])
            '''Only calculate the parabola if you actually 
            have like more than 50 points to fit the curve to.
            This is because the points may just be noise if they are 
            very less'''
            if(len(X) <= 40):
                # shift window 
                right = left 
                left = ((right[0][0] - shift,right[0][1]),(right[1][0]-shift,right[1][1]))
                continue 
                
            # polyfit returns a 3 tuple with the 
            # x^2 coefficient as 1st element, x as 2nd and constant as 3rd
            parabola = np.polyfit(X,Y,2)
            # save polynomial coords 
            self.right_coords.append(parabola)
            
            
        A,B,C = [],[],[]
        for k in self.right_coords:
            A.append(k[0])
            B.append(k[1])
            C.append(k[2])
        
        #average out 
        try:
            A = sum(A)/len(A)
            B = sum(B)/len(B)
            C = sum(C)/len(C)
        # tuple with right lane parameters
        except:
            return None    
        return (A,B,C)
        