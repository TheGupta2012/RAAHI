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
        
# updated the points acquiring 
    def get_lane_points(self,frame,left,right):
        '''PARAMETERS: frame : segmented binary image 
                       left : left lane of the trapezium segment 
                       right : right lane of the trapezium segment 

        RETURNS : points : a list of 2 tuples of proposed lane coords'''
        points = []
        left_x = left[0][0]
        left_y = left[0][1]
        width,height = frame.shape[1],frame.shape[0]
        height-=1 # corner case 
        # side of the square
        side = 50
        right_x = right[0][0] - side
        # generate square : bottom left,bottom right, top right, top left 
        square = [(left_x,height),(left_x+side,height),(left_x+side,height - side),(left_x,height - side)]
        # start at the bottom 
        # while the base of the square is inside the trapezium : assuming trapezium top at half of the height
        while (square[0][1] > int(0.3*height)):
            # while top left corner under the right lane, continue
            while(square[2][0] < right_x):
                # traverse left to right 
                left_end= square[0][0]
                right_end = left_end + side
                bottom_end = square[0][1]
                upper_end = bottom_end - side 
                # start moving bottom to top 
                sq_points = []
                count = 0
                for i in range(bottom_end,upper_end,-1):
                    # start moving left to right
                    for j in range(left_end,right_end):
                        if(frame[i][j] > 0):
                            count+=1
                            sq_points.append((i,j))
                # if the percentage of points is greater than 15% , keep else reject 
                perc = count/25 # x% of 2500 
                if(perc < 10):
                    for k in sq_points:
                        points.append(k)
                # update the square 
                x,y = square[1][0],square[1][1]
                square = [(x,y),(x+side,y),(x+side,y-side),(x,y-side)]

            # update the base of square
            left_x += side # shift right 
            left_y -= side # shift up 
            right_x-= side # shift the right point left
            square = [(left_x,left_y),(left_x+side,left_y),(left_x+side,left_y-side),(left_x,left_y-side)]

        return points

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
        