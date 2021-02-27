import numpy as np
import cv2
import math
from ImagePrep import PrepareImage
# For preprocessing the image, returns the canny image, and original image
# def imagePreprocessing(image):
#     imggrey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     imgblur=cv2.GaussianBlur(imggrey,(7,7),0)
#     imgcany=cv2.Canny(imgblur,50,150)
#     return (imgcany,image)


# # Gets the required area on the basis of the points given
# def getRequiredImage(img,*pts):
#     imgblack=np.zeros_like(img)
#     mask=np.array([pts])
#     cv2.fillPoly(imgblack,mask,255)
#     img=cv2.bitwise_and(img,imgblack)
#     return img


def draw_curve(img,x,y,left,right):
    img2=np.zeros_like(img)
    y=-y
    a,b,c=np.polyfit(x,y,2)
    x_start=left[0][0]
    x_end=right[0][0]
    y_start=left[1][1]
    y_end=left[0][1]
    for i in range(min(x),max(x)):
        y_=int(-1*(a*i*i+b*i+c))
        try:
            if(y_<img2.shape[0]):
                img2[i,y_]=255
        except:
            pass
    return img2

def noiseReducer(img,left,right,pxx=10,pxy=30):
    # Setting the box size to be pxx pxy
    # White % .6667
    x_start=left[0][0]
    x_end=right[0][0]
    y_start=left[1][1]
    y_end=left[0][1]
    x=np.array([],dtype=np.uint32)
    y=np.array([],dtype=np.uint32)
    for i in range(y_start,y_end,pxy):
        for j in range(x_start,x_end,pxx):
            if((pxx*pxy)/40<np.count_nonzero(img[i:i+pxx,j:j+pxy])<(pxx*pxy)/15):
                nz=np.nonzero(img[i:i+pxx, j:j+pxy])
                x=np.hstack((x,nz[0]+i))
                y=np.hstack((y,nz[1]+j))

    return np.transpose((x,y))

def curve_ready(img,left,right,pxx,pxy,indicator):
    if(indicator==1):
        x,y=np.transpose(noiseReducer(img,left,right,10,30))

        curve=draw_curve(img,x,y,left,right)
        return curve
    else:
        flipped_img = cv2.flip(img, 1)
        x,y=np.transpose(noiseReducer(flipped_img,left,right,10,30))
        flipped_right_curve=draw_curve(flipped_img,x,y,left,right)
        curve=cv2.flip(flipped_right_curve,1)
        return curve


def curveTrace(img,left,right):
    height,width=img.shape

    # splitting the image to two parts 
    left_img=img[:,:width//2]
    right_img=img[:,width//2+1:]
    left_curve=None
    right_curve=None
    left_curve=curve_ready(left_img,left,right,30,5,1)
    right_curve=curve_ready(right_img,left,right,30,5,-1)
    img2=np.hstack((left_curve,right_curve))
    return img2


def overlayImage(image,curve):
    height,width,col=image.shape
    for i in range(curve.shape[0]):
        for j in range(curve.shape[1]):
            if(curve[i,j]!=0):
                for x in range(6):
                    try:
                        image[i,j+x]=[255,0,0]
                    except:
                        pass
    return image


def detectCurve(img):
    image=np.copy(img)

    # Getting the canny and original image
    ImagePrepocessor=PrepareImage((7,7),(0,0),False,50,150,0.6,0.6)

    image=ImagePrepocessor.get_binary_image(image)
    left,right=ImagePrepocessor.get_poly_maskpoints(image)
    

    # Tracing the curve
    try:
        curve = curveTrace(image,left,right)
    except:
        return img
    image = overlayImage(img, curve)
    return image



def main():
    cap = cv2.VideoCapture('Test Data/curve_road.mp4')
    i=0
    while(cap.isOpened()):
        ret,frame=cap.read()
        if(ret):
            frame=cv2.resize(frame,(600,400))
            if(i%3==0):
                i=0
                curve=detectCurve(frame)
            cv2.imshow('curve', curve)
            if(cv2.waitKey(1)&0xff == ord('q')):
                break
        else:
            break
        i+=1
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

