from ImagePrep import PrepareImage
from CurveDetector import Curve
import cv2
def main():
    cap = cv2.VideoCapture('Test Data/curve_road.mp4')
    i = 0
    ImagePreprocessor = PrepareImage((7,7),(0,0),False,50,170,0.6,0.6)
    CurveMaker = Curve()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            frame = cv2.resize(frame, (600, 400))
            if(i % 3 == 0):
                i = 0
                image=ImagePreprocessor.get_binary_image(frame)
                left, right = ImagePreprocessor.get_poly_maskpoints(image)
                curve=CurveMaker.curveTrace(image,left,right)
                curve=CurveMaker.drawCurve(frame,curve,(255,0,0),6)
            cv2.imshow('curve', curve)
            if(cv2.waitKey(1) & 0xff == ord('q')):
                break
        else:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
