# RAAHI
## Navigation for the Visually Impaired
How does a visually impaired person find his or her way around anywhere? Can computer vision help in that , atleast to some extent?
### Objective
To make a computer vision model which detects the lanes, objects and obstacles in a person's path and provides 
information for his or her navigation.

### Tasks
<ul><li> To make a lane detector which identifies walkable lanes and produce an output on the fly.</li>
    <li> To produce an object detection model which identifies potential obstacles in the path of 
      an impaired person.</li>
</ul>

### What we did 
 - Used openCV for producing a lane detector. Inherent techniques used were Canny detection and Hough transforms.
 - Used Convolutional Neural Networks to work upon the limitations of the openCV model.'
 - Used the standard YOLO algorithm for the implementation of the object detectors.
 
# PROCESS
## openCV
 - Demonstrating the process on a sample image <br>
 <img src ="https://github.com/TheGupta2012/RAAHI/blob/master/Detection%20Stages%20and%20%20Examples/cannyOrig.jpg" width = 360px height = 260px>
 - Gaussian Blur <br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/Detection%20Stages%20and%20%20Examples/GaussianBlur.png" width = 360px height = 260px>
 - Canny Filtered and Segmented Canny<br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/Detection%20Stages%20and%20%20Examples/Cannysample.jpg" width = 360px height = 260px style="float:left">
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/Detection%20Stages%20and%20%20Examples/Segemented%20Canny%20Sample.png" width = 360px height  =260px>
 
## Final Output on Dataset
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/Snippets/snipgif.gif" width = 420px height = 360px>
