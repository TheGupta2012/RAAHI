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
 - To run 
    - In the file Step By Step Lane.ipynb, replace the line of code 
    > ```%cd "E:/InnerveHackathon/"```
    with the directory in which your file resides.
    - In the last cell of the same file, add the name of your input file with extension : 
    > ```cap = cv.VideoCapture("name_of_vid.mp4")```
    - Run all cells of the file and 4 output windows should appear on your screen.
 - <b>Final Output on Dataset</b><br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Snippets/snipgif.gif" width = 420px height = 360px><br>
 - Demonstrating the process on a sample image <br>
 <img src ="https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Detection%20Stages%20and%20%20Examples/cannyOrig.jpg" width = 360px height = 260px>
 - Gaussian Blur <br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Detection%20Stages%20and%20%20Examples/GaussianBlur.png" width = 360px height = 260px>
 - Canny Filtered and Segmented Canny<br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Detection%20Stages%20and%20%20Examples/Cannysample.jpg" width = 360px height = 260px style="float:left">
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Detection%20Stages%20and%20%20Examples/Segemented%20Canny%20Sample.png" width = 360px height  =260px>
 


## Convolutional Neural Networks 
 - A standard implementation of deep neural networks was used to overcome the shortcomings of the Lane Detection through openCV. Some of them 
  were <b>not being able to detect the curvature of the path of a lane</b> and <b>shifting of lines regarding to the noise present in the image
  such as gravel on road or patterns in a sidewalk</b>
  

 
