# RAAHI
## Navigation for the Visually Impaired
How does a visually impaired person find his or her way around anywhere? Can computer vision help in that , atleast to some extent?
### Objective
To make a computer vision model which detects the lanes, objects and obstacles in a person's path and provides 
information for his or her navigation.<br>

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
 
## Lane Detection - openCV
 > Contributed by [Harshit Gupta](https://github.com/TheGupta2012)
 - To run 
    - In the file Step By Step Lane.ipynb, replace the line of code 
    > ```%cd "E:/InnerveHackathon/"```
    with the directory in which your file resides.
    - In the last cell of the same file, add the name of your input file with extension : 
    > ```cap = cv.VideoCapture("name_of_vid.mp4")```
    - Run all cells of the file and 4 output windows should appear on your screen.
    - Press <b>q</b> to quit the windows any time.
 - The openCV library of python was used to detect the lanes in the frames of our dataset.
 - The hierarchy algorithms used were - 
    - Grayscaling for elimination of RGB channels.
    - Gaussian Blurring for removing noise
    - Canny Edge Detection and Image segmentation
    - Hough transform to detect prominent lines from detected edges.
 - The final lines resulting from Hough Transform were overlayed on the frames to produce final output.
 - <b>Final Output on Dataset</b><br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/openCV%20Lanes/Snippets/snipgif.gif" width = 420px height = 360px><br>

## Lane Detection - CNN and YOLO Object Detection
 > Contributed by [Aditya Karn](https://github.com/AdityaKarn)
 - A standard implementation of deep neural networks was used to overcome the shortcomings of the Lane Detection through openCV. Some of them 
  were <b>not being able to detect the curvature of the path of a lane</b> and <b>shifting of lines regarding to the noise present in the image
  such as gravel on road or patterns in a sidewalk</b>
## Lane Detection - CNN
 - Since this was not a straightforward classification or regression problem, we had to first identify what to predict with CNN. 
 - With CNN, we predicted 12 things. What were they? We assumed a lane to be majorly composed of six <b>anchor points</b> and tried to predict the x and y
   co-ordinates of anchor points of the left and the right lanes through CNN.
 - OUR DATASET LABELLER<br>
 <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/CNN%20Lanes/Labelling%20Script%20and%20Labels/labeller-ss.jpg" width = 280px height = 200px><br>
 - We used a standard implementation of a CNN for the predictions which turned out to be better than expected, given that the dataset was limited 
   and the number of epochs were limited to 40 to avoid overfitting.
 - OUR RESULTS<br>
   <img src = "https://github.com/TheGupta2012/RAAHI/blob/master/CNN%20Lanes/Results/cnn4.jpg" width = 280px height = 400px>
## Object Detection - YOLO Algorithm
- The task of <b>Object Detection</b> in video frames was implemented using the YOLO algorithm on pretrained weights.
- We detetected numerous classes of objects that were termed as obstacles for the 
visually impaired and produced an output on the fly. The neural net model used for the predictions was <b>Darknet-53</b>
- It can be clearly seen that the results obtained on the frames were quite clear and the accuracy was very high.
- OUR RESULTS<br>
<img src = "https://github.com/TheGupta2012/RAAHI/blob/master/YOLO/Results/yolo1.png" width = 380px height = 290px> <br>
> Implementation details available [here](https://github.com/AdityaKarn/innerve-hackathon#how-to-run)

## Where to go from here?
- Our project currently comprises of three different models that are used for lane detection and object detection separately. We intend
 to combine all our models and make a final overlayed frame that is capable of detection of lanes as well as objects.
- Another aspect of our project that needs to be looked over is deployment. We intend to deploy our project through the upcoming 
  Tensorflow lite for mobile applications and hope to make this into a small application that can be personalized according to the user.
- Last but not the least, linking GPS functionality to our project is the one thing that would enable us to give directions for not only
 safe navigation but also ensure the user can reach his or her destination independently with little or no help.
 
