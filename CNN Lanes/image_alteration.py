#%%
import numpy as np
import pickle
import os
import cv2
import copy
from imgaug import augmenters as iaa
import random
import noise
from scipy import misc
from sklearn import preprocessing
from PIL import Image, ImageFilter
import time

labels_path = 'labels/'
labels_pahts = os.listdir(labels_path)
#%%
def draw_points(img, right_lane, left_lane):
    for point in right_lane:
        cv2.circle(img, (point[0], point[1]), radius=3, thickness=3, color=(255,0,0))
    for point in left_lane:
        cv2.circle(img, (int(point[0]), int(point[1])), radius=3, thickness=3, color=(0,255,0))
    return img
#%%
def draw_pixels(src_img, lane):
    src_img = copy.deepcopy(src_img)
    img = np.zeros_like(src_img)
    for points in lane:
        # img[points[1], points[0], 1] = 255
        cv2.circle(img, (points[0], points[1]), radius=1, thickness=5, color=(0,255,0))
    return img
#%%
def blobs_to_lane(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 3
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(img)

    points = []
    for keypoint in keypoints:
        points.append([int(keypoint.pt[0]), int(keypoint.pt[1])])
    points = sorted(points, key=lambda x: x[1])
    return points
#%%
def crop(img, width, height, center=True):
    if(center):
        d_x = int((img.shape[1] - width) / 2)
        d_y = int((img.shape[0] - height) / 2)
        _img = img[d_y:img.shape[0]-d_y, d_x:img.shape[1]-d_x]
    else:
        d_x = int((img.shape[1] - width))
        d_y = int((img.shape[0] - height))
        _img = img[d_y:img.shape[0], 0:img.shape[1]-int(d_x/3)]
    return _img

#%%
def lens_distort(img, right_lane, left_lane, amount=10):
    org_img = copy.deepcopy(img)
    org_rl = copy.deepcopy(right_lane)
    org_ll = copy.deepcopy(left_lane)
    
    amount = random.randint(0, amount)
    width = img.shape[1]
    height = img.shape[0]

    distCoeff = np.zeros((4,1),np.float64)

    k1 = 0.0001 * amount # barell distortion less than 0 pincushion > 0 recommended max 0.003
    k2 = 0
    p1 = 0 
    p2 = 0

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y

    right_lane = draw_pixels(img, right_lane)
    left_lane = draw_pixels(img, left_lane)

    img = cv2.undistort(img, cam, distCoeff)
    right_lane = cv2.undistort(right_lane, cam, distCoeff)
    left_lane = cv2.undistort(left_lane, cam, distCoeff)
    # Zoom in amount is based on my experiments. I dont know the maths to calculate it.
    if amount < 40:
        zoom_amt = 1 + 0.010 * amount
    elif amount < 100:
        zoom_amt = 1 + 0.007 * amount
    else:
        zoom_amt = 1 + 0.005 * amount
    img = cv2.resize(img, None, fx=zoom_amt, fy=zoom_amt)
    right_lane = cv2.resize(right_lane, None, fx=zoom_amt, fy=zoom_amt)
    left_lane = cv2.resize(left_lane, None, fx=zoom_amt, fy=zoom_amt)

    img = crop(img, width, height)
    right_lane = crop(right_lane, width,height)
    left_lane = crop(left_lane, width, height)

    right_lane = blobs_to_lane(right_lane)
    left_lane = blobs_to_lane(left_lane)
    # If any of the points got lost due to transformation return original image and points
    if(len(right_lane) + len(left_lane) < 12):
        img = org_img
        right_lane = org_rl
        left_lane = org_ll
    return(img, right_lane, left_lane)
#%%
def zoom_pan_rotate(img, right_lane, left_lane, amount):
    org_img = copy.deepcopy(img)
    org_rl = copy.deepcopy(right_lane)
    org_ll = copy.deepcopy(left_lane)

    right_lane = draw_pixels(img, right_lane)
    left_lane = draw_pixels(img, left_lane)

    random_scale = random.randint(70, 100)/100
    random_trans_x = random.randint(-16, 16)/100 
    random_trans_y = random.randint(-15, -10)/100 
    random_rotate = random.randint(-15, 15)
    random_shear = random.randint(-6,6)
    transform = iaa.Affine(
        scale={"x": (random_scale), "y": (random_scale)},
        translate_percent={"x": (random_trans_x), "y": (random_trans_y)},
        rotate=(random_rotate),
        shear=(random_shear)
    )
    img = transform.augment_image(img)
    right_lane = transform.augment_image(right_lane)
    left_lane = transform.augment_image(left_lane)

    right_lane = blobs_to_lane(right_lane)
    left_lane = blobs_to_lane(left_lane)

    if(len(right_lane) + len(left_lane) < 12):
        img = org_img
        right_lane = org_rl
        left_lane = org_ll
    return img, right_lane, left_lane
#%%
def blur(img, amount):
    blur = iaa.GaussianBlur(sigma=(0, amount))
    return blur.augment_image(img)
#%%
def gaussian_noise(img):
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel. For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the pixels.
    gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.08*255), per_channel=0.5)
    return gaussian_noise.augment_image(img)
#%%
def color_transforms(img):
    # Strengthen or weaken the contrast in each image.
    contrast = iaa.ContrastNormalization((0.75, 1.5))
    img = contrast.augment_image(img)
    # Make some images brighter and some darker.
    # In 40% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    brightness = iaa.Multiply((0.6, 1.4), per_channel=0.4)
    img = brightness.augment_image(img)
    saturation = iaa.AddToHueAndSaturation((-15, 15), per_channel=0.2)
    img = saturation.augment_image(img)
    return img
#%%
def perlin_shadows(img):
    width = img.shape[0]
    height = img.shape[1]
    shape = (int(width/7), int(height/7)) #Reduce resolution to improve speed (improves it a lot!)
    scale = random.randint(15,60)
    octaves = 1 #Lower value - less detailed perlin noise, but faster computation
    persistence = 10
    lacunarity = 1
    
    base = random.randint(0,500)
    perl_noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            perl_noise[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=width, 
                                        repeaty=height, 
                                        base=base)

    perl_noise = cv2.resize(perl_noise,(height,width)) #Resize perlin noise to fit the image
    # Perlin noise returns values from -0.5 to 0.5 (based on my observation) normalize it to 0 to 255 and convert to int
    perl_noise = cv2.normalize(perl_noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # Perl noise is only 1 layer deep, but for mixing it has to have as many diemensions as input
    perl_noise = cv2.cvtColor(perl_noise, cv2.COLOR_GRAY2BGR)
    # Increase contrast on the noise, so shadows are more pronounced
    contrast = iaa.ContrastNormalization((3, 3))
    perl_noise = contrast.augment_image(perl_noise)
    # Mix the images
    mix = cv2.addWeighted(img,1 ,perl_noise,0.2, -45)
    return mix
#%%
# TEST
# cv2.namedWindow("test", cv2.WINDOW_KEEPRATIO)
# for i in range(20):
#     for label in labels_pahts:
#         if label.endswith(".p"):
#             start_time = time.time()
#             label = pickle.load(open(labels_path + label, "rb"))
#             img = label['img']
#             left_lane = label['left_lane']
#             right_lane = label['right_lane']
            
#             if np.random.rand() < 0.5:
#                 img = perlin_shadows(img)
#             if np.random.rand() < 0.3:
#                 img, right_lane, left_lane = lens_distort(img, right_lane, left_lane, 50)
#             if np.random.rand() < 0.5:
#                 img, right_lane, left_lane = zoom_pan_rotate(img, right_lane, left_lane, 0.7)
#             if np.random.rand() < 0.2:
#                 img = blur(img, 1)
#             if np.random.rand() < 0.5:
#                 img = color_transforms(img)
#             if np.random.rand() < 0.5:
#                 img = gaussian_noise(img)
#             ddist_img = draw_points(img, right_lane, left_lane)
            

#             fps =  int(1.0 / (time.time() - start_time))
#             perf = fps/60
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             ddist_img = cv2.putText(ddist_img,f'FPS: {fps}',(30,30), font, 1,(0, perf * 255,(1-perf) * 255 ),2,cv2.LINE_AA)

#             cv2.imshow("test", ddist_img)

#             key = cv2.waitKey(0)
#             if(len(left_lane) + len(right_lane) < 12):
#                 print("Missing Points!")
#                 cv2.waitKey(0)
#             if key==ord('q'):
#                 break
#%%
