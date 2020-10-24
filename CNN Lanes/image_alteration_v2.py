#%%
import math
import numpy as np
import pickle
import os
import cv2
import copy
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug import ia as ia
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
    img = copy.copy(img)
    for point in right_lane:
        cv2.circle(img, (int(point[0]), int(point[1])), radius=4, thickness=-1,  color=(255,0,0))
    for point in left_lane:
        cv2.circle(img, (int(point[0]), int(point[1])), radius=4, thickness=-1, color=(0,255,0))
    return img
#%%
def draw_pixels(src_img, lane):
    src_img = copy.copy(src_img)
    img = np.zeros_like(src_img)
    for points in lane:
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
def pts_2_imgaug_keypts(points, img):
    kps = []
    for point in points:
        keypoint = Keypoint(x=point[0], y=point[1])
        kps.append(keypoint)
    kpsoi = KeypointsOnImage(kps, shape=img.shape)
    return kpsoi
#%%
def lens_distort(img, right_lane, left_lane, amount=10):
    org_img = copy.copy(img)
    org_rl = copy.copy(right_lane)
    org_ll = copy.copy(left_lane)
    
    amount = random.randint(0, abs(amount))
    width = img.shape[1]
    height = img.shape[0]

    distCoeff = np.zeros((4,1),np.float64)
    distCoeff[0,0] = 0.0001 * amount # barell distortion less than 0 pincushion > 0 recommended max 0.003

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 12.        # define focal length x
    cam[1,1] = 12.        # define focal length y

    right_lane = draw_pixels(img, right_lane)
    left_lane = draw_pixels(img, left_lane)

    # Apply the distortion to the images
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
    # Zoom in to avoid the poinst being small
    zoom = iaa.Affine(scale={'x': zoom_amt, 'y': zoom_amt})
    img, right_lane, left_lane = zoom.augment_images((img, right_lane, left_lane))

    # Convert pictures of blobs into points
    right_lane = blobs_to_lane(right_lane)
    left_lane = blobs_to_lane(left_lane)

    # If any of the points got lost due to transformation return original image and points
    if(len(right_lane) + len(left_lane) < 12):
        return(org_img, org_rl, org_ll)
    return(img, right_lane, left_lane)
#%%
def distances(width, height, right_lane, left_lane, amount=1):
    both_lanes = np.concatenate((right_lane, left_lane), 0)
    xs = both_lanes[:,0]
    ys = both_lanes[:,1]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    dist_x1 = int(x1 * amount) # Distance from the left side of the image
    dist_x2 = int((width - x2) * amount) # Distance from the right side of the image
    dist_y1 = int(y1 * amount) # Distance from the top  of the image
    dist_y2 = int((height - y2) * amount) # Distance from the bottom of the image

    return dist_x1, dist_x2, dist_y1, dist_y2
#%%
def rotate(img, right_lane, left_lane, amount = 0.9):
    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)

    roation = random.randint(-15 * abs(amount), 15 * abs(amount))
    aug = iaa.Affine(rotate=roation)
    img, keypoints = aug(image=img, keypoints = (right_lane, left_lane))

    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    right_lane = np.array(right_lane, dtype=int)
    left_lane = np.array(left_lane, dtype=int)

    return img, right_lane, left_lane
#%%
def adaptive_centering(img, right_lane, left_lane, amount = 0.9, adaptive_centering_amt = 0.2):
    dist_x1, dist_x2, dist_y1, dist_y2 = distances(img.shape[1], img.shape[0], right_lane, left_lane, amount)
    dist_y1 -= 0  #Image is later cropped to 200px so make sure it doesn't zoom too far
    # Find translate amounts for centering
    translate_x = int(((dist_x2 - dist_x1) / 2) * adaptive_centering_amt)
    translate_y = int(((dist_y2 - dist_y1) / 2) * adaptive_centering_amt)
    # Convert the points
    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)
    # Apply centering
    aug = iaa.Affine(translate_px={'x': translate_x, 'y':translate_y})
    img, keypoints = aug(image=img, keypoints = (right_lane, left_lane))
    # Revert back to array points
    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    return img, right_lane, left_lane 
#%%
def smart_zoom(img, right_lane, left_lane, amount = 0.9, adaptive_centering_amt = 0.1):
    width = img.shape[1]
    height = img.shape[0]

    # img, right_lane, left_lane = adaptive_centering(img, right_lane, left_lane, amount, adaptive_centering)
    img, right_lane, left_lane = adaptive_centering(img, right_lane, left_lane, amount=amount, adaptive_centering_amt=adaptive_centering_amt)
    # Calculate distances after centering
    dist_x1, dist_x2, dist_y1, dist_y2 = distances(width, height, right_lane, left_lane, amount)
    # Find the zoom amounts
    zoom_x = min([dist_x1, dist_x2]) * 2
    zoom_y = min([dist_y1, dist_y2]) * 2

    if zoom_x < zoom_y:
        offset = random.uniform(-0.10, (zoom_x / width)) 
        zoom = 1 + (offset * amount)
    else:
        offset = random.uniform(-0.10, (zoom_y / height))
        zoom = 1 +  (offset * amount)

    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)

    aug = iaa.Affine(scale={'x': zoom, 'y':zoom})
    img, keypoints = aug(image=img, keypoints = (right_lane, left_lane))
    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    right_lane = np.array(right_lane, dtype=int)
    left_lane = np.array(left_lane, dtype=int)

    return img, right_lane, left_lane
#%%
# This function is similar to smart_zoom but it only scales the image on y axis to simulate wider roads
def smart_stretch(img, right_lane, left_lane, amount = 1, adaptive_centering_amt = 0):
    width = img.shape[1]
    height = img.shape[0]

    # img, right_lane, left_lane = adaptive_centering(img, right_lane, left_lane, amount, adaptive_centering)
    img, right_lane, left_lane = adaptive_centering(img, right_lane, left_lane, amount=amount, adaptive_centering_amt=adaptive_centering_amt)
    dist_x1, dist_x2, _, _ = distances(width, height, right_lane, left_lane, amount)
    # Find the zoom amounts
    stretch = min([dist_x1, dist_x2]) * 2
    stretch_min = 0.7
    stretch_max = (width + stretch) / width
    # Get random zoom amount
    stretch = random.uniform(stretch_min, stretch_max)
    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)
    aug = iaa.Affine(scale={'x': stretch, 'y':1})
    img, keypoints = aug(image=img, keypoints = (right_lane, left_lane))
    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    right_lane = np.array(right_lane, dtype=int)
    left_lane = np.array(left_lane, dtype=int)

    return img, right_lane, left_lane
#%%
def smart_shear(img, right_lane, left_lane, amount = 0.9):
    # This approach is not that good due to the points not always being in the center of the screen. It's good enough for image generator though
    #  d = sin(a) * b
    width = img.shape[1]
    height = img.shape[0]
    dist_x1, dist_x2, dist_y1, dist_y2 = distances(width, height, right_lane, left_lane, amount)

    # Workaround for a rare case when x distance is greater than image height 
    if (dist_x1 > height):
        dist_x1 = height - 1
    if (dist_x2 > height):
        dist_x2 = height - 1
    max_lean_left = int(math.asin(dist_x1 / height) * (57.2958 * amount))
    max_lean_right = int(math.asin(dist_x2 / height) * (57.2958 * amount)) 

    lean = min([max_lean_left, max_lean_right])
    lean = random.randint(-abs(lean), abs(lean))

    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)
    aug = iaa.Affine(shear=lean)
    img, keypoints = aug(image=img, keypoints = (right_lane, left_lane))
    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    right_lane = np.array(right_lane, dtype=int)
    left_lane = np.array(left_lane, dtype=int)

    return img, right_lane, left_lane
#%%
def smart_translate(img, right_lane, left_lane, amount = 0.9):
    width = img.shape[1]
    height = img.shape[0] 

    dist_x1, dist_x2, dist_y1, dist_y2 = distances(width, height, right_lane, left_lane, amount)

    right_lane = pts_2_imgaug_keypts(right_lane, img)
    left_lane = pts_2_imgaug_keypts(left_lane, img)

    aug_x = random.randint(-abs(dist_x1), abs(dist_x2))
    # Image's top 26%  thats why 26% of height is added to crop
    aug_y = random.randint(-abs(dist_y1) + int(height * 0.26), abs(dist_y2))

    aug = iaa.Affine(translate_px={"x": aug_x, "y": aug_y})
    img, keypoints = aug(image = img, keypoints = (right_lane, left_lane))

    right_lane = keypoints[0].to_xy_array()
    left_lane = keypoints[1].to_xy_array()
    right_lane = np.array(right_lane, dtype=int)
    left_lane = np.array(left_lane, dtype=int)

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
    contrast = iaa.ContrastNormalization((0.4, 1.7))
    img = contrast.augment_image(img)
    # Make some images brighter and some darker.
    # In 40% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    brightness = iaa.Multiply((0.8, 2), per_channel=0.4)
    img = brightness.augment_image(img)
    saturation = iaa.AddToHueAndSaturation((-10, 10), per_channel=0.2)
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
# Function which applies the alteration with recommended amount 
def default_alter(img, right_lane, left_lane):
    if np.random.rand() < 0.3:
        img = perlin_shadows(img)
    if np.random.rand() < 0.6:
        img, right_lane, left_lane = rotate(img, right_lane, left_lane, amount=1)
    if np.random.rand() < 0.2:
        img, right_lane, left_lane = lens_distort(img, right_lane, left_lane, 50)
    if np.random.rand() < 0.5:
        img, right_lane, left_lane = smart_shear(img, right_lane, left_lane, amount=1.4)
    if np.random.rand() < 0.6:
        img, right_lane, left_lane = smart_zoom(img, right_lane, left_lane)
    if np.random.rand() < 0.6:
        img, right_lane, left_lane = smart_stretch(img, right_lane, left_lane)
    if np.random.rand() < 0.7:
        img, right_lane, left_lane = smart_translate(img, right_lane, left_lane, amount=0.9)
    if np.random.rand() < 0.5:
        img = color_transforms(img)
    if np.random.rand() < 0.2:
        img = blur(img, 1)
    if np.random.rand() < 0.3:
        img = gaussian_noise(img)
    return img, right_lane, left_lane
#%%
# TESTER
# cv2.namedWindow("test", cv2.WINDOW_KEEPRATIO)
# for label in labels_pahts[0:1000]:
#     if label.endswith(".p"):
#         start_time = time.time()
#         label = pickle.load(open(labels_path + label, "rb"))
#         img = label['img']
#         left_lane = label['left_lane']
#         right_lane = label['right_lane']

#         img, right_lane, left_lane = default_alter(img, right_lane, left_lane)

#         ddist_img = draw_points(img, right_lane, left_lane)
        
#         fps =  int(1.0 / (time.time() - start_time))
#         perf = fps/60
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         ddist_img = cv2.putText(ddist_img,f'FPS: {fps}',(30,30), font, 1,(0, perf * 255,(1-perf) * 255 ),2,cv2.LINE_AA)
#         cv2.line(img, (0, 70), (480, 70), color = (255,0,0), thickness=1, lineType=8, shift=0)
#         cv2.line(img, (240, 0), (240, 270), color = (255,0,0), thickness=1, lineType=8, shift=0)
#         cv2.imshow("test", ddist_img)


#         if(len(left_lane) + len(right_lane) < 12):
#             print("Missing Points!" )
#             cv2.waitKey(0)
#         key = cv2.waitKey(1)
#         if key==ord('q'):
#             break
#%%

