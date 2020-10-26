#%%
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import copy
import pickle
import time
#%%
vid_path = 'valid.mp4'

#%%
cv2.namedWindow("left: green --- right: blue", cv2.WINDOW_KEEPRATIO)
cap = cv2.VideoCapture(vid_path)

def set_points(event, x, y,flags, params):
    img = params['img']
    click_count = params['click_count']
    right_lane = params['right_lane']
    left_lane = params['left_lane']
    top = params['top']
    bottom = params['bottom']
    click_count = params['click_count']

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
            top = y
            arr = np.geomspace(top, bottom, 6)
            for val in arr:
                val = int(val)
                left_lane.append([0, val])
                right_lane.append([0, val])
                cv2.line(img, (0, val), (img.shape[1], val), color=(0,0,255), thickness=2)
        elif click_count in range (1,7):
            left_lane[click_count-1][0] = x
            cv2.circle(img, (x, left_lane[click_count-1][1]), radius=3, thickness=6, color=(0,255,0))
        elif click_count in range (7,13):
            right_lane[click_count-7][0] = x
            cv2.circle(img, (x, right_lane[click_count-7][1]), radius=3, thickness=3, color=(255,0,0))
        params['lane_img'] = copy.deepcopy(img)
        click_count += 1
        if click_count == 13:
            params['done'] = True
    if event == cv2.EVENT_MOUSEMOVE:
        if click_count > 0:
            img = copy.deepcopy(params['lane_img'])
            if click_count in range (1,7):
                img = cv2.circle(img, (x, left_lane[click_count-1][1]), radius=3, thickness=3, color=(0,255,0))
            elif click_count in range (7,13):
                img = cv2.circle(img, (x, right_lane[click_count-7][1]), radius=3, thickness=3, color=(255,0,0))

    params['img'] = img
    params['click_count'] = click_count
    params['top'] = top
    params['bottom'] = bottom
    params['right_lane'] = right_lane
    params['left_lane'] = left_lane

#%%
def save_label(params):
    img = params['org_img']
    right_lane = params['right_lane']
    left_lane = params['left_lane']
    img_scale = params['img_scale']

    img = cv2.resize(img, (int(img.shape[1]/img_scale), int(img.shape[0]/img_scale)))
    left_lane = np.array(left_lane)
    right_lane = np.array(right_lane)
    left_lane = left_lane/img_scale
    right_lane = right_lane/img_scale
    left_lane = left_lane.astype(int)
    right_lane = right_lane.astype(int)
    label = {
        'img': img,
        'left_lane': left_lane,
        'right_lane': right_lane
    }
    name = f"{time.time()}.p"
    pickle.dump(label, open(f"labels/{name}", "wb"))
    print(f"Saved label: {name}")
#%%
def draw_points(img, right_lane, left_lane):
    print (right_lane)
    for point in right_lane:
        cv2.circle(img, (point[0], point[1]), radius=3, thickness=3, color=(255,0,0))
    for point in left_lane:
        cv2.circle(img, (int(point[0]), int(point[1])), radius=3, thickness=3, color=(0,255,0))
    cv2.imshow("points", img)
    return img
#%%
def label(cap, frame_num):
    # Set the video frame
    cap.set(1, frame_num)
    # load the image
    ret,img = cap.read() 
    if not ret:
        print("Video Read error, or end of the video")
        return 0
    # Assign the img to img
    img_edit = copy.deepcopy(img)
    # Create window, assign callback
    params = {
        'org_img' : img,
        'lane_img' : img,
        'img' : img_edit,
        'click_count' : 0,
        'top' : 0,
        'bottom': img_edit.shape[0] - 100,
        'right_lane' : [],
        'left_lane':[],
        'img_scale':4,
        'done': False
    }
    cv2.setMouseCallback("left: green --- right: blue", set_points, param=params)
    while True:
        cv2.imshow("left: green --- right: blue",params['img'])
        key = cv2.waitKey(33)
        done = params['done']

        frame_dict = {
            ord('a'): 12,
            ord('s'): 50,
            ord('d'): 100,
            ord('f'): 1000,
            ord('z'):-12,
            ord('x'): -50,
            ord('c'): -100,
            ord('v'): -1000
        }
        if key != -1:
            increment = frame_dict.get(key, 0)
            if (increment != 0):
                frame_num += increment
                if done:
                    print (frame_num)
                    save_label(params)
                break
            
        # To quit the app press q
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        # To skip current label even if all the points have been set press w
        elif key == ord('w'):
            break
    # Cleanup after the function
    del params
    del img
    del img_edit
    return label(cap, frame_num)
label(cap, 1)