# utils functionality module for filters
import numpy as np
import cv2 as cv
from PIL import Image
import math

# background blurring filter
def background_blur(org, img, mask):   
    out = np.zeros_like(org) 
    img_copy = np.asarray(cv.blur(np.asarray(org),(15, 15)))
    rev_mask = np.where((mask==0)|(mask==1), mask^1, mask)
    remove_mask_resgion = img_copy*rev_mask
    image_array = np.where(remove_mask_resgion == 0, 1, remove_mask_resgion)
    # weighted addition of blurred frame captued and the mask  
    img_new = cv.addWeighted(image_array, 1, img, 1, 0)
    return img_new

# background replacing filter 
def background_replace(mask, blend_img):
    bg = Image.open('./images/3685070.jpg')
    bg = bg.resize((blend_img.shape[1], blend_img.shape[0]))
    bg = cv.cvtColor(np.asarray(bg), cv.COLOR_BGR2RGB)
    rev_mask = np.where((mask==0)|(mask==1), mask^1, mask)
    remove_mask_resgion = bg*rev_mask
    
    image_array = np.where(remove_mask_resgion == 0, 1, remove_mask_resgion)
    img_new = cv.addWeighted(np.asarray(image_array), 1, blend_img, 1, 0)
    
    return img_new

# face detection using haarcascade model
def face_detection(frame):
    face_cascade = cv.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(np.array(frame), cv.COLOR_BGR2GRAY)
    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# face distotion (pixel level)
def face_distortion_fn(img, w1, h1):
    h,w,_ = img.shape
    flex_x = np.zeros((h,w),np.float32)
    flex_y = np.zeros((h,w),np.float32)

    scale_y= 1
    scale_x = 1
    alpha = -1.8
    # take face center
    center_x, center_y = (w1 // 2, h1 // 2)
    radius = h/5

    # altering the pixel values for distortion effect
    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y

            if distance >= (radius * radius):
                flex_x[y, x] = x
                flex_y[y, x] = y
            else:
                theta = np.arctan2(delta_x,delta_y) + alpha*(radius-math.sqrt(distance))/radius
                r_sin = math.sqrt(distance)*np.cos(theta)
                r_cos = math.sqrt(distance)*np.sin(theta)
                flex_x[y, x] = r_cos + center_x
                flex_y[y, x] = r_sin + center_y

    #remapping the pixel values with new pixel information
    dst = cv.remap(img, flex_x, flex_y, cv.INTER_LINEAR)
    return dst

# face distortion filter
def face_distort(frame):
    faces = face_detection(frame)
    try:
        x, y, w, h = faces[0]
    except:
        return frame
    roi=frame[y:y+h, x:x+w]
    dis_roi = face_distortion_fn(roi, w, h)
    frame[y:y+h, x:x+w] = dis_roi
    return frame

# face replace with external mask image
def face_replace(frame):
    cat_img = Image.open('./images/cat_head.jpg')
    # finding the contours of the mask to get perfect overlaying
    gray = cv.cvtColor(np.array(cat_img), cv.COLOR_BGR2GRAY)
    _ , img_thresh = cv.threshold(gray, 0, 250, cv.THRESH_BINARY) 
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    big_cntrs = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 15000:
            big_cntrs.append(contour)
    final = cv.drawContours(np.asarray(cat_img), big_cntrs, 0, (0, 255, 0), 3)

    diff_img = final-cat_img

    mask1 = np.zeros_like(diff_img)
    mask2 = cv.drawContours(mask1, big_cntrs, 0, (1,1,1), -1)

    extract_cat = cat_img*mask2

    frame = np.asarray(frame)
    frame1 = frame.copy()

    faces = face_detection(frame)
    for face in faces:
        x, y, w, h = face
        try:
            roi=frame[y-30:y+h+30, x-30:x+w+30]
            face_mask = cv.resize(mask2,(roi.shape[0], roi.shape[1]))
        except:
            return frame
        extract_cat = cv.resize(extract_cat,(roi.shape[0], roi.shape[1]))
        face_mask = np.where((face_mask==0)|(face_mask==1), face_mask^1, face_mask)
        filter_img = roi*face_mask
        # overlay mask on the face detected in the frame coordinates
        replaced_img = cv.addWeighted(filter_img, 1,extract_cat,1,0)
        frame1[y-30:y+h+30, x-30:x+w+30] = replaced_img
    return frame1

# gif filter
def custom_filter(frame, count):
    video_path = './images/gif.mp4'
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    flag = False
    # read frames of gif
    for frame_num in range(frame_count):
        # Read a frame from the video
        ret, frame1 = cap.read()
        frames.append(frame1)      
    try:
        frame2 = cv.resize(frames[count//2],(frame.shape[1],frame.shape[0]))
    except:
        flag = True
        frame2 = cv.resize(frames[0],(frame.shape[1],frame.shape[0]))
    # overlay gif frames on real time frames
    dstimg = cv.addWeighted(np.array(frame),1,np.array(frame2),0.5,0)
    return dstimg, flag
