#import necessary packages
import argparse
import numpy as np
import cv2 as cv
import time
from inference_utils import PPHumanSeg
from utils import background_blur, background_replace, face_distort, face_replace,custom_filter

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

frame_count = 0
start_time = time.time()

# define gstreamer pipeline to put together all multimedia workflows
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

print("here")
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
]

# defined all arguments
parser = argparse.ArgumentParser(description='PPHumanSeg (https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/PP-HumanSeg)')
parser.add_argument('--model', '-m', type=str, default='human_segmentation_pphumanseg_2023mar.onnx',
                    help='Usage: Set model path, defaults to human_segmentation_pphumanseg_2023mar.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
args = parser.parse_args()

#  main body of the function
if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath='./models/human_segmentation_pphumanseg_2023mar.onnx', backendId=backend_id, targetId=target_id)
 
    deviceId = 0
    cap = cv.VideoCapture(deviceId)
    # cap = cv.VideoCapture(gstreamer_pipeline(flip_method=2), cv.CAP_GSTREAMER) # initiate CSI camera
    d_mode = 0
    if cap.isOpened():
        window_handle = cv.namedWindow("CSI Camera", cv.WINDOW_AUTOSIZE)
        # Window
        count = 0
        while cv.getWindowProperty("CSI Camera", 0) >= 0:
            #capture frames
            ret, frame = cap.read() 
            if ret:
                # keyboard press detection enables for each modes
                if d_mode == 0:
                    img = frame
                elif d_mode == 1 or d_mode == 2:
                    mask = model.infer(frame)
                    mask = np.where(np.asarray(mask[0])!=0,1, np.asarray(mask[0]))
                    mask = np.stack([mask, mask, mask], axis=-1)
                    blend_img = frame * mask
                    if d_mode == 1:
                        img = background_blur(frame, blend_img, mask)
                    elif d_mode == 2:
                        img = background_replace(mask, blend_img)
                elif d_mode == 3:
                    img = face_distort(frame)
                elif d_mode == 4:
                    img = face_replace(frame)
                elif d_mode == 5:
                    img, flag = custom_filter(frame,count)             
                    if flag:
                        count = 0       
                    count += 1      
                cv.imshow("CSI Camera", img)
            keyCode = cv.waitKey(1)
            if keyCode == ord("q"):
                break
            elif keyCode == ord("0"):
                d_mode = 0
            elif keyCode == ord("1"):
                d_mode = 1
            elif keyCode == ord("2"):
                d_mode = 2
            elif keyCode == ord("3"):
                d_mode = 3
            elif keyCode == ord("4"):
                d_mode = 4
            elif keyCode == ord("5"):
                d_mode = 5
            # Increment frame count
            frame_count += 1

            # Calculate FPS every 10 frames
            if frame_count % 10 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
        cap.release()
        cv.destroyAllWindows()
    else:
        print("Unable to open camera")