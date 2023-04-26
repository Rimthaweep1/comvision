import torch
import cv2
import yaml
from PIL import ExifTags, Image, ImageOps
#load model


model = torch.hub.load('ultralytics/yolov5', 'custom', path='sign.pt')
model.conf = 0.25 # NMS confidence threshold
model.iou = 0.45  # IoU threshold
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

    # detection part
video = cv2.VideoCapture(0) # Read USB Camera
while(video.isOpened()):
        # Read Frame
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        cv2.imshow('Object detector', cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR))
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'): 
            break
video.release()
cv2.destroyAllWindows()

