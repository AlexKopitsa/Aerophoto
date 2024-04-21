import os.path

import inline
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

#Default model trained on the COCO dataset
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#Custom model trained on the VEDAI dataset
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)

def image_capture():
    #img = str(input("Enter image URL:"))
    #image URL: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzzApXNGeoJVWR3GhO-3bjzMXVq5YwReYhiGQvE9pL9A&s'
    #image URL: 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Droga_ekspresowa_S5S10_Stryszek-Bia%C5%82e_B%C5%82ota_a.jpg'
    img = os.path.join('yolov5/train/images/00000045_jpg.rf.aba8b608499fc826627f1f2c3b36116a.jpg')
    results = model(img)
    results.print()
    matplotlib.use('TkAgg')
    plt.imshow(np.squeeze(results.render()))
    plt.show()

def video_capture():
    video_file = str(input("Enter video file name:"))
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        results = model(frame)
        cv2.imshow("video_capture", np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#image_capture()
#video_capture()




