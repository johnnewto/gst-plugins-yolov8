"""
This script demonstrates how to use YOLOv8 model for object detection using GStreamer.

It loads a pretrained YOLOv8 model and performs object detection on an input image using GStreamer pipeline.
The detected objects are then overlaid on the image and displayed.

Requirements:
- OpenCV (cv2)
- NumPy
- GStreamer
- gst-python
- gi (GObject introspection)
- ultralytics (YOLOv8 model)

Usage:
1. Make sure you have all the required dependencies installed.
2. Set the path to the input image in the `im` variable.
3. Run the script.

Note: This script assumes that the YOLOv8 model file 'yolov8n.pt' is present in the current directory.

"""

import cv2
import numpy as np
from gst.python import gst_detection_overlay as gdo
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from gstreamer.gst_objects_info_meta import gst_meta_write

# Initialize GStreamer
Gst.init(None)

# Import the YOLO class from ultralytics
# find the labels
# model = YOLO('yolov8n.pt')
# print(model.names)


from ultralytics import YOLO

if __name__ == '__main__':
    # cv2 read image
    im = cv2.imread('bus.jpg')
    # Convert the image to RGB
    imageRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    imageBGRA = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    # Get the image shape
    height, width, channels = imageBGRA.shape
    # Get the image size
    buffer_size = height * width * channels
    # Get the image format
    video_format = 'RGB'

    buffer = Gst.Buffer.new_wrapped(bytes(imageBGRA))
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Run inference on an image
    # results = model('bus.jpg')  # results list
    results = model.predict(source=imageRGB)  


    # View results

    result = results[0].boxes.cpu().numpy()

    # for scores, boxes, labels in result['scores'], result['boxes'], result['labels']:
    detections = []
    for conf, xywhn, cls in zip(result.conf, result.xywhn, result.cls):

        class_name = str(model.names.get(cls.item()))
        if not class_name or conf < 0.35:
            continue
        
        x = int(width  * (xywhn[0] - xywhn[2]/2))
        y = int(height * (xywhn[1] - xywhn[3]/2))
        w = int(width  * xywhn[2])
        h = int(height * xywhn[3])


        detections.append({
            'confidence': float(conf),
            'bounding_box': [x, y, w, h],
            'class_name': class_name,
        })

    # Encode class_name as UTF-8 before passing to gst_meta_write
    gst_meta_write(buffer, detections)
    # Display the image with the detection overlay
    overlay = gdo.GstDetectionOverlay()
    # create Gst.Buffer
    overlay.set_height(height)
    overlay.set_width(width)    

    overlay.do_transform_ip(buffer)

    # convert Gst.Buffer to numpy array
    shape = (buffer.get_size(),)
    array = np.ndarray(shape=buffer_size ,
                       buffer=buffer.extract_dup(0, buffer_size),
                       dtype='uint8')

    array = array.reshape(height, width, channels).squeeze()

    cv2.imshow('Overlay', array)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    model.close()



