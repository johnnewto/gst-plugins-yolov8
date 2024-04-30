#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=5
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
export GST_DEBUG=python:4


# gst-launch-1.0 filesrc location="/home/john/data/maui-data/karioitahi_13Aug2022/PhantomDrone/DJI_0087.MP4" ! \
gst-launch-1.0 filesrc location="data/videos/paragliders.mp4" ! \
 decodebin ! videoconvert ! \
 gst_yoloV8_detection model=models/yolov8m.pt ! videoconvert ! gst_detection_overlay ! \
 videoconvert ! videoscale ! video/x-raw,width=1280,height=960 ! autovideosink
# videoconvert ! videoscale ! video/x-raw,width=1280,height=960 ! x264enc ! mp4mux ! filesink location="output.mp4"
# /maui-data/karioitahi_13Aug2022/PhantomDrone/DJI_0087.MP4

#  gst-launch-1.0 multifilesrc location="/home/john/data/images/dir/image_%04d.jpg" index=1 caps="image/jpeg,framerate=1/1" ! jpegdec ! videoconvert ! \
#  gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! \
#  gst_detection_overlay ! videoconvert ! autovideosink