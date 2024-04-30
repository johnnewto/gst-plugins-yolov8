# gst-plugins-yoloV8
- based on the good works of https://github.com/jackersson/gst-plugins-tf
- Allows to run yoloV8 inference and inject metadata labels into Gstreamer Pipeline in Python
- [COCO Labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data)

## Installation

First read and or follow https://docs.ultralytics.com/quickstart/

the following might work, i used `pipreqs . ` to generate the requirements.txt
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt
```

### Install Tensorflow
- Tested on TF-GPU==2.13 (CPU)
#### TF-CPU
```bash
pip install tensorflow~=2.13
```

```bash
pip install tensorflow-gpu~=2.13
```

## Usage

### Run example
```bash
./run_example.sh
```

### To enable plugins implemented in **gst/python**
```bash
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
```

### Plugins
#### gst_yoloV8_detection -> gst_detection_overlay
    gst-launch-1.0 filesrc location="data/videos/paragliders.mp4" ! \
    decodebin ! videoconvert ! \
    gst_yoloV8_detection model=yolov8m.pt ! videoconvert ! gst_detection_overlay ! \
    videoconvert ! videoscale ! video/x-raw,width=1280,height=960 ! autovideosink
##### Parameters

 - **model**: local copy of 
       https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt


### Utils
 - [convert_labels_pbtxt_to_yml](https://github.com/jackersson/gst-plugins-tf/blob/master/utils/convert_labels_pbtxt_to_yml.py)
 
       python convert_labels_pbtxt_to_yml.py -f mscoco_label_map.pbtxt


### Additional
#### Enable/Disable TF logs
```bash
export TF_CPP_MIN_LOG_LEVEL={0,1,2,3,4,5 ...}
```

#### Enable/Disable Gst logs
```bash
export GST_DEBUG=python:{0,1,2,3,4,5 ...}
```

#### Enable/Disable Python logs
```bash
export GST_PYTHON_LOG_LEVEL={0,1,2,3,4,5 ...}
```
       
## License
MIT License
