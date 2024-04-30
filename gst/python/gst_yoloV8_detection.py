"""
Usage
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
    export GST_DEBUG=python:4

    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
        gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! autovideosink
"""

import os
import logging
import pdb
import cv2
from ultralytics import YOLO
import typing as typ
import yaml
import numpy as np

from gstreamer import Gst, GObject, GstBase, GstVideo
import gstreamer.utils as utils
from gstreamer.gst_objects_info_meta import gst_meta_write


def _get_log_level() -> int:
    return int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


log = logging.getLogger('gst_python')
log.setLevel(_get_log_level())


# def _is_gpu_available() -> bool:
#     """Check is GPU available or not"""
#     try:
#         from tensorflow.python.client import device_lib
#         return any(d.device_type == 'GPU' for d in device_lib.list_local_devices())
#     except ImportError:
#         return os.path.isfile('/usr/local/cuda/version.txt')


# def _parse_device(device: str) -> str:
#     """Parse device on which run model
#     Device value examples:
#         - GPU|CPU
#         - CPU
#         - CPU:0
#         - GPU
#         - GPU:0
#         - GPU:1
#     For device name "GPU|CPU": use GPU if available else use CPU
#     """

#     result = device
#     if device == 'GPU|CPU':
#         result = 'GPU' if _is_gpu_available() else 'CPU'

#     if 'GPU' in result and not _is_gpu_available():
#         raise ValueError('Specified "{}" device but GPU not available'.format(device))

#     return result if ':' in result else f'{result}:0'


def is_gpu(device: str) -> bool:
    return "gpu" in device.lower()

# todo jn
# def create_config(device: str = 'CPU', *,  jn
#                   per_process_gpu_memory_fraction: float = 0.0,
#                   log_device_placement: bool = False) -> tf.ConfigProto:
# def create_config(device: str = 'CPU', *,
#                   per_process_gpu_memory_fraction: float = 0.0,
#                   log_device_placement: bool = False):
#     # """Creates tf.ConfigProto for specifi device"""
#     # config = tf.compat.v1.ConfigProto(log_device_placement=log_device_placement)
#     if is_gpu(device):
#         if per_process_gpu_memory_fraction > 0.0:
#             config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
#         else:
#             config.gpu_options.allow_growth = True
#     else:
#         config.device_count['GPU'] = 0

#     return config






class GstYoloDetectionPluginPy(GstBase.BaseTransform):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/

    GST_PLUGIN_NAME = 'gst_yoloV8_detection'

    __gstmetadata__ = ("Name",
                       "Transform",
                       "Description",
                       "Author")

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format=RGB"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format=RGB"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (str,
                  "model",
                  "YOLOv8",
                  None,  # default
                  GObject.ParamFlags.READWRITE),

        "config": (str,
                   "Path to config file",
                   "not sure if needed , Contains path to config *.yml supported by ultralytics YOLOv8 model (e.g. yolov8n.yaml)",
                   None,  # default
                   GObject.ParamFlags.READWRITE
                   ),
    }

    def __init__(self):
        super().__init__()

        self.model = None
        # print("Loading yolov8n model")
        # self.model = YOLO('yolov8m.pt')
        self.config = None

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if self.model is None:
            Gst.warning(f"No model speficied for {self}. Plugin working in passthrough mode")
            return Gst.FlowReturn.OK

        try:
            # Convert Gst.Buffer to np.ndarray
            image = utils.gst_buffer_with_caps_to_ndarray(buffer, self.sinkpad.get_current_caps())

            # model inference
            # objects = self.model.process_single(image)
            results = self.model.predict(source=image)  
            result = results[0].boxes.cpu().numpy()
            Gst.debug(f"Frame id ({buffer.pts // buffer.duration}). Detected {str(results)}")

            # for scores, boxes, labels in result['scores'], result['boxes'], result['labels']:
            detections = []
            for conf, xywh, cls in zip(result.conf, result.xywh, result.cls):

                class_name = str(self.model.names.get(cls.item()))
                if not class_name or conf < 0.35:
                    continue
                
                x = int(1  * (xywh[0] - xywh[2]//2))
                y = int(1 * (xywh[1] - xywh[3]//2))
                w = int(1  * xywh[2])
                h = int(1 * xywh[3])


                detections.append({
                    'confidence': float(conf),
                    'bounding_box': [x, y, w, h],
                    'class_name': class_name,
                })


            # write objects to as Gst.Buffer's metadata
            # Explained: http://lifestyletransfer.com/how-to-add-metadata-to-gstreamer-buffer-in-python/
            gst_meta_write(buffer, detections)
        except Exception as err:
            # pdb.set_trace()
            logging.error("Error %s: %s", self, err)

        return Gst.FlowReturn.OK

    def do_get_property(self, prop: GObject.GParamSpec):
        # pdb.set_trace()
        if prop.name == 'model':
            return self.model
        if prop.name == 'config':
            return self.config
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        # pdb.set_trace()
        if prop.name == 'model':
            self._do_set_model(value)
        elif prop.name == "config":
            # self._do_set_model(from_config_file(value))
            self.config = value
            Gst.info(f"Model's config updated from {self.config}")
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def _do_set_model(self, model):
        import gc
        # stop previous instance
        # pdb.set_trace()
        if self.model:
            # Currently, the Ultralytics library doesn’t provide a specific method to 'close' or 'destroy' the model
            del model
            # Collect garbage
            gc.collect()

        self.model = YOLO(model)

        # # start new instance
        # if self.model:
        #     self.model.startup()

    def __exit__(self, exc_type, exc_val, exc_tb):

        Gst.info(f"Shutdown {self}")

        if self.model:
            self.model.shutdown()

        Gst.info(f"Destroyed {self}")


# Required for registering plugin dynamically
# Explained: http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstYoloDetectionPluginPy)
__gstelementfactory__ = (GstYoloDetectionPluginPy.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstYoloDetectionPluginPy)
