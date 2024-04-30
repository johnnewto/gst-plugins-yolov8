from ultralytics import YOLO
import cv2
from sahi import AutoDetectionModel
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from sahi.utils.yolov8 import download_yolov8s_model
# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')
# Create a YOLOv8 model with agnostic NMS enabled
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    # device="cpu",  # or 'cuda:0'
    device='cuda:0'
    # agnostic_nms=True,
)
# With a numpy image
result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
result.export_visuals(export_dir="demo_data/")
# # # Create a new YOLO model from scratch
# # model = YOLO('yolov8n.yaml')

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
# results = model.track(source="data/videos/paragliders.mp4", show=True, save_dir='runs')  # Tracking with default tracker
# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg', save_dir='runs')
print(result   )
# cv2.imshow('image', results[0].plot())
# cv2.waitKey(0)

# # Perform object detection on a video using the model
# results = model('video.mp4')
# # Export the model to ONNX format
# success = model.export(format='onnx')