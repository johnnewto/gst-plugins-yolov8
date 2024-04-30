from ultralytics import YOLO
import cv2

# # Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
# results = model.track(source="data/videos/paragliders.mp4", show=True, save_dir='runs')  # Tracking with default tracker
# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg', save_dir='runs')
print(results   )
cv2.imshow('image', results[0].plot())
cv2.waitKey(0)

# # Perform object detection on a video using the model
# results = model('video.mp4')
# # Export the model to ONNX format
# success = model.export(format='onnx')