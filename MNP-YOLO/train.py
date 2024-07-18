from ultralytics import YOLOv8

# Load pre-trained model
model = YOLOv8.from_pretrained('ultralytics/yolov8n')

# Train the model with various hyperparameters
model.train(
    data='ultralytics/cfg/datasets/MP.yaml',
    epochs=200,
    batch=20,
    imgsz=512,
    weight_decay=0.001,
    degrees=0.05,
    translate=0.05,
    scale=0.05,
    shear=0.0
)

# Make predictions (optional, depends on your workflow)
model.predict()

# Export the model to ONNX format
model.export(format="onnx")
