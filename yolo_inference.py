import os
from ultralytics import YOLO

# Get absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Full path to model and video
model_path = os.path.join(script_dir, 'models', 'best.pt')
video_path = os.path.join(script_dir, 'input_videos', '08fd33_4.mp4')

# Load the YOLO model
model = YOLO(model_path)

# Run inference on video
results = model.predict(video_path, save=True)

# Print prediction info
print(results[0])
print('==========================')
for box in results[0].boxes:
    print(box)
