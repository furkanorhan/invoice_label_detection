import cv2
import numpy as np
from PIL import Image
from IPython.display import display
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Perform prediction
image_path = "invoice_sample.jpeg"
results = model(image_path, conf=0.3)

# Load the original image (OpenCV BGR format)
img = cv2.imread(image_path)

# Loop through predicted boxes and labels
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    cls = int(box.cls[0])                  # Class ID
    conf = float(box.conf[0])              # Confidence score
    label = results[0].names[cls]          # Label name

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write label (small font)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA)

# Convert BGR to RGB (for display purposes)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save the image with bounding boxes
output_path = "output_with_boxes.jpg"
cv2.imwrite(output_path, img)
print(f"Saved result image: {output_path}")
