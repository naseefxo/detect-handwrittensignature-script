import os
import cv2
import supervision as sv
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Paths

REPO_ID = "tech4humans/yolov8s-signature-detector"
MODEL_FILE = "yolov8s.pt"



# Download model
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

# Load YOLO model
model = YOLO(model_path)
# Step 3: Load your image
image_path = "handwritten4.png"  # Change this to your image
image = cv2.imread(image_path)
# Step 4: Perform inference
try:
    results = model(image)
except Exception as e:
    print("Error during inference:", e)
    exit(1)

# Step 5: Visualize detections using Supervision
for result in results:
    detections = sv.Detections.from_ultralytics(result)
    box_annotator = sv.BoxAnnotator()



    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    # Display result
    cv2.imshow("Handwritten Signature Detections", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optional: Save annotated image
    cv2.imwrite("annotated_result.jpg", annotated_image)



print("✅ Signature detection complete.")


