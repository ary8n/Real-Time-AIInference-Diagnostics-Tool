# AI Inference Diagnostic tool

This project is a real-time diagnostics pipeline for image classification models, designed for robust and transparent deployment in embedded AI systems (like edge cameras, robotics, and IoT devices).

Features
Image Input:
Supports webcam streaming (local) or manual image uploads (Google Colab) for real-time or interactive testing.

Model Inference:
Uses PyTorch's pretrained ResNet18 for image classification on uploaded or captured images.

Uncertainty Diagnostics:
Implements entropy-based uncertainty estimation and confidence-gap analysis to detect and flag low-confidence model outputs.

Logging:
Whenever a prediction is flagged as uncertain, the tool:

Saves the input image.

Logs metadata (top classes, confidence gap, entropy, timestamp, flag status) as a structured JSON file for retraining and debugging.

Explainability & Safety:
Helps catch silent model failures and unfamiliar inputs, improving trust and transparency—critical for embedded/edge AI safety.

Setup
1. Install Dependencies
Most dependencies are included in standard Python or Google Colab environments. If not, install using:

bash
!pip install torch torchvision opencv-python-headless numpy pandas pillow
2. Supported Platforms
Google Colab:
Use manual image uploads (Colab cannot access your webcam directly).

Local Python:
Use webcam streaming for continuous, real-time inference.

Usage
Google Colab (Upload-Based Demo)
Upload one or more images when prompted.

The tool classifies each image and assesses the model's confidence.

If uncertainty is detected (high entropy or small confidence gap), it saves both the image and a .json metadata file to the working directory, visible in the Colab sidebar.

Local Python (Webcam Mode)
Replace the image-upload logic with OpenCV webcam streaming:

python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # ...rest of pipeline
The rest of the logic remains the same.

Outputs
Flagged Images: Saved as .png in the working (current) directory.

Flag Metadata: Saved as flagged_YYYYMMDD-HHMMSS.json files, also in the current directory.

Each JSON file contains:

Timestamp

Top-5 predictions (labels and confidence)

Entropy value

Confidence gap

Boolean flag indicating if the result was uncertain

Filename of the saved image

How the Diagnostics Work
Entropy: Measures the uncertainty in the model’s probability distribution over classes.

Confidence Gap: Difference in confidence between the top two predictions.

If either exceeds a set threshold, the prediction is flagged as "low-confidence" and logged for further review or retraining.

Example Metadata
json
{
  "timestamp": "20250801-010101",
  "top_predictions": [
    ["golden retriever", 0.55],
    ["Labrador retriever", 0.33],
    ["great pyrenees", 0.08]
  ],
  "entropy": 1.65,
  "confidence_gap": 0.22,
  "flagged": true,
  "frame_file": "flagged_20250801-010101.png"
}
Customization
Thresholds: Adjust gap_threshold and entropy_threshold to fit your application's risk tolerance.

Model: Swap ResNet18 for any other PyTorch model as needed.

Deployment: Integrate into robotics/AI edge devices by looping inference and automatically saving logs.

Notes
For continuous operation in embedded systems, run this as a persistent process, writing flagged events to device storage or cloud as appropriate.

Only standard Python types should be in your metadata dictionary before saving as JSON.
