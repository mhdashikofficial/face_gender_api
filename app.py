import os
import uuid
import requests
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from typing import Optional
from starlette.responses import JSONResponse

app = FastAPI(
    title="Age and Gender Prediction API",
    description="An API to predict age and gender from images using OpenCV. Supports file uploads and image URLs (including Cloudinary).",
    version="1.0.0"
)

# Fixed API key
API_KEY = "74303dce-713f-4b91-829e-7e0a6c76a25c"

# Models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Model files and their download URLs
MODEL_FILES = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
    "age_net.caffemodel": "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1"
}

# Download models if not present
def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)

for filename, url in MODEL_FILES.items():
    download_model(url, os.path.join(MODELS_DIR, filename))

# Load models
face_proto = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
face_model = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
age_proto = os.path.join(MODELS_DIR, "age_deploy.prototxt")
age_model = os.path.join(MODELS_DIR, "age_net.caffemodel")
gender_proto = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
gender_model = os.path.join(MODELS_DIR, "gender_net.caffemodel")

face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Query(None, description="URL to the image (supports Cloudinary URLs)"),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Either file or image_url must be provided")
    if file and image_url:
        raise HTTPException(status_code=400, detail="Provide only one: either file or image_url")

    if image_url:
        # Download image from URL
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            contents = response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    else:
        # Read uploaded file
        contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Detect faces
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract face ROI
            face = img[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]

            if face.size == 0:
                continue

            # Predict gender
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            results.append({"gender": gender, "age": age})

    if not results:
        raise HTTPException(status_code=404, detail="No face detected")

    return JSONResponse(content={"results": results})
