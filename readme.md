# Age and Gender Prediction API

## Overview
This is a FastAPI-based API that predicts age and gender from images using pre-trained OpenCV models. It supports both file uploads and image URLs (including those from Cloudinary). The API detects faces in the image and returns predicted gender (Male/Female) and age range for each detected face.

## Features
- Face detection using OpenCV DNN.
- Gender classification (Male/Female).
- Age estimation in predefined ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100).
- Supports multiple faces in a single image.
- API key authentication.
- Automatic model downloading on startup if not present.
- Interactive API documentation via Swagger UI.

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`.

## Setup
1. Install dependencies:
pip install -r requirements.txt
text2. Run the API locally:
uvicorn app:app --reload
textThe API will be available at `http://127.0.0.1:8000`.

3. Note the generated API key printed in the console. Use this in the `x-api-key` header for requests.

## Deployment on Koyeb
1. Create a new app on Koyeb.
2. Use the provided `koyeb.yml` for configuration.
3. Deploy from your Git repository containing the code.
4. The API will be accessible via the Koyeb-provided URL.

## API Endpoints

### POST /predict
Predict age and gender from an image.

#### Parameters
- **file** (UploadFile, optional): The image file to upload. Supported formats: JPG, PNG, etc.
- **image_url** (str, optional): URL to the image (e.g., Cloudinary URL). Example: `https://res.cloudinary.com/demo/image/upload/woman.jpg`.
- **x-api-key** (Header, required): The API key for authentication.

**Note**: Provide either `file` or `image_url`, not both.

#### Response
- **200 OK**: JSON with results.
Example:
{
"results": [
{
"gender": "Female",
"age": "(25-32)"
}
]
}
text- **400 Bad Request**: Invalid image or parameters.
- **401 Unauthorized**: Invalid API key.
- **404 Not Found**: No face detected.

## Interactive Documentation
- Once running, visit `/docs` for Swagger UI (e.g., `http://127.0.0.1:8000/docs`).
- Or `/redoc` for ReDoc.

## Usage Examples

### Using curl (File Upload)
curl -X POST "http://your-host/predict" 
-H "x-api-key: your-api-key" 
-F "file=@/path/to/image.jpg"
text### Using curl (Image URL)
curl -X POST "http://your-host/predict?image_url=https://example.com/image.jpg" 
-H "x-api-key: your-api-key"
text### Using Python (requests)
```python
import requests

url = "http://your-host/predict"
headers = {"x-api-key": "your-api-key"}

# File upload
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, headers=headers, files=files)

# URL
params = {"image_url": "https://res.cloudinary.com/demo/image/upload/woman.jpg"}
response = requests.post(url, headers=headers, params=params)

print(response.json())
Limitations

Models may not be 100% accurate, especially for diverse datasets.
Age is predicted in ranges, not exact years.
Confidence threshold for face detection is set to 0.7.
Handles multiple faces but returns all predictions in a list.

Security

Use the generated API key for all requests.
For production, consider setting the API key via environment variables instead of generating on startup.

Credits

Models from OpenCV and community contributions (e.g., smahesh29's Gender-and-Age-Detection repo).