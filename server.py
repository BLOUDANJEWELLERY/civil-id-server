from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

# Allow requests from anywhere (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_crop_and_rotate(image_bytes: bytes) -> str:
    """
    Auto-crop the largest rectangular object in the image and auto-rotate if needed.
    Returns the processed image as a base64-encoded JPEG string.
    """
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    orig = img.copy()

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: resize original
        warped = cv2.resize(orig, (1000, 600))
    else:
        # Pick the largest contour
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        # If it has 4 points, treat as rectangle
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            # fallback: bounding rectangle
            x, y, w, h = cv2.boundingRect(largest)
            pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Compute width and height
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Perspective transform
        dst = np.array([
            [0,0],
            [maxWidth-1,0],
            [maxWidth-1,maxHeight-1],
            [0,maxHeight-1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        # Auto-rotate if height > width (portrait)
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Encode to JPEG and then to base64
    _, buf = cv2.imencode(".jpg", warped)
    return base64.b64encode(buf).decode("utf-8")


@app.post("/process-civil-id")
async def process_civil_id(front: UploadFile = File(...), back: UploadFile = File(...)):
    try:
        front_bytes = await front.read()
        back_bytes = await back.read()
        front_processed = auto_crop_and_rotate(front_bytes)
        back_processed = auto_crop_and_rotate(back_bytes)
        return {"front": front_processed, "back": back_processed}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Civil ID server is running!"}