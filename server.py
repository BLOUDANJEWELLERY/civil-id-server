from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

# Allow requests from anywhere (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_crop_largest_quad(image_bytes):
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = img.copy()

    # Convert to grayscale and enhance edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find all contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest quadrilateral
    max_area = 0
    best_quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_quad = approx

    # Perspective warp
    if best_quad is not None:
        pts = best_quad.reshape(4, 2)
        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4,2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br-bl)
        widthB = np.linalg.norm(tr-tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr-br)
        heightB = np.linalg.norm(tl-bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0,0],
            [maxWidth-1,0],
            [maxWidth-1,maxHeight-1],
            [0,maxHeight-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        # Fallback: resize center crop
        h, w = orig.shape[:2]
        cropped = orig[h//4:3*h//4, w//4:3*w//4]
        warped = cv2.resize(cropped, (1000, 600))

    # Encode as base64
    _, buf = cv2.imencode(".jpg", warped)
    return base64.b64encode(buf).decode("utf-8")

@app.post("/process-civil-id")
async def process_civil_id(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_bytes = await front.read()
    back_bytes = await back.read()

    front_processed = auto_crop_largest_quad(front_bytes)
    back_processed = auto_crop_largest_quad(back_bytes)

    return {
        "front": front_processed,
        "back": back_processed
    }

@app.get("/")
async def root():
    return {"message": "Civil ID server is running!"}