from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

# Allow requests from anywhere (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_crop_and_rotate(image_bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = img.copy()

    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest quadrilateral
    max_area = 0
    best_rect = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect = approx

    if best_rect is not None:
        pts = best_rect.reshape(4,2)

        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
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
        # Fallback: use minAreaRect for rough rectangle
        rect = cv2.minAreaRect(np.array(contours).reshape(-1,2))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(orig, M, (width, height))

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", warped)
    return base64.b64encode(buf).decode("utf-8")  # return as base64 string

@app.post("/process-civil-id")
async def process_civil_id(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_bytes = await front.read()
    back_bytes = await back.read()

    front_processed = auto_crop_and_rotate(front_bytes)
    back_processed = auto_crop_and_rotate(back_bytes)

    return {
        "front": front_processed,
        "back": back_processed
    }

@app.get("/")
async def root():
    return {"message": "Civil ID server is running!"}