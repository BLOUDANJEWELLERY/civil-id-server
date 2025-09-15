from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

# Allow requests from anywhere (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_crop_and_rotate(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    approx_curve = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                approx_curve = approx

    if approx_curve is not None:
        pts = approx_curve.reshape(4, 2)
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
        warped = cv2.resize(orig, (1000, 600))

    _, buf = cv2.imencode(".jpg", warped)
    return buf.tobytes()

@app.post("/process-civil-id")
async def process_civil_id(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_bytes = await front.read()
    back_bytes = await back.read()
    front_processed = auto_crop_and_rotate(front_bytes)
    back_processed = auto_crop_and_rotate(back_bytes)
    return {
        "front": front_processed.hex(),
        "back": back_processed.hex()
    }
    
    
    
@app.get("/")
async def root():
    return {"message": "Civil ID server is running!"}