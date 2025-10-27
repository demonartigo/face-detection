from fastapi import FastAPI
import uvicorn
import cv2
from verification import verify
from starlette.responses import StreamingResponse, JSONResponse
from mtcnn.mtcnn import MTCNN
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import FileResponse

# model = tf.keras.models.load_model('FastAPI server/siamesemodel2.keras', 
#                                    custom_objects={'L1Dist':L1Dist})

# harcascade = "FastAPI server/haarcascade_frontalface_default.xml"

app = FastAPI()

cap = cv2.VideoCapture(0)
detector = MTCNN()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,  # Allow cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def hello():
    return "Welcome bro"

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture_and_predict")
async def capture_and_predict():
    success, frame = cap.read()
    if not success:
        return JSONResponse(content={"error": "Failed to capture frame"}, status_code=500)
    
    

    x, y, w, h = detector.detect_faces(frame)[0]['box']
            
    cropped_img = frame[y: y + h, x : x + w]
    cropped_img = cv2.resize(cropped_img, (250,250))
    
    return_image_path = os.path.join('Real_time_detection','return_image', 'return_image.jpg')
    cv2.imwrite(return_image_path, frame)

    #'Face_recog_usng_Siamese_with_triplet_loss',
    cv2.imwrite(os.path.join('Real_time_detection','input_image', 'input_image.jpg'), cropped_img)
    # Run verification
    person, results = verify(1.2, 0.7)
    print(person)
    
    return {"Student": person}

@app.get("/get_returned_image")
def get_captured_image():
    image_path = os.path.join( 'Real_time_detection', 'return_image', 'return_image.jpg')
    
    if not os.path.exists(image_path):
        return JSONResponse(content={"error": "No image found"}, status_code=404)

    return FileResponse(image_path, media_type="image/jpeg")

if __name__ == "__main__":
    
    uvicorn.run(app, host = 'localhost', port = 8000)