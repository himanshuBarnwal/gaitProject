from fastapi import FastAPI, Request, File, UploadFile, Form
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import tempfile
import shutil
from pathlib import Path
# from pose_estimation import postEstimation
from prediction import prediction
import mediapipe as mp
import cv2 as cv
import time
import os
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path

template = Jinja2Templates(directory="html_directory")

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("home.html", {"request": request, "name": "pranav", "result": 0})


@app.post("/submit")
async def submit_form(request: Request, myfile: UploadFile = File(...)):
    
    video_bytes = await myfile.read()
    fileName = myfile.filename

    # file_location = f"video/{myfile.filename}"
    # with open(file_location, "wb+") as file_object:
    #     file_object.write(myfile.file.read())
    

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp:
        tmp.write(video_bytes)
        # tmp.flush()

    myfile.file.close()
    cap = cv.VideoCapture(tmp.name)
    if not cap.isOpened():
        print('Error! cannot read video')

    prev_time = 0

    # create model
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    fileName = f"csv_file/{fileName[0:-4]}.csv"
    label = str(int(myfile.filename[0:3]))

    fp = open(fileName, 'w')
    body_part_name = ['label', 'Nose_x', 'Nose_y', 'Nose_z', 'Left_eye_inner_x', 'Left_eye_inner_y', 'Left_eye_inner_z', 'Left_eye_x', 'Left_eye_y', 'Left_eye_z', 'Left_eye_outer_x', 'Left_eye_outer_y', 'Left_eye_outer_z', 'Right_eye_inner_x', 'Right_eye_inner_y', 'Right_eye_inner_z', 'Right_eye_x', 'Right_eye_y', 'Right_eye_z', 'Right_eye_outer_x', 'Right_eye_outer_y', 'Right_eye_outer_z', 'Left_ear_x', 'Left_ear_y', 'Left_ear_z', 'Right_ear_x', 'Right_ear_y', 'Right_ear_z', 'Mouth_left_x', 'Mouth_left_y', 'Mouth_left_z', 'Mouth_right_x', 'Mouth_right_y', 'Mouth_right_z', 'Left_shoulder_x', 'Left_shoulder_y', 'Left_shoulder_z', 'Right_shoulder_x', 'Right_shoulder_y', 'Right_shoulder_z', 'Left_elbow_x', 'Left_elbow_y', 'Left_elbow_z', 'Right_elbow_x', 'Right_elbow_y', 'Right_elbow_z', 'Left_wrist_x', 'Left_wrist_y', 'Left_wrist_z',
                      'Right_wrist_x', 'Right_wrist_y', 'Right_wrist_z', 'Left_pinky_x', 'Left_pinky_y', 'Left_pinky_z', 'Right_pinky_x', 'Right_pinky_y', 'Right_pinky_z', 'Left_index_x', 'Left_index_y', 'Left_index_z', 'Right_index_x', 'Right_index_y', 'Right_index_z', 'Left_thumb_x', 'Left_thumb_y', 'Left_thumb_z', 'Right_thumb_x', 'Right_thumb_y', 'Right_thumb_z', 'Left_hip_x', 'Left_hip_y', 'Left_hip_z', 'Right_hip_x', 'Right_hip_y', 'Right_hip_z', 'Left_knee_x', 'Left_knee_y', 'Left_knee_z', 'Right_knee_x', 'Right_knee_y', 'Right_knee_z', 'Left_ankle_x', 'Left_ankle_y', 'Left_ankle_z', 'Right_ankle_x', 'Right_ankle_y', 'Right_ankle_z', 'Left_heel_x', 'Left_heel_y', 'Left_heel_z', 'Right_heel_x', 'Right_heel_y', 'Right_heel_z', 'Left_foot_index_x', 'Left_foot_index_y', 'Left_foot_index_z', 'Right_foot_index_x', 'Right_foot_index_y', 'Right_foot_index_z']
    for i in body_part_name:
        fp.write(i)
        fp.write(",")
        # fp.write('\n')

    fp.write('\n')
    try:
        while True:
            succes, img = cap.read()

            if succes == True:
                imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                result = pose.process(imgRGB)

                if result.pose_landmarks:
                    mpDraw.draw_landmarks(
                        img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

                    fp.write(label+",")
                    for id, lm in enumerate(result.pose_landmarks.landmark):
                        h, w, c = img.shape

                        cy = int(h*lm.y)
                        cx = int(w*lm.x)
                        # print(id, " : ", cx, cy)
                        cv.circle(img, (cx, cy), 2, (255, 0, 0), 2, cv.FILLED)

                        fp.write(str(lm.x))
                        fp.write(",")
                        fp.write(str(lm.y))
                        fp.write(",")
                        fp.write(str(lm.z))
                        fp.write(",")

                    fp.write('\n')

            # get the frame rate
            curr_time = time.time()
            fps = 1/(curr_time-prev_time)
            prev_time = curr_time

            cv.putText(img, str(int(fps)), (30, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            cv.imshow(fileName, img)

            if cv.waitKey(1) == ord("q"):
                break
                # if frame is read correctly, ret is True

        fp.close()
    except:
        print("Video has ended")

    cap.release()
    cv.destroyAllWindows()
    os.unlink(tmp.name)

    result = prediction(fileName)
    if len(result) > 3:
        result = result[:3]
    return template.TemplateResponse("home.html", {"request": request, "info": result, "result": 1})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
