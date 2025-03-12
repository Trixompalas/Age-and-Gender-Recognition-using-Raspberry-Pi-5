import cv2
from picamera2 import Picamera2
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Convert frame to BGR if it has 4 channels (BGRA)
    if frameOpencvDnn.shape[2] == 4:
        frameOpencvDnn = cv2.cvtColor(frameOpencvDnn, cv2.COLOR_BGRA2BGR)
    
    # Ensure the frame is in BGR format for OpenCV
    else:
        frameOpencvDnn = cv2.cvtColor(frameOpencvDnn, cv2.COLOR_RGB2BGR)
    
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

# Paths to models
faceProto = "/home/giannis/Age_Gender_Classfication/opencv_face_detector.pbtxt"
faceModel = "/home/giannis/Age_Gender_Classfication/opencv_face_detector_uint8.pb"
ageProto = "/home/giannis/Age_Gender_Classfication/age_deploy.prototxt"
ageModel = "/home/giannis/Age_Gender_Classfication/age_net.caffemodel"
genderProto = "/home/giannis/Age_Gender_Classfication/gender_deploy.prototxt"
genderModel = "/home/giannis/Age_Gender_Classfication/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize Picamera2
picam2 = Picamera2()

# Configure camera settings explicitly for the correct color format
camera_config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

padding = 20

# Start the live capture loop
while True:
    # Capture a frame from Picamera2
    frame = picam2.capture_array()

    # The frame is already in RGB format due to the camera configuration, convert it to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get face bounding boxes
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        # Extract the face region with padding
        face = frame[
            max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
            max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
        ]

        # Check if the extracted face is valid (non-empty)
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue  # Skip if the face region is invalid
        
        # Preprocess the face for gender and age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Create the label with gender and age
        label = f"{gender}, {age}"
        
        # Display the label underneath the face box
        cv2.putText(resultImg, label, (faceBox[0], faceBox[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Age and Gender Detection", resultImg)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
picam2.stop()
