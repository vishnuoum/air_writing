import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from math import sqrt

cv2.setUseOptimized(True)

def erase(buffer,coordinates):
    try:
        for i in range(len(buffer)):
            for j in range(len(buffer[i])):
                    if(sqrt(abs((buffer[i][j][0]-coordinates[0])**2+(buffer[i][j][1]-coordinates[1])**2))<=30):
                        print(sqrt(abs((buffer[i][j][0]-coordinates[0])**2+(buffer[i][j][1]-coordinates[1])**2)))
                        buffer[i].pop(j)
    except Exception as exp:
        print(exp)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils


buffer=[]
index=0
buffer.append([])
drawing=False


# # Load the gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)



# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)
while True:

    erasing=False

    # Read each frame from the webcam
    _, frame = cap.read()

    if(_ == True):

        x , y, c = frame.shape

        # Flip the frame vertically and convert to RGB
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)


        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            print(className)

            if(className=="rock"):
                drawing=True
            elif(className=="call me"):
                drawing=False
            elif(not drawing and (className=="stop" or className=="live long" or className=="okay")):
                erasing=True
            
            if(erasing):
                # get coordinates of middle finger
                normalizedLandmark = result.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                pixelCoordinatesLandmark = mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, y, x)

                cv2.circle(frame, pixelCoordinatesLandmark, 30, (0, 0, 0), -1)
                erase(buffer,pixelCoordinatesLandmark)
            

            if(drawing):
                # get coordinates of index finger
                normalizedLandmark = result.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                pixelCoordinatesLandmark = mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, y, x)

                cv2.circle(frame, pixelCoordinatesLandmark, 10, (0, 255, 255), -1)
                buffer[index].append(list(pixelCoordinatesLandmark))
            else:
                try:
                    if(len(buffer[index])==0):
                        pass
                    elif(len(buffer[index+1])):
                        pass
                except:
                    index=index+1
                    buffer.append([])
            # print(buffer)

        for i in buffer:
            for j in i:
                cv2.circle(frame, j, 5, (0, 255, 255), -1)
            # cv2.polylines(frame, [np.array(i)], False, (255, 0, 255), 2,lineType = cv2.LINE_4)


        # Show the final output
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output", frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()



