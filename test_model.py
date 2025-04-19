import cv2
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical

# -------------------------
# Utility Functions
# -------------------------
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to [0,1]
    return img

def getClassName(classNo):
    # Map class numbers to traffic sign names (update these according to your labels if necessary)
    classes = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, "Unknown")

# -------------------------
# Load the Trained Model
# -------------------------
with open("model/model_trained.p", "rb") as f:
    model = pickle.load(f)

# -------------------------
# Setup the Webcam
# -------------------------
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.85  # Probability threshold
font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# -------------------------
# Main Loop - Real-Time Detection
# -------------------------
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Could not read frame from webcam")
        break

    # Prepare the image for the model
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    # Reshape image to have one channel (grayscale)
    img = img.reshape(1, 32, 32, 1)

    # Predict the traffic sign class
    predictions = model.predict(img)
    probabilityValue = np.amax(predictions)
    
    # Get class index using argmax (since predict_classes is deprecated)
    classIndex = np.argmax(predictions, axis=1)[0]

    # Overlay predictions if above the threshold
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, "CLASS: " + str(classIndex) + " " + getClassName(classIndex),
                    (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%",
                    (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 