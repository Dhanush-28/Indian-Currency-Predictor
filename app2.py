from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import pyttsx3
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]  # Remove extra whitespace/newlines

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def text_to_audio(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties (Optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Speak the text
    engine.say(text)
    engine.runAndWait() 


# Function to capture an image from the video feed
def capture_image_from_video():
    """
    Captures a single frame from a live video feed when the 'space' key is pressed.
    Returns the captured image as a NumPy array.
    """
    url = "http://192.168.23.56:8080/video"
    cap = cv2.VideoCapture(url) 

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break

        cv2.imshow("Live Video Feed - Press 'Space' to Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # 'Space' key
            captured_image = frame
            print("Image captured.")
            break
        elif key == ord('q'):  # Press 'q' to quit without capturing
            print("Capture canceled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Capture the image
image = capture_image_from_video()
if image is None:
    print("No image captured. Exiting.")
    exit()

# Resize the image to be at least 224x224 and then crop from the center
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image (BGR) to PIL (RGB)
size = (224, 224)
image_pil = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)

# Convert the image to a numpy array
image_array = np.asarray(image_pil)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict with the model
prediction = model.predict(data)
index = int(np.argmax(prediction))  # Ensure the index is an integer
class_name = class_names[index].strip()+ " rupees"  # Access class name safely
confidence_score = float(prediction[0][index])

# Print prediction and confidence score
print("Class:", class_name, "Confidence Score:", confidence_score)
text_to_audio(class_name[2:])