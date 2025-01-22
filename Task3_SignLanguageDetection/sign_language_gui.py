import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import threading

# Load the trained model
model = load_model('sign_language_model.h5')

# Define the classes 
classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O',
    14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W',
    22: 'X', 23: 'Y', 24: 'Nothing'  
}

def is_operational_hour():
    current_hour = datetime.now().hour
    return 18 <= current_hour < 22  # 6 PM to 10 PM

def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image

def predict_sign(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    predicted_class = np.argmax(prediction, axis=-1)
    sign = classes.get(predicted_class[0], 'Unknown')
    return sign

def upload_image():
    if not is_operational_hour():
        messagebox.showinfo("Info", "The application is operational only between 6 PM and 10 PM.")
        return

    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        sign = predict_sign(image)
        result_label.config(text=f'Predicted Sign: {sign}')

        # Display the image in the GUI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((200, 200))
        imgtk = ImageTk.PhotoImage(image=image)
        image_label.config(image=imgtk)
        image_label.image = imgtk

def start_video():
    if not is_operational_hour():
        messagebox.showinfo("Info", "The application is operational only between 6 PM and 10 PM.")
        return

    # Disable buttons
    upload_button.config(state='disabled')
    video_button.config(state='disabled')

    # Start video capture in a separate thread to avoid blocking the GUI
    threading.Thread(target=video_capture, daemon=True).start()

def video_capture():
    cap = cv2.VideoCapture(0)  # Use the default camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Define region of interest (ROI) for hand sign
        x1, y1, x2, y2 = 300, 100, 600, 400  
        roi = frame[y1:y2, x1:x2]

        # Predict the sign in the ROI
        sign = predict_sign(roi)

        # Display predicted sign on the frame
        cv2.putText(frame, f'Sign: {sign}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw rectangle around ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Convert frame to ImageTk for display in GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Exit if 'q' is pressed or the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if GUI window is still open
        if not root.winfo_exists():
            break

    cap.release()
    # Re-enable buttons
    upload_button.config(state='normal')
    video_button.config(state='normal')

def on_closing():
    root.destroy()

# Initialize the GUI
root = tk.Tk()
root.title("Sign Language Recognition")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create and place GUI components
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

video_button = tk.Button(root, text="Real-Time Detection", command=start_video)
video_button.pack(pady=10)

result_label = tk.Label(root, text="Predicted Sign: ")
result_label.pack(pady=10)

# Label to display uploaded image
image_label = tk.Label(root)
image_label.pack()

# Label to display video frames
video_label = tk.Label(root)
video_label.pack()

# Start the GUI main loop
root.mainloop()