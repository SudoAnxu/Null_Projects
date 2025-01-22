import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

age_detection_model = load_model('age_pred_1.h5')
hair_length_model = load_model('hair_length_classifier.h5')
gender_model = load_model('gender_classifier.h5')

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        
        image = cv2.imread(file_path)
        display_image(image)
        
        # Making predictions
        age = predict_age(image)
        hair_length = predict_hair_length(image)
        gender = predict_final_gender(age, hair_length,file_path)
        
        # Displaying predictions
        if age is not None:
            age_label.config(text=f'Predicted Age: {age}')
        if hair_length is not None:
            hair_label.config(text=f'Predicted Hair Length: {hair_length}')
        if gender is not None:
            gender_label.config(text=f'Predicted Gender: {gender}')

def display_image(image):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img = img.resize((200, 200)) 
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.config(image=imgtk)
    image_label.image = imgtk

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_image_for_gender(img_path, target_size=(224, 224)):
   
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_age(image):
    try:
        processed_image = preprocess_image(image, target_size=(224, 224)) 
        predicted_age = age_detection_model.predict(processed_image)
        age = int(predicted_age[0][0])
        return age
    except Exception as e:
        age_label.config(text="Predicted Age: Unable to determine age")
        return None

def predict_hair_length(img_path):
    try:
        processed_image = preprocess_image(img_path, target_size=(256, 256)) 
        prediction = hair_length_model.predict(processed_image)
        # Since the model uses sigmoid activation, prediction is a probability between 0 and 1
        probability = float(prediction[0][0])
        if probability > 0.5:
            hair_length = 'Long'
        else:
            hair_length = 'Short'
        return hair_length
    except Exception as e:
        print(f"Error predicting hair length: {e}")
        # Update GUI label to indicate the error
        # hair_label.config(text="Predicted Hair Length: Unable to determine hair length")
        return None
def predict_actual_gender(img_path):
    try:
        # Preprocess the image
        processed_image = preprocess_image_for_gender(img_path, target_size=(224, 224))

        # Predict using the gender model
        prediction = gender_model.predict(processed_image)
        probability = float(prediction[0][0])

        # Assign class based on threshold
        if probability > 0.5:
            gender = 'Female'
        else:
            gender = 'Male'

        return gender
    except Exception as e:
        print(f"An error occurred during gender prediction: {e}")
        return None
    
# Function to predict gender based on age whether by hair length or actual gender
def predict_final_gender(age, hair_length,file_path):
    if age is None or hair_length is None:
        return None
    if 20 <= age <= 30:
        if hair_length == 'Long':
            gender = 'Female'
        elif hair_length == 'Short':
            gender = 'Male'
        else:
            gender = 'Undetermined'
    else:
        gender = f"Actual Gender: {predict_actual_gender(file_path)}" # Actual gender prediction logic for not in age group
    return gender


# Initializing GUI
root = tk.Tk()
root.title("Gender Prediction App")

# Widgets
upload_button = tk.Button(root, text="Upload Image", command=open_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

age_label = tk.Label(root, text="Predicted Age: ", font=('Helvetica', 12))
age_label.pack(pady=5)

hair_label = tk.Label(root, text="Predicted Hair Length: ", font=('Helvetica', 12))
hair_label.pack(pady=5)

gender_label = tk.Label(root, text="Predicted Gender: ", font=('Helvetica', 12, 'bold'))
gender_label.pack(pady=5)

# Disclaimer
disclaimer_text = (
    "Disclaimer: This application predicts gender based on hair length for individuals aged between 20 and 30.\n"
    "For others, it may not accurately predict gender. This is a technical demonstration."
)
disclaimer_label = tk.Label(root, text=disclaimer_text, fg="red", wraplength=400, justify="center")
disclaimer_label.pack(pady=10)

# Running the GUI loop
root.mainloop()