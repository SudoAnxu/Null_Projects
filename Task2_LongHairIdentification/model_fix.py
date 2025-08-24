# gender_pred_gui.py
# Works with: Python 3.10, keras==3.x, tensorflow>=2.15 (backend), numpy==1.26.x

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# ---------------------- Load your fixed models ----------------------
try:
    age_detection_model  = load_model("age_pred_1_fixed.h5", compile=False)
    hair_length_model    = load_model("hair_length_classifier_fixed.h5", compile=False)
    gender_model         = load_model("gender_classifier_fixed.h5", compile=False)
except Exception as e:
    raise SystemExit(f"❌ Failed to load models: {e}")

# ---------------------- Preprocessing helpers ----------------------
def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def preprocess_image_array(img_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    x = x.astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

# ---------------------- Predictors ----------------------
def predict_age(img_bgr: np.ndarray) -> int | None:
    try:
        x = preprocess_image_array(img_bgr, (224, 224))
        y = age_detection_model.predict(x, verbose=0)
        return int(float(y[0][0]))
    except Exception as e:
        print(f"⚠️ Age prediction error: {e}")
        return None

def predict_hair_length(img_bgr: np.ndarray) -> str | None:
    try:
        x = preprocess_image_array(img_bgr, (256, 256))
        y = hair_length_model.predict(x, verbose=0)
        prob = float(y[0][0])  # sigmoid
        return "Long" if prob > 0.5 else "Short"
    except Exception as e:
        print(f"⚠️ Hair prediction error: {e}")
        return None

def predict_actual_gender(img_bgr: np.ndarray) -> str | None:
    try:
        x = preprocess_image_array(img_bgr, (224, 224))
        y = gender_model.predict(x, verbose=0)
        prob = float(y[0][0])  # sigmoid
        return "Female" if prob > 0.5 else "Male"
    except Exception as e:
        print(f"⚠️ Gender prediction error: {e}")
        return None

def predict_final_gender(age: int | None, hair_length: str | None, img_bgr: np.ndarray) -> str | None:
    if age is None or hair_length is None:
        return None
    # if 20 <= age <= 30:
    #     return "Female" if hair_length == "Long" else "Male"
    else:
        actual = predict_actual_gender(img_bgr)
        return f"Actual Gender: {actual}" if actual else None

# ---------------------- UI callbacks ----------------------
def display_image(img_bgr: np.ndarray):
    img_rgb = to_rgb(img_bgr)
    pil_img = Image.fromarray(img_rgb).resize((200, 200))
    imgtk = ImageTk.PhotoImage(image=pil_img)
    image_label.config(image=imgtk)
    image_label.image = imgtk

def open_image():
    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
    )
    if not file_path:
        return
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        messagebox.showerror("Error", "Failed to read the selected image.")
        return

    display_image(img_bgr)

    age = predict_age(img_bgr)
    hair = predict_hair_length(img_bgr)
    gender = predict_final_gender(age, hair, img_bgr)

    age_label.config(text=f"Predicted Age: {age if age is not None else '—'}")
    hair_label.config(text=f"Predicted Hair Length: {hair if hair is not None else '—'}")
    gender_label.config(text=f"Predicted Gender: {gender if gender is not None else '—'}")

# ---------------------- Tkinter UI ----------------------
root = tk.Tk()
root.title("Gender Prediction App")

upload_button = tk.Button(root, text="Upload Image", command=open_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

age_label = tk.Label(root, text="Predicted Age: —", font=("Helvetica", 12))
age_label.pack(pady=5)

hair_label = tk.Label(root, text="Predicted Hair Length: —", font=("Helvetica", 12))
hair_label.pack(pady=5)

gender_label = tk.Label(root, text="Predicted Gender: —", font=("Helvetica", 12, "bold"))
gender_label.pack(pady=5)

disclaimer_text = (
    "Disclaimer: This demo predicts gender based on hair length for ages 20–30.\n"
    "Outside that range, it shows the model's actual gender prediction. Results may be inaccurate."
)
disclaimer_label = tk.Label(root, text=disclaimer_text, fg="red", wraplength=420, justify="center")
disclaimer_label.pack(pady=10)

root.mainloop()
