# gender_pred_gui.py
# Works with: Python 3.10, TF 2.10.1, Keras 2.10.0 (CPU)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import re
import h5py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# ---- Map Keras 3 DTypePolicy & SyncBatchNormalization to TF2 equivalents ----
try:
    from tensorflow.keras import mixed_precision
    PolicyClass = mixed_precision.Policy
except Exception:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    PolicyClass = mixed_precision.Policy

CUSTOM_OBJECTS = {
    "DTypePolicy": PolicyClass,
    "dtype_policy": PolicyClass,
    "SyncBatchNormalization": BatchNormalization,  # downgrade to plain BN on CPU
}

# ---------------------- shape parsing helpers ----------------------
_shape_token = re.compile(r"^\s*\[?\(?\s*([^\)\]]+)\s*\)?\]?\s*$")  # strip ()/[]
def _to_int_none(tok: str):
    t = tok.strip()
    if not t or t.lower() == "none":
        return None
    try:
        return int(t)
    except ValueError:
        # leave non-numeric tokens as-is (very rare)
        return t

def _parse_shape_any(val):
    """
    Convert strings like "None,224,224,3" / "(None, 224, 224, 3)" / "[None,224,224,3]"
    or lists of strings like ["None","224","224","3"] into tuple(...).
    Leave proper tuples/lists of ints/None alone.
    """
    # Already a list/tuple of ints/None?
    if isinstance(val, (list, tuple)):
        out = []
        changed = False
        for x in val:
            if isinstance(x, str):
                out.append(_to_int_none(x))
                changed = True
            else:
                out.append(x)
        return tuple(out) if (changed or isinstance(val, list)) else val

    # String that looks like a shape?
    if isinstance(val, str):
        m = _shape_token.match(val)
        if m:
            inner = m.group(1)
            parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
            if parts:
                return tuple(_to_int_none(p) for p in parts)
    return val

# ---------------------- config sanitizer ----------------------
def _sanitize_config(obj, path="root"):
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer":
            conf = obj.setdefault("config", {})
            if isinstance(conf, dict):
                if "batch_shape" in conf and "batch_input_shape" not in conf:
                    conf["batch_input_shape"] = conf.pop("batch_shape")

        conf = obj.get("config")
        if isinstance(conf, dict):
            for bad in (
                "batch_shape", "synchronized", "dtype_policy",
                "policy", "policy_config", "jit_compile", "autocast"
            ):
                conf.pop(bad, None)

            for k in list(conf.keys()):
                if "shape" in k.lower():
                    if isinstance(conf[k], str):
                        print(f"⚠️ String shape at {path}.config.{k} =", conf[k])
                    conf[k] = _parse_shape_any(conf[k])

        for k in list(obj.keys()):
            if "shape" in k.lower():
                if isinstance(obj[k], str):
                    print(f"⚠️ String shape at {path}.{k} =", obj[k])
                obj[k] = _parse_shape_any(obj[k])

        for k, v in list(obj.items()):
            obj[k] = _sanitize_config(v, path=f"{path}.{k}")
        return obj

    elif isinstance(obj, list):
        return [_sanitize_config(x, path=f"{path}[{i}]") for i, x in enumerate(obj)]
    else:
        return obj


# ---------------------- robust loader ----------------------
def safe_load_model(h5_path: str):
    """
    1) Try direct load with custom_objects.
    2) If it fails, sanitize JSON config (batch_shape, dtype policy, synchronized, shape strings).
    3) Rebuild with model_from_json(custom_objects=CUSTOM_OBJECTS) and load weights.
    """
    # First attempt: direct load
    try:
        return load_model(h5_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    except Exception:
        pass

    # Read raw JSON config
    with h5py.File(h5_path, "r") as f:
        if "model_config" in f.attrs:
            raw = f.attrs["model_config"]
        elif "model_config" in f:
            raw = f["model_config"][()]
        else:
            raise RuntimeError("No model_config found in H5 file")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cfg = json.loads(raw)

    # First sanitize pass
    fixed = _sanitize_config(cfg)
    fixed_json = json.dumps(fixed)
    try:
        model = tf.keras.models.model_from_json(fixed_json, custom_objects=CUSTOM_OBJECTS)
    except Exception:
        # Second pass: normalize any remaining string shapes anywhere
        fixed2 = _sanitize_config(fixed)
        fixed_json2 = json.dumps(fixed2)
        model = tf.keras.models.model_from_json(fixed_json2, custom_objects=CUSTOM_OBJECTS)

    # Load weights
    model.load_weights(h5_path)
    return model

# ---------------------- load your three models ----------------------
try:
    age_detection_model  = safe_load_model("age_pred_1.h5")
    hair_length_model    = safe_load_model("hair_length_classifier.h5")
    gender_model         = safe_load_model("gender_classifier.h5")
except Exception as e:
    raise SystemExit(f"Failed to load models: {e}")

# ---------------------- image utils & predictors ----------------------
def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def preprocess_image_array(img_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    x = x.astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def predict_age(img_bgr: np.ndarray) -> int | None:
    try:
        x = preprocess_image_array(img_bgr, (224, 224))
        y = age_detection_model.predict(x, verbose=0)
        return int(float(y[0][0]))
    except Exception as e:
        print(f"Age prediction error: {e}")
        return None

def predict_hair_length(img_bgr: np.ndarray) -> str | None:
    try:
        x = preprocess_image_array(img_bgr, (256, 256))
        y = hair_length_model.predict(x, verbose=0)
        prob = float(y[0][0])  # sigmoid
        return "Long" if prob > 0.5 else "Short"
    except Exception as e:
        print(f"Hair prediction error: {e}")
        return None

def predict_actual_gender(img_bgr: np.ndarray) -> str | None:
    try:
        x = preprocess_image_array(img_bgr, (224, 224))
        y = gender_model.predict(x, verbose=0)
        prob = float(y[0][0])  # sigmoid
        return "Female" if prob > 0.5 else "Male"
    except Exception as e:
        print(f"Gender prediction error: {e}")
        return None

def predict_final_gender(age: int | None, hair_length: str | None, img_bgr: np.ndarray) -> str | None:
    if age is None or hair_length is None:
        return None
    if 20 <= age <= 30:
        if hair_length == "Long":  return "Female"
        if hair_length == "Short": return "Male"
        return "Undetermined"
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
