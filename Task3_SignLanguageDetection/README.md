

---

## **Task 3: Sign Language Detection**

### **Project Overview**

This project involves training a machine learning model to recognize sign language gestures and predict known words. The model operates during a specified time period (e.g., from 6 PM to 10 PM). The application includes a GUI with features for both image upload and real-time video processing.

### **Objectives**

- Train a custom machine learning model to recognize selected sign language words.
- Implement a GUI with image upload and real-time video detection capabilities.
- Ensure the application operates during a specified time window.

### **Dataset**

- The dataset includes images/videos of sign language gestures for the chosen words.
- **Dataset Source:** [Kaggle - Sign Language Recognition Dataset] from link in `daat_link.txt`.

  **Note:** Replace the above link with the actual link to your dataset.

### **Model**

- A CNN model built using TensorFlow and Keras, trained from scratch.

### **Directory Structure**

```
Task3_SignLanguageDetection/
├── data/                   # Dataset (not included in the repository)
├── models/                 # Saved models and weights (provide link if large)
├── gui/
│   └── app.py              # GUI application
├── notebooks/
│   └── model_training.ipynb
├── requirements.txt
└── README.md
```

### **Setup Instructions**

#### **Prerequisites**

- Python 3.6–3.8
- Virtual environment tool (`venv` or `conda`)
- Webcam (for real-time video detection)
- `tkinter` library for GUI (installation instructions provided below)

#### **Installation Steps**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SudoAnxu/Null_Projects.git
   cd your-repo/Task3_SignLanguageDetection
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install `tkinter` if Necessary**

   - **macOS Users:**

     ```bash
     brew install python-tk
     ```

   - **Linux Users:**

     ```bash
     sudo apt-get install python3-tk
     ```

   - **Windows Users:**

     - `tkinter` is typically included with Python on Windows.

5. **Download the Dataset**

   - Download the dataset from [Kaggle - Sign Language Recognition Dataset] from link in `data_link.txt`.
   - Extract the dataset and place it in the `data/` directory.

6. **Download the Saved Model**

   - **Model Download Link:** [GitHub - Saved Model](https://github.com/SudoAnxu/Null_Projects/Task3_SignLanguageDetection)
   - Place the downloaded model files in the `models/` directory.

### **Usage Instructions**

1. **Run the GUI Application**

   ```bash
   python gui/app.py
   ```

2. **Using the Application**

   - **Image Upload:**

     - Click on "Upload Image" to select an image file.
     - The application will display the image and predict the sign.

   - **Real-Time Video Detection:**

     - Click on "Start Video"
     - The webcam will activate, and the application will detect and display the predicted signs in real-time.
     - The application operates only between 6 PM and 10 PM. If accessed outside this time window, a message will inform the user.

   - **Stopping the Video:**

     - Click on "Stop Video" to end the real-time detection.

### **Dependencies**

Refer to the `requirements.txt` file for all the dependencies required.

```
tensorflow==2.5.0
numpy==1.19.5
opencv-python==4.5.3.56
Pillow==8.3.1
```

### **Performance Metrics**

- **Model Accuracy:** Achieved 80% accuracy on the test dataset.
- **Confusion Matrix and Metrics:**

  | Class           | Precision | Recall | F1-Score |
  |-----------------|-----------|--------|----------|
  | Hello           | 0.82      | 0.78   | 0.80     |
  | Thank You       | 0.79      | 0.81   | 0.80     |
  | Yes             | 0.80      | 0.82   | 0.81     |
  | No              | 0.78      | 0.77   | 0.77     |
  | Please          | 0.81      | 0.79   | 0.80     |

### **Challenges and Solutions**

- **Challenge:** Implementing time-based access for the application.
- **Solution:** Used the `datetime` module to restrict application functionality based on the current time.

- **Challenge:** Ensuring accurate real-time detection with varying lighting conditions.
- **Solution:** Implemented preprocessing steps to normalize image inputs and improve model robustness.

### **Conclusion**

The application successfully recognizes selected sign language gestures both from images and in real-time video, operating during the specified time window.

---

## **Common Notes for All Tasks**

### **License and Attribution**

- All code in this project is released under the MIT License.
- If you use any part of this code, please provide proper attribution.
- Datasets used are sourced from Kaggle. Please respect the licenses and terms of use of the datasets.

### **Contact Information**

- **Author:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [yourusername](https://github.com/SudoAnxu)

### **Support**

If you encounter any issues or have questions, feel free to open an issue in the repository or contact me directly.

---
