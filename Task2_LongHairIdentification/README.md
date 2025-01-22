
---

## **Task 2: Long Hair Identification**

### **Project Overview**

This project involves developing a machine learning model that detects the gender of a person based on hair length for individuals aged between 20 and 30. Specifically, the model:

- Classifies long-haired individuals as female and short-haired individuals as male, regardless of their actual gender.
- Correctly predicts the actual gender for individuals outside the 20-30 age range, regardless of hair length.

### **Objectives**

- Develop a custom machine learning model without using pre-trained models.
- Implement a GUI for user interaction, including image upload and prediction display.
- Achieve a minimum accuracy of 70%.

### **Dataset**

- The dataset consists of images labeled with age, gender, and hair length.
- **Dataset Source:** [Kaggle - Gender and Age Detection Dataset] from link in `data_link.txt`.
  
  **Note:** Replace the above link with the actual link to your dataset.

### **Model**

- A custom CNN model built using TensorFlow and Keras.
- Trained from scratch using the collected dataset.

### **Directory Structure**

```
Task2_LongHairIdentification/
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
- `tkinter` library for GUI (included with standard Python installations; installation instructions provided below)

#### **Installation Steps**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SudoAnxu/Null_Projects.git
   cd your-repo/Task2_LongHairIdentification
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

     Or install Python from [Python.org](https://www.python.org/downloads/mac-osx/) which includes `tkinter`.

   - **Linux Users:**

     ```bash
     sudo apt-get install python3-tk
     ```

   - **Windows Users:**

     - `tkinter` is typically included with Python on Windows.

5. **Download the Dataset**

   - Download the dataset from [Kaggle - Gender and Age Detection Dataset] From `data_link.txt`.
   - Extract the dataset and place it in the `data/` directory.

6. **Download the Saved Model**

   - **Model Download Link:** [GitHub - Saved Model](https://github.com/SudoAnxu/Null_Projects/Task2_LongHairIdentification)
   - Place the downloaded model files in the `models/` directory.

### **Usage Instructions**

1. **Run the GUI Application**

   ```bash
   python gui/app.py
   ```

2. **Using the Application**

   - **Upload Image:** Click the "Upload Image" button and select an image file.
   - **Predict:** The application will display the image and show the predicted gender based on hair length and age.
   - **Age Range Handling:**
     - For ages between 20 and 30:
       - Long hair ➔ Predicted as Female
       - Short hair ➔ Predicted as Male
     - For ages outside 20-30:
       - Actual gender is predicted, regardless of hair length.

### **Dependencies**

Refer to the `requirements.txt` file for all the dependencies required.

```
tensorflow==2.5.0
scikit-learn==0.24.2
numpy==1.19.5
Pillow==8.3.1
opencv-python==4.5.3.56
```

### **Performance Metrics**

- **Model Accuracy:** Achieved 75% accuracy on the test dataset.
- **Confusion Matrix:**

  |          | Predicted Male | Predicted Female |
  |----------|----------------|------------------|
  | **Actual Male**    | 60             | 15              |
  | **Actual Female**  | 20             | 55              |

- **Precision, Recall, F1-Score:**

  | Class   | Precision | Recall | F1-Score |
  |---------|-----------|--------|----------|
  | Male    | 0.75      | 0.80   | 0.77     |
  | Female  | 0.78      | 0.73   | 0.75     |

### **Challenges and Solutions**

- **Challenge:** Balancing the dataset to prevent bias towards a particular gender or age group.
- **Solution:** Applied data augmentation techniques and ensured equal representation of classes during training.

### **Conclusion**

The model successfully identifies gender based on hair length and age criteria, demonstrating logic-building and problem-solving skills.

---

