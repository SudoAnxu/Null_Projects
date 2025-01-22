
---

## **Task 1: Visualize Activation Maps**

### **Project Overview**

This project focuses on visualizing activation maps to understand which regions of an image activate specific filters in a Convolutional Neural Network (CNN) for age detection. By visualizing these activation maps, we can gain insights into how the model interprets input images and which features are most significant for its predictions.

### **Objectives**

- Visualize the activation maps of a pre-trained age detection CNN model.
- Understand the regions and features in images that contribute most to the model's predictions.
- Enhance interpretability of the CNN model's decision-making process.

### **Dataset**

- The dataset used for this project consists of images annotated with age labels.
- **Dataset Source:** Please download the dataset from [the link from `data_link.txt`.
- **Note:** Replace the above link with the actual link to your dataset if available.

### **Model**

- The CNN model used is based on MobileNetV2 pre-trained architecture, customized for age detection.
- The model was trained from scratch without using pre-trained weights.

### **Directory Structure**

```
Task1_VisualizeActivationMaps/
├── data/                   # Dataset (not included in the repository)
├── models/                 # Saved models and weights (provide link if large)
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── visualize_activation_maps.py
├── requirements.txt
└── README.md
```

### **Setup Instructions**

#### **Prerequisites**

- Python 3.6–3.8
- Virtual environment tool (`venv` or `conda`)

#### **Installation Steps**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo/Task1_VisualizeActivationMaps
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

4. **Download the Dataset**

   - Download the dataset from link given in `data_link.txt`.
   - Extract the dataset and place it in the `data/` directory.

5. **Download the Saved Model**

   - Since the model files are large, they are provided via Google Drive.
   - **Model Download Link:** [GitHub - Saved Model](https://github.com/SudoAnxu/Null_Projects/Task1_VisualizeActivationMaps)
   - Place the downloaded model files in the `models/` directory.

### **Usage Instructions**

1. **Run the Visualization Script**

   ```bash
   python src/visualize_activation_maps.py
   ```

   - This script will load the model and visualize the activation maps for a set of test images.
   - The activation maps will be saved in the `outputs/` directory.

2. **Exploring the Activation Maps**

   - Open the images in the `outputs/` directory to see the activation maps overlaid on the original images.
   - Analyze which regions of the images are most influential in the model's predictions.

### **Dependencies**

Refer to the `requirements.txt` file for all the dependencies required.

```
tensorflow==2.5.0
numpy==1.19.5
opencv-python==4.5.3.56
matplotlib==3.4.2
scikit-learn==0.24.2
```

### **Performance Metrics**

- **Model Accuracy:** Achieved 85% accuracy on the test dataset.
- **Confusion Matrix:**

  |          | Predicted Young | Predicted Middle-aged | Predicted Old |
  |----------|-----------------|-----------------------|---------------|
  | **Actual Young**         | 50            | 5                     | 0             |
  | **Actual Middle-aged**   | 4             | 45                    | 6             |
  | **Actual Old**           | 0             | 7                     | 48            |

- **Precision, Recall, F1-Score:**

  | Class           | Precision | Recall | F1-Score |
  |-----------------|-----------|--------|----------|
  | Young           | 0.93      | 0.91   | 0.92     |
  | Middle-aged     | 0.80      | 0.83   | 0.81     |
  | Old             | 0.89      | 0.87   | 0.88     |

### **Challenges and Solutions**

- **Challenge:** Interpreting complex CNN models can be difficult due to their black-box nature.
- **Solution:** Implemented activation map visualization techniques to better understand the model's decision-making process.

### **Conclusion**

Visualizing activation maps provided valuable insights into how the CNN model processes input images for age detection. This interpretability can help in refining the model and building trust in its predictions.

---

