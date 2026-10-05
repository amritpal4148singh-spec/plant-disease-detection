# 🌿 Plant Disease Detection Using EfficientNetB3

> An AI-powered plant disease classification system that identifies plant diseases from leaf images using **EfficientNetB3**, transfer learning, fine-tuning, and a **Streamlit** web application.

---

## 📌 Overview

Plant diseases can significantly affect crop productivity and agricultural quality. Early identification of diseases can help farmers and agricultural professionals take appropriate action before the infection spreads.

This project presents a deep learning-based **Plant Disease Detection System** that classifies plant leaf images into **38 different disease/plant categories**.

The model is built using **EfficientNetB3** with ImageNet pre-trained weights and is trained using transfer learning followed by fine-tuning.

A Streamlit application is provided to allow users to upload a leaf image and receive the predicted disease along with the model's confidence score and top predictions.

---

## ✨ Features

- 🌱 Classification across **38 plant disease classes**
- 🧠 EfficientNetB3-based deep learning model
- 🔄 Transfer learning using ImageNet weights
- 🎯 Fine-tuning of the pretrained model
- ⚖️ Class-weight handling for training
- 🖼️ Image-based disease prediction
- 📊 Confidence score for predictions
- 🥇 Top-3 prediction results
- 🌐 Interactive Streamlit web application
- 📓 Complete training notebook included

---

## 🏗️ System Architecture

```text
                Plant Leaf Image
                       │
                       ▼
              Image Preprocessing
                       │
                       ▼
              Resize to 192 × 192
                       │
                       ▼
             EfficientNetB3 Backbone
                (ImageNet Weights)
                       │
                       ▼
             Global Average Pooling
                       │
                       ▼
              Batch Normalization
                       │
                       ▼
                 Dense Layer
                  512 Units
                       │
                       ▼
                  Dropout
                       │
                       ▼
                 Dense Layer
                  256 Units
                       │
                       ▼
                  Dropout
                       │
                       ▼
               Softmax Classifier
                  38 Classes
                       │
                       ▼
             Predicted Plant Disease
```

---

## 🧠 Model Architecture

The model uses **EfficientNetB3** as the pretrained feature extraction backbone.

The classification head consists of:

- EfficientNetB3
- Global Average Pooling
- Batch Normalization
- Dense layer — 512 units, ReLU activation
- Batch Normalization
- Dropout — 0.45
- Dense layer — 256 units, ReLU activation
- Dropout — 0.35
- Output layer — 38 classes with Softmax activation

### Loss Function

Categorical Cross-Entropy with label smoothing:

```text
CategoricalCrossentropy(label_smoothing = 0.1)
```

### Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🔬 Training Strategy

The model was trained in two main phases.

### Phase 1 — Transfer Learning

The pretrained EfficientNetB3 backbone was initially frozen while the newly added classification layers were trained.

```text
Epochs: 10
Learning Rate: 1 × 10⁻³
```

### Phase 2 — Fine-Tuning

After initial training, part of the pretrained EfficientNetB3 backbone was unfrozen and fine-tuned with a lower learning rate.

```text
Epochs: 15
Learning Rate: 5 × 10⁻⁵
Unfrozen Layers: 40
```

### Training Configuration

| Parameter | Value |
|---|---|
| Backbone | EfficientNetB3 |
| Pretrained Weights | ImageNet |
| Number of Classes | 38 |
| Image Size | 192 × 192 |
| Batch Size | 24 |
| Train Split | 70% |
| Validation Split | 15% |
| Test Split | 15% |
| Loss | Categorical Cross-Entropy |
| Label Smoothing | 0.1 |
| Optimizer Strategy | Transfer Learning + Fine-Tuning |

---

## 📊 Dataset

The project uses the **PlantVillage Dataset**.

| Property | Value |
|---|---:|
| Total Images | 54,305 |
| Number of Classes | 38 |
| Image Type | Color Images |
| Training Split | 70% |
| Validation Split | 15% |
| Testing Split | 15% |

The dataset was obtained through the Kaggle PlantVillage dataset.

> **Note:** The dataset itself is not included in this repository because of its size.

---

## 📈 Model Performance

The trained model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| **Accuracy** | **98.55%** |
| **Precision** | **98.60%** |
| **Recall** | **98.55%** |
| **F1 Score** | **98.55%** |

These results indicate strong classification performance on the test dataset.

---

## 🖥️ Streamlit Application

The project includes a Streamlit-based interface for making predictions.

### Application Workflow

```text
Upload Leaf Image
        ↓
Image Preprocessing
        ↓
EfficientNetB3 Model
        ↓
Disease Classification
        ↓
Confidence Score
        ↓
Top-3 Predictions
```

### Example

The user uploads a leaf image through the Streamlit interface.

The application then displays:

```text
Prediction: <Predicted Class>

Confidence: XX.XX%

Top Predictions:
1. Class A : XX.XX%
2. Class B : XX.XX%
3. Class C : XX.XX%
```

---

## 📸 Application Screenshots

### Main Application

![Plant Disease Detection App](screenshots/app.png)

### Prediction Result

![Prediction Result](screenshots/prediction.png)

> Replace `app.png` and `prediction.png` with the actual filenames inside the `screenshots` folder.

---

## 📂 Project Structure

```text
plant-disease-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── model/
│   ├── plant_disease_38class.h5
│   ├── class_names.json
│   └── config.json
│
├── notebook/
│   └── Untitled4.ipynb
│
└── screenshots/
    ├── app.png
    └── prediction.png
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/amritpal4148singh-spec/plant-disease-detection.git
```

### 2. Navigate to the Project

```bash
cd plant-disease-detection
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Making a Prediction

1. Open the Streamlit application.
2. Click **Upload a leaf image**.
3. Select a JPG, JPEG, or PNG image.
4. The image is resized to **192 × 192 pixels**.
5. The trained EfficientNetB3 model processes the image.
6. The predicted class and confidence score are displayed.
7. The application also displays the **Top-3 predictions**.

---

## 🛠️ Technologies Used

### Programming

- Python

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- EfficientNetB3
- NumPy
- Pillow

### Web Application

- Streamlit

### Development

- Google Colab
- Visual Studio Code
- Git
- GitHub
- Git LFS

---

## 📓 Training Notebook

The complete model development and training workflow is available in:

```text
notebook/Untitled4.ipynb
```

The notebook contains the training pipeline, model construction, transfer learning, fine-tuning, evaluation, and model export workflow.

---

## 💾 Trained Model

The trained model is stored using **Git LFS** because of its file size.

```text
model/plant_disease_38class.h5
```

Additional model configuration files:

```text
model/class_names.json
model/config.json
```

---

## 🚀 Future Improvements

Potential future improvements include:

- 📱 Mobile-friendly deployment
- 🌐 Cloud deployment
- 📷 Real-time camera-based detection
- 🌾 Additional plant species and disease classes
- 📊 Improved visualization of prediction probabilities
- 🧑‍🌾 Disease treatment and prevention recommendations
- 🔍 Explainable AI using techniques such as Grad-CAM
- ⚡ Model optimization for edge/mobile devices

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

Predictions generated by the model should not be considered a substitute for professional agricultural diagnosis.

---

## 👨‍💻 Author

**Amritpal Singh**

Computer Science & Engineering — AI & ML

---

## ⭐ Acknowledgements

- PlantVillage Dataset
- TensorFlow / Keras
- EfficientNet
- Streamlit
- Kaggle