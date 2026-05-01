## Facial Expression Recognition (FER) Using CNN

This project implements a real-time facial expression recognition system using a **Convolutional Neural Network (CNN)**. The model is trained on the **FER-2013** dataset and can classify human faces into seven distinct emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**[cite: 1].

---

### 🚀 Key Features
*   **Deep Learning Architecture**: A multi-block CNN featuring convolutional layers, batch normalization, and dropout for high-accuracy feature extraction[cite: 1].
*   **Real-Time Inference**: Integrated OpenCV pipeline for live webcam detection with a mirrored, full-screen display[cite: 1].
*   **Data Augmentation**: Robust training using `ImageDataGenerator` to improve model generalization against various head angles and lighting[cite: 1].
*   **Optimized Performance**: Achieved ~64.3% validation accuracy, nearing human-level performance on the FER-2013 dataset[cite: 1].

---

### 🧠 Project Architecture

The model architecture is designed to handle grayscale images of $48 \times 48$ pixels[cite: 1].



1.  **Preprocessing**: Grayscale conversion, resizing to $48 \times 48$, and pixel normalization (0-1)[cite: 1].
2.  **Feature Extraction Blocks**: Four sequential blocks using `Conv2D`, `BatchNormalization`, and `ELU` activation to capture facial patterns[cite: 1].
3.  **Regularization**: Strategic use of `Dropout` (ranging from 0.2 to 0.5) to prevent overfitting during deep training[cite: 1].
4.  **Classification Head**: A `Flatten` layer followed by `Dense` (fully connected) layers and a `Softmax` output for 7-class probability[cite: 1].

---

### 📊 Performance Analysis

The model was evaluated using a classification report and a confusion matrix to identify patterns in emotion recognition[cite: 1].

*   **Top Performer**: "Happy" showed the highest F1-score (0.83), as the features for a smile are distinct[cite: 1].
*   **Challenge Areas**: The model sometimes confuses "Neutral" with "Sad" due to subtle facial feature similarities in the dataset[cite: 1].
*   **Optimization**: Used `ReduceLROnPlateau` and `ModelCheckpoint` to ensure only the best version of the model is saved during training[cite: 1].



---

### 🛠️ Installation & Usage

#### Prerequisites
*   Python 3.10+
*   TensorFlow / Keras
*   OpenCV
*   Scikit-learn

#### Instructions
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/facial-expression-recognition.git
    cd facial-expression-recognition
    ```
2.  **Run the Live Demo**:
    Ensure your webcam is connected. Run the final cell in `FER_Project_5.ipynb` or your specific camera script[cite: 1].
    *   Press **'q'** to exit the full-screen demo[cite: 1].

---

### 📂 File Structure
*   `FER_Project_5.ipynb`: The primary Jupyter Notebook containing data loading, model architecture, training, and evaluation[cite: 1].
*   `improved_model.h5`: The pre-trained weights for the CNN[cite: 1].
*   `archive/`: Contains the training and testing datasets (FER-2013).

---

### 🤝 Contributing
Contributions are welcome! If you have ideas to improve the "Neutral" vs "Sad" classification or want to optimize the real-time frame rate, feel free to open an issue or submit a pull request.
```
