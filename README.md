# 🎙️ Speech Emotion Recognition (SER) using CNN-BiLSTM

This project focuses on detecting human emotions from speech using a deep learning model built with a combination of Convolutional Neural Networks (CNN) and Bidirectional LSTM layers.

## 📌 Highlights

- Built using **TensorFlow, Keras, Librosa** and **SoundFile**
- Uses **custom feature extraction** (MFCC, ZCR, Chroma, RMS)
- **Bidirectional LSTM + CNN** architecture
- **Data augmentation** performed for **'surprised'** emotion to handle class imbalance
- Trained on **RAVDESS**, **TESS**, **CREMA-D**, and **SAVEE** datasets

---

## 📂 Datasets Used

| Dataset | Description | Kaggle Link |
|--------|-------------|-------------|
| RAVDESS | Ryerson Audio-Visual Database of Emotional Speech and Song | [🔗 RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) |
| TESS | Toronto Emotional Speech Set | [🔗 TESS on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |
| CREMA-D | Crowd-Sourced Emotional Multimodal Actors Dataset | [🔗 CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) |
| SAVEE | Surrey Audio-Visual Expressed Emotion Dataset | [🔗 SAVEE on Kaggle](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) |

> Note: Downloading may require a Kaggle account and acceptance of dataset license terms.

---

### 🛠️ Built With

- 🧪 **TensorFlow & Keras** — for building and training the neural network  
- 🎵 **Librosa** — for audio feature extraction (MFCC, Chroma, ZCR, etc.)  
- 🔊 **SoundFile** — for audio file handling

---

## 🧬 Feature Extraction

The following features were extracted from each audio sample:

- 🎵 **MFCC** (40 Mel-frequency cepstral coefficients)
- 🔄 **Zero Crossing Rate (ZCR)**
- 🎹 **Chroma STFT**
- 🔊 **RMS Energy**

These features were padded or truncated to ensure a consistent shape of **300 frames**, resulting in a final feature matrix of shape **(300, 54)** per audio file.  
Before feeding into the model, all features were **standardized using StandardScaler** for better convergence.

---

## 🔁 Data Augmentation (for ‘Surprised’ Emotion)

To address class imbalance, particularly for the **‘surprised’** emotion, the following audio augmentation techniques were applied:

- 🎛️ **Pitch Shifting**: Randomly altered the pitch of the audio without changing tempo.
- ⏩ **Time Stretching**: Adjusted playback speed while preserving pitch.
- 🌫️ **Gaussian Noise**: Added subtle white noise to increase robustness.

These augmentations enhanced the model’s ability to generalize, especially on underrepresented emotional classes, and helped reduce overfitting.


**Shape of features:** `(300, 54)`  
**Input shape to model:** `(batch_size, 300, 1)` (after reshaping)

---

## 🧠 Model Architecture

The model combines:
- 2 Convolutional layers for local feature extraction
- Batch Normalization and MaxPooling for stability and dimensionality reduction
- Dropout layers to prevent overfitting
- Bidirectional LSTM to capture temporal context
- Dense layers for final emotion classification

---

## 📈 Training

- Optimizer: Adam (`lr=0.0001`)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## 📊 Results

The model achieves high accuracy across multiple emotions including:
- Happy
- Sad
- Fear
- Angry
- Disgust
- Surprise
- Neutral

---

## 🚀 Model Performance

The trained CNN-BiLSTM model performs **exceptionally well** during prediction. It demonstrates:

- ✅ High accuracy across multiple emotion classes
- ✅ Robust performance on unseen data from diverse datasets
- ✅ Excellent generalization due to augmentation and regularization

Especially after applying data augmentation to the underrepresented **‘Surprised’** emotion class, the model shows a significant improvement in handling class imbalance.

📊 The final model achieves strong results in terms of **precision, recall, and F1-score**, making it highly reliable for **real-time emotion recognition** tasks.

---

## 📁 Project Files

📁 speech-emotion-recognition/  
├── 📓 speech_emotion_recognition.ipynb     — Jupyter Notebook (model training & evaluation)  
├── 🧠 speech_emotion_model.keras           — Trained deep learning model  
└── 📄 README.md                            — Project documentation

---

Feel free to explore, run, or improve upon it! 🚀

---

## 📄 License

This project is licensed under the **MIT License** — a permissive license that allows anyone to freely use, modify, and distribute the code, provided proper credit is given.

You are welcome to:

- ✅ Use the code for personal, academic, or commercial purposes
- ✅ Modify and adapt the project to your needs
- ✅ Share or distribute your versions with attribution

You must:

- 🏷️ Include the original copyright and license
- 🚫 Not hold the authors liable for any damage caused by the use of this software
