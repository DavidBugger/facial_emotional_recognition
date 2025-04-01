
![facial_recognition](https://github.com/user-attachments/assets/b32b4021-41ca-4dc7-96c5-02059d0ddf84)
<br>

# Facial Emotion Recognition System

## Overview

This project is an end-to-end **Facial Emotion Recognition System** that detects emotions such as Happy, Sad, Angry, Neutral, etc. It consists of:

- **Deep Learning Model (CNN)** trained on facial emotion datasets (e.g., FER2013)
- **Django Web Application** to serve predictions via an API
- **Webcam Integration** for real-time emotion detection
- **Deployment Strategy** to run the system in production

---

## Features

✅ **Train a Deep Learning Model** using TensorFlow & Keras\
✅ **Real-Time Emotion Detection** using OpenCV\
✅ **Django API for Model Inference**\
✅ **Web Interface with Webcam**\
✅ **Deployable via Docker/AWS/Heroku**

---


## Installation & Setup

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/your-repo/FacialEmotionRecognition.git
cd FacialEmotionRecognition
```

### 2️⃣ **Set Up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Train the Model (Optional)**

If you want to train a new model, run the Jupyter notebook:

```bash
cd model_training
jupyter notebook train_model.ipynb
```

### 5️⃣ **Run the Django Server**

```bash
cd backend
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` in your browser.

---

## Deployment

For deploying the system, consider:

- **Docker**
- **AWS EC2 / Lambda**
- **Heroku / Render**

---

## Future Improvements

🔹 Use **Transfer Learning** (ResNet, MobileNet) for better accuracy\
🔹 Enhance UI with **React.js** for a smoother experience\
🔹 Improve real-time detection using **YOLO or Faster R-CNN**

---

## Contributors

👤 **David Akanang**\
🔗 [GitHub](https://github.com/DavidBugger)\
✉️ Contact: [devdavesolutions@gmail.com](mailto\:devdavesolutions@gmail.com)
<br>


![happy](https://github.com/user-attachments/assets/066048b1-c19e-40ce-989d-6f07d676d00d)
<br>

![neutral_real](https://github.com/user-attachments/assets/5b9389a6-2439-43ac-853a-5cb8970690da)

