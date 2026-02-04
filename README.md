
# 🌾 CropSpectra - AI-Powered Crop Disease Detection System

**CropSpectra** is an intelligent crop disease prediction system that uses **Deep Learning** and **Computer Vision** to detect diseases in crops like Tomato, Potato, and Bell Pepper.

---

🔗 **Live Deployment (Render – Free Tier):**  
[https://your-render-app-link](https://cropspectra.onrender.com)

⚠️ *Note:* Due to limited resources in Render Free Tier, live image prediction is not supported in the deployed version.

---


## 🎥 Project Demo Video

A complete working demonstration of the project is shown in the video below.

🎬 **Demo Video Link:**  
https://drive.google.com/your-video-link  
(or YouTube unlisted link)

The demo video shows:
- Running the application via terminal
- Loading the trained deep learning model
- Performing crop disease prediction on real images
- Correct disease output with details



## 🚀 Features

✅ **AI-Based Disease Detection** - Deep learning powered predictions  
✅ **Multi-Language Support** - Hindi, Bengali, Marathi, Bhojpuri  
✅ **Text-to-Speech** - Audio output in multiple languages  
✅ **User Authentication** - Secure login system  
✅ **Real-Time Prediction** - Instant disease identification  

---

## 📋 System Requirements

- **Python**: 3.8, 3.9, or 3.10 (Recommended: 3.9)
- **Operating System**: Windows / Linux / macOS
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space

---

## 🔐 Quick Login

**Username:** `crop`  
**Password:** `crop`

*(Hardcoded credentials for quick access)*

---

## 📂 Project Structure

```
CropSpectra/
│
├── venv/                       # Virtual environment (pre-configured)
├── app.py                      # Main Flask application
├── classes.json                # Crop disease class names
├── disease_info.json           # Disease details
├── crop_disease_model.h5       # Trained ML model
├── train_features.pkl          # Feature vectors
├── train_labels.pkl            # Training labels
├── users.db                    # SQLite database (auto-created)
├── requirements.txt            # Python dependencies
├── feedback.csv                # User feedback data
│
├── templates/                  # HTML templates
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│   ├── predict.html
│   ├── about.html
│   ├── blogs.html
│   └── contact.html
│
└── static/
    └── uploads/                # Uploaded images & audio files
```



## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Flask | 2.3.3 | Web framework |
| TensorFlow | 2.13.0 | Deep learning |
| deep-translator | 1.11.4 | Translation |
| gTTS | 2.3.2 | Text-to-speech |
| bcrypt | 4.0.1 | Password hashing |

---

## 📝 How to Use

1. **Login** - Use `crop`/`crop` or create a new account
2. **Upload Image** - Go to "Predict" page and upload crop leaf image
3. **View Results** - Get disease prediction with details
4. **Text-to-Speech** - Click speaker button to listen
5. **Translate** - Select language (Hindi/Bengali/Marathi/Bhojpuri) and translate

---

## 🌱 Supported Crops

- **Tomato** (9 diseases)
- **Potato** (3 diseases)
- **Bell Pepper** (2 diseases)

---

## 🔄 Stop Application

Press `Ctrl+C` in terminal

To deactivate virtual environment:
```bash
deactivate
```
---

## 🚧 Deployment Limitation

The project is deployed on Render using the free tier.

Due to limited RAM and storage:
- The deep learning model (.h5 / .pkl) cannot be fully loaded
- Live image prediction is not supported on the deployed version

✅ The application works correctly in the local environment,  
which is demonstrated in the provided demo video.

---

## 📧 Contact & Support

📧 **Email:** cropspectra@gmail.com  
🌍 **Website:** [CropSpectra](#)

---

## 🙏 Acknowledgments

- PlantVillage Dataset
- TensorFlow & Keras Community
- Flask Framework
- Google Translate API

---



**© 2024 CropSpectra | Smart Vision for a Healthier Harvest 🌿**
