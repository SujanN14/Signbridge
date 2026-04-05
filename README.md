# Signbridge

> A mobile application that detects sign language hand gestures in real-time using your phone's camera — bridging communication between the hearing and deaf communities.

![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=flat)
![Flutter](https://img.shields.io/badge/Flutter-02569B?style=flat&logo=flutter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)

---

## 📱 About the Project

Signbridge is a mobile app that uses the phone camera to detect and recognize sign language hand gestures in real-time. The goal is to make communication more accessible for people who are deaf or hard of hearing by using AI to understand sign language automatically.

---

## ✨ Features

- 📷 Real-time hand gesture detection using phone camera
- 🤖 AI-powered sign language recognition using a trained ML model
- 📱 Smooth mobile experience built with Flutter
- ☁️ Firebase integration for backend services
- 🐍 Python-based ML model training and inference

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile App | Flutter (Dart) |
| ML Model | TensorFlow / Keras |
| Backend | Python |
| Database & Auth | Firebase |

---

## 🏗️ Project Architecture
Signbridge/
├── flutter_app/        # Mobile frontend (Flutter)
│   ├── lib/
│   └── pubspec.yaml
├── ml_model/           # Sign detection model
│   ├── train.py
│   └── model/
├── backend/            # Python backend API
│   └── app.py
└── README.md

---

## 🚀 Getting Started

### Prerequisites
- Flutter SDK installed
- Python 3.8+
- Firebase account

### Installation

1. Clone the repository
```bash
git clone https://github.com/SujanN14/Signbridge.git
cd Signbridge
```

2. Install Flutter dependencies
```bash
flutter pub get
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Add your Firebase config file (`google-services.json`) to the `android/app` folder

5. Run the app
```bash
flutter run
```

---

## 📊 Current Status

- [x] Flutter app base structure
- [x] Camera integration
- [x] ML model — basic version working
- [x] Firebase setup
- [ ] Full sign language alphabet detection
- [ ] Text-to-speech output
- [ ] Offline mode support

---

## 🎯 Goal

To create an accessible, affordable tool that helps break the communication barrier between the deaf community and the rest of the world — using nothing but a smartphone.

---

## 👨‍💻 Developer

**Sujan N**  
Computer Science Student | Mysuru, Karnataka, India  
[GitHub](https://github.com/SujanN14)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
