#  SignBridge – Gesture to Text Conversion System

> A real-time Indian Sign Language (ISL) gesture-to-text mobile application that uses a smartphone camera to recognize hand gestures and convert them into readable text — making communication accessible for the hearing-impaired community in India.

![Status](https://img.shields.io/badge/Status-Basic%20Version%20Working-orange?style=flat)
![Flutter](https://img.shields.io/badge/Flutter-02569B?style=flat&logo=flutter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)

---

 Problem Statement

Over 6 crore people in India live with hearing impairment. Communication between hearing-impaired individuals who use Indian Sign Language (ISL) and the general public is a major barrier in daily life. Existing systems are either computationally expensive, not trained on ISL, or require specialized hardware beyond a standard smartphone.

SignBridge addresses this gap by providing a **lightweight, real-time, mobile-based ISL gesture-to-text converter** that runs entirely on a standard Android smartphone — no additional hardware required.

---

##  Features

- Real-time hand gesture detection using phone camera
-  21-point hand landmark extraction using MediaPipe Hands
-  On-device ISL gesture classification using TensorFlow Lite (MLP model)
-  Secure user authentication via Firebase
-  Live text overlay on camera feed
- Translation history screen with timestamps
-  Inference latency under 1 second | 12–15 FPS on mid-range devices

---

##  Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Mobile Frontend | Flutter / Dart | Cross-platform Android UI |
| Hand Detection | MediaPipe Hands | Real-time 21-point landmark extraction |
| ML Model (Training) | TensorFlow / Keras (Python) | MLP gesture recognition model |
| ML Model (Deployment) | TensorFlow Lite | On-device inference on Android |
| Authentication | Firebase Authentication | Secure email/password login |
| Database | Firebase Firestore | User data and translation history |
| Language (ML Pipeline) | Python 3.x | Data collection, preprocessing, training |
| Version Control | Git / GitHub | Code versioning |

---

##  System Architecture
SignBridge/
├── flutter_app/              # Mobile frontend (Flutter/Dart)
│   ├── lib/
│   │   ├── main.dart         # App entry point
│   │   ├── auth_screen.dart  # Login & registration
│   │   ├── home_screen.dart  # Main camera screen
│   │   └── signbridge_model.dart  # TFLite inference + MediaPipe
│   └── pubspec.yaml
├── ml_model/                 # Python ML pipeline
│   ├── feature_extraction.py # MediaPipe landmark extraction → CSV
│   ├── train.py              # MLP model training (.h5)
│   ├── convert_tflite.py     # Convert .h5 → .tflite
│   └── dataset/              # ISL gesture images per label
├── assets/
│   └── model.tflite          # Deployed model
└── README.md
---

## How It Works

1. User opens app and logs in via Firebase Authentication
2. Camera streams live video frames at ≥10 FPS
3. MediaPipe Hands extracts **21 3D landmarks** per hand → 126-dimensional feature vector (2 hands × 21 landmarks × 3 axes)
4. Feature vector is normalized (wrist-relative, scale-invariant) and passed to TFLite MLP model
5. Model classifies the gesture into an ISL label with a confidence score
6. Predicted letter is displayed as a real-time text overlay on the camera screen
7. Translation history is saved to Firebase Firestore

---

##  Getting Started

### Prerequisites
- Flutter SDK (Android target: Android 8.0+)
- Python 3.x
- Firebase project with Authentication and Firestore enabled
- MediaPipe hand landmarker model file (`hand_landmarker.task`)

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

3. Add your Firebase config file to `android/app/`:
   - Go to [console.firebase.google.com](https://console.firebase.google.com)
   - Open your Firebase project → Project Settings
   - Under "Your apps" → Android app → click "Download google-services.json"
   - Place the downloaded file here:
```
   android/
   └── app/
       └── google-services.json   ← place it here
```

   >  Never commit this file to GitHub. Make sure `google-services.json` is in your `.gitignore`

4. Install Python dependencies for ML pipeline
```bash
pip install mediapipe tensorflow opencv-python pandas numpy scikit-learn
```

5. Run the app
```bash
flutter run
```

---

##  Performance Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Test Cases Passed | 9/9 | ✅ 9/9 (100%) |
| Inference Latency | < 1 second | ✅ ~0.6 – 0.8 seconds |
| FPS on Mid-range Device | ≥ 10 FPS | ✅ 12–15 FPS |
| Authentication Success Rate | 100% | ✅ 100% |
| Gesture Recognition Accuracy | ≥ 90% | ✅ ~93% (good lighting) |

> Note: Accuracy drops to ~78% under poor lighting — a retry prompt is shown in this case.

---

##  Current Status

- [x] Firebase authentication (login, registration, email verification)
- [x] Live camera feed with MediaPipe hand landmark detection
- [x] TFLite MLP model — ISL static gesture recognition
- [x] Real-time text overlay on camera screen
- [x] Translation history screen
- [x] Skeleton painter overlay on detected hand
- [ ] Dynamic gesture sequence recognition (LSTM/Transformer)
- [ ] Multi-language output (Kannada, Hindi)
- [ ] Animated ISL avatar (text-to-sign)
- [ ] iOS support
- [ ] Text-to-Speech (TTS) toggle
- [ ] Cloud-based dataset expansion

---

##  Social Impact
SignBridge is built for the **6 crore+ hearing-impaired individuals in India** who use Indian Sign Language. By running entirely on a standard Android smartphone with no internet required for inference, it is accessible, affordable, and practical for everyday use.

---

##  Security Note

The `google-services.json` file is **not included** in this repository for security reasons.
To run this project locally, generate your own Firebase project and download
your own config file as described in the setup steps above.

---

##  License

This project is open source and available under the [MIT License](LICENSE).
