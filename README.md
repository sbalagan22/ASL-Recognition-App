# ASL Recognition

A modern, real-time American Sign Language (ASL) recognition application using computer vision and deep learning. This app provides an intuitive, professional interface for recognizing and transcribing ASL hand signs from your webcam, with a focus on extensibility and ease of use for both end-users and researchers.

---

## 📝 Overview

**ASL Recognition** is a desktop application that uses your webcam to detect and recognize American Sign Language hand signs in real time. It leverages MediaPipe for hand tracking and custom-trained neural networks for gesture classification. The app is designed for:

- **ASL learners and educators**: Practice and visualize sign recognition.
- **Accessibility**: Help bridge communication between ASL users and non-signers.
- **Research and prototyping**: Easily collect new data and retrain models for custom signs.

---

## ✨ Features

- **Live webcam hand sign recognition** with a modern, dark-themed UI.
- **Sentence builder**: Recognized signs are appended to a text box, allowing you to build sentences naturally.
- **Undo, Clear, and Signs buttons**:
  - **Undo**: Remove the last word from the sentence.
  - **Clear**: Clear the entire sentence.
  - **Signs**: View all available hand signs with their images.
- **Visual feedback**: Timer for sign hold, and a "VALID" indicator when a sign is successfully recognized.
- **Resizable, professional layout**: Facecam, controls, and text box are proportioned for usability and aesthetics.
- **Easy extensibility**: Add new signs by collecting data and retraining the model.

---

## 🖥️ User Interface

- **Top Bar**: Bold title bar for the application.
- **Left Side (5/7 width)**:
  - **Facecam**: Live video feed with hand tracking and sign overlays.
  - **Buttons**: Undo, Clear, and Signs, centered and visually prominent.
- **Right Side (2/7 width)**:
  - **Sentence Box**: Editable text area where recognized signs are appended as words.
- **Signs Window**: Shows all available signs with their names and example images.

---

## ⚙️ How It Works

- **Hand Tracking**: Uses [MediaPipe](https://google.github.io/mediapipe/) to detect and track hand landmarks in real time.
- **Sign Classification**: Two TensorFlow Lite models:
  - `keypoint_classifier.tflite`: Classifies static hand poses.
  - `point_history_classifier.tflite`: (Optional) Classifies dynamic gestures based on hand movement history.
- **Recognition Logic**:
  - When you hold a sign for at least 1 second, the app recognizes it and appends the corresponding word to the sentence box.
  - Visual feedback (timer and "VALID" indicator) helps you know when a sign is registered.
- **Sign Images**: The `/signs` folder contains example images for each available sign, used in the Signs window.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-folder>/hand-gesture-recognition-mediapipe-main
```

### 2. Set Up a Python Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 4. Ensure Model Files Are Present
- `model/keypoint_classifier/keypoint_classifier.tflite`
- `model/point_history_classifier/point_history_classifier.tflite`

If you retrain or add new signs, you will need to update these files (see below).

### 5. Run the Application
```bash
python3 app_gui.py
```

---

## 🖐️ How to Add or Train New Signs

1. **Collect Data**
   - Run the original `app.py` in data collection mode (see its instructions or code comments).
   - Use the key logging feature to record new hand sign data. Each sign should have a unique label.
   - The data is saved as CSV files in the `model/keypoint_classifier/` and `model/point_history_classifier/` folders.

2. **Train the Model**
   - Use the provided Jupyter notebooks (`keypoint_classification.ipynb`, etc.) to train a new model on your collected data.
   - Export the trained model as a `.tflite` file.

3. **Update the Model Files**
   - Replace the old `.tflite` files in `model/keypoint_classifier/` and `model/point_history_classifier/` with your new ones.
   - Update the label CSV files to include your new sign names (use the same format as existing labels).

4. **Add Example Images (Optional)**
   - Place a PNG image for each new sign in the `/signs` folder, named exactly as the label (e.g., `My_Sign.png`).
   - The Signs window will automatically display these images.

5. **Restart the App**
   - Run `python3 app_gui.py` again. Your new signs will be recognized and available in the UI.

---

## 📁 Folder Structure

```
hand-gesture-recognition-mediapipe-main/
├── app_gui.py                # Main PyQt5 application
├── app.py                    # Original OpenCV application (for data collection)
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint_classifier_label.csv
│   └── point_history_classifier/
│       ├── point_history_classifier.tflite
│       ├── point_history_classifier_label.csv
├── signs/                    # Example images for each sign
├── requirements.txt
└── ...
```

---

## 🙏 Credits & Acknowledgments

- Built with [MediaPipe](https://google.github.io/mediapipe/), [OpenCV](https://opencv.org/), and [TensorFlow](https://www.tensorflow.org/).
- PyQt5 for the modern desktop UI.
- Inspired by open-source ASL and gesture recognition projects.
- Special thanks to the ASL community and contributors.

---

## 📬 Contact & Issues

For questions, suggestions, or bug reports, please open an issue or contact the maintainer.

---

Enjoy using ASL Recognition to learn, teach, and bridge communication with sign language! ✋🤟
