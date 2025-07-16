# ASL Hand Gesture Recognition

A real-time American Sign Language (ASL) recognition system using computer vision and machine learning. This application can recognize hand gestures and convert them into text, making communication more accessible.

## Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe and OpenCV for hand detection and landmark extraction
- **Machine Learning Classification**: TensorFlow-based models for gesture classification
- **User-friendly GUI**: PyQt5-based desktop application with intuitive controls
- **Custom Training**: Tools to collect data and train your own gesture recognition models
- **Multiple Gesture Support**: Pre-trained models for common ASL signs
- **Visual Feedback**: Real-time hand landmark visualization and gesture confirmation

## Project Structure

```
ASL/
├── main/
│   ├── app.py                          # Main GUI application
│   ├── test_app.py                     # Data collection and model testing tool
│   ├── keypoint_classification.ipynb   # Keypoint model training notebook
│   ├── point_history_classification.ipynb  # Gesture model training notebook
│   ├── model/                          # Trained models
│   │   ├── keypoint_classifier/
│   │   └── point_history_classifier/
│   └── utils/                          # Utility functions
├── signs/                              # Hand sign reference images
└── requirements.txt                    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam for real-time recognition

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ASL-Recognition-App.git
   cd ASL-Recognition-App
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Application

```bash
python main/app.py
```

This launches the main GUI application where you can:
- Use your webcam for real-time ASL recognition
- View recognized gestures and build sentences
- Access the available signs reference
- Undo or clear recognized text

### Data Collection and Model Testing

```bash
python main/test_app.py
```

This tool allows you to:
- Collect training data for new gestures
- Test existing models
- Switch between different modes (normal, keypoint collection, point history collection)

**Controls:**
- `0-9`: Select gesture number for data collection
- `n`: Normal mode
- `k`: Keypoint collection mode
- `h`: Point history collection mode
- `ESC`: Exit

### Training Custom Models

1. **Collect Training Data:**
   - Use `test_app.py` to collect keypoint data for new gestures
   - Data is saved to CSV files in the model directories

2. **Train Keypoint Classifier:**
   - Open `keypoint_classification.ipynb`
   - Follow the notebook to train your custom keypoint classifier

3. **Train Point History Classifier:**
   - Open `point_history_classification.ipynb`
   - Follow the notebook to train gesture recognition models

## Model Architecture

- **Keypoint Classifier**: Neural network that classifies hand poses based on 21 hand landmarks
- **Point History Classifier**: Classifies gestures based on the movement history of the index finger tip

## Supported Gestures

The pre-trained models support common ASL signs including:
- Hello, Thank You, Please, Sorry
- Yes, No, I Love You
- Food, Water, Home
- And more...

## Customization

### Adding New Gestures

1. **Collect Data:**
   ```bash
   python main/test_app.py
   ```
   - Press `k` to enter keypoint collection mode
   - Press `0-9` to select gesture number
   - Perform the gesture multiple times to collect sufficient data

2. **Train Model:**
   - Use the provided Jupyter notebooks to train new models
   - Replace the existing model files with your trained models

### Modifying the GUI

The main application (`app.py`) uses PyQt5 and can be customized by:
- Modifying the UI layout in the `init_ui()` method
- Adjusting the gesture recognition logic
- Adding new features to the interface

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **Webcam not detected**
   - Ensure your webcam is connected and not in use by another application
   - Check camera permissions in your operating system

3. **Model files not found**
   - Ensure the `model/` directory contains the required `.tflite` and `.csv` files
   - Check file paths in the code if you've moved files

4. **Poor recognition accuracy**
   - Ensure good lighting conditions
   - Keep your hand clearly visible to the camera
   - Consider retraining models with more diverse data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow for machine learning capabilities
- PyQt5 for the GUI framework
- OpenCV for computer vision functionality

## Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Open an issue on GitHub with detailed information about your problem

---

**Note**: This application is designed for educational and accessibility purposes. For production use in critical applications, additional testing and validation is recommended.
