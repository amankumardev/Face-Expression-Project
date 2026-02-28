# 🎭 AI Face Expression Detection System

A professional real-time facial emotion recognition application powered by computer vision and deep learning. Detects faces via webcam and classifies emotions with dynamic colored bounding boxes and labels.

## ✨ Features

🚀 **Dual Model Support**: Switch between FER (fast) and DeepFace (accurate) models  
⚡ **Real-time Detection**: Live webcam feed with instant emotion detection (25+ FPS)  
🎯 **Multi-Face Support**: Detect and classify emotions for multiple faces simultaneously  
🎨 **Premium UI**: Glassmorphic effects, glow effects, and corner accents  
📊 **Performance Metrics**: Real-time FPS counter with color-coded status  
📸 **Screenshot Capability**: Save snapshots of detected expressions  
🔄 **Flexible Configuration**: Easy model switching and customization

## 🎨 Detected Emotions

| Emotion | Color | Hex |
|---------|-------|-----|
| 😊 **Happy** | Bright Green | `#1EFF1E` |
| 😢 **Sad** | Deep Blue | `#FF6432` |
| 😠 **Angry** | Bright Red | `#FF3200` |
| 😲 **Surprise** | Cyan/Yellow | `#FFFF00` |
| 😨 **Fear** | Purple | `#B464B4` |
| 🤢 **Disgust** | Teal | `#8C8C00` |
| 😐 **Neutral** | Light Gray | `#C8C8C8` |

## 🚀 Quick Start

### Option 1: Quick Demo (FER Only)
For a fast, lightweight demo:
```bash
python fer_demo.py
```

### Option 2: Full Application (Supports Both Models)
For advanced features and model selection:
```bash
python main.py
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

### Setup Instructions

1. **Navigate to project directory**:
   ```bash
   cd "c:\coding\Python\Face Expression Project"
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: First installation may take several minutes:
   - **FER**: ~100MB (lightweight, fast)
   - **DeepFace**: ~500MB (TensorFlow included, more accurate)

## ⚙️ Configuration

Edit [`config.py`](file:///c:/coding/Python/Face%20Expression%20Project/config.py) to customize:

### Model Selection
```python
MODEL_TYPE = 'FER'  # or 'DEEPFACE'
```

**Model Comparison**:

| Feature | FER | DeepFace |
|---------|-----|----------|
| Speed | ⚡⚡⚡ Fast (25-30 FPS) | ⚡⚡ Moderate (15-20 FPS) |
| Accuracy | 🎯🎯 Good | 🎯🎯🎯 Excellent |
| Size | 📦 ~100MB | 📦 ~500MB |
| CPU Usage | 💻 Low | 💻💻 Medium |
| Best For | Real-time demos | High accuracy needs |

###Other Settings
- **Camera**: Index, resolution, FPS target
- **Detection**: Confidence thresholds, face size
- **UI Effects**: Glow effects, colors, fonts
- **Performance**: Frame skip rate, GPU usage

## 🎮 Usage

### Running the Application

```bash
python main.py
```

The application will:
1. Display selected model (FER or DeepFace)
2. Initialize face and emotion detection
3. Open your default webcam
4. Show live video with emotion detection
5. Display real-time performance metrics

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save screenshot to `screenshots/` folder |

### Tips for Best Results

✅ **Lighting**: Ensure good, even lighting on your face  
✅ **Distance**: Position yourself 1-3 feet from the camera  
✅ **Angle**: Face the camera directly for optimal detection  
✅ **Expressions**: Make clear, distinct facial expressions  
✅ **Multiple Faces**: Application supports multiple people in frame

## 📁 Project Structure

```
Face Expression Project/
├── main.py                      # Main application (full featured)
├── fer_demo.py                  # Quick demo script (FER only)
├── config.py                    # Configuration settings
├── emotion_detector.py          # Unified model interface
├── fer_classifier.py            # FER model implementation
├── expression_classifier.py     # DeepFace model implementation
├── face_detector.py             # OpenCV face detection
├── ui_overlay.py                # Enhanced UI rendering
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
├── screenshots/                 # Saved screenshots (auto-created)
└── README.md                    # This file
```

## 🛠️ Troubleshooting

### Camera Not Opening

- **Issue**: "ERROR: Could not open camera!"
- **Solution**: 
  - Check if another application is using the webcam
  - Try changing `CAMERA_INDEX` in `config.py` (try 0, 1, or 2)
  - Ensure webcam drivers are installed

### Low FPS / Laggy Performance

- **Issue**: Frame rate below 15 FPS
- **Solution**:
  - Switch to FER model: `MODEL_TYPE = 'FER'` in `config.py`
  - Increase `SKIP_FRAMES` value (try 3 or 4)
  - Reduce camera resolution in `config.py`
  - Disable glow effects: `ENABLE_GLOW_EFFECT = False`
  - Close other resource-intensive applications

### Face Not Detected

- **Issue**: Bounding box doesn't appear around face
- **Solution**:
  - Improve lighting conditions
  - Move closer to camera
  - Face the camera directly
  - Adjust `FACE_DETECTION_CONFIDENCE` in `config.py`

### FER Library Error

- **Issue**: "FER library not installed" or import error
- **Solution**:
  ```bash
  pip install fer
  ```

### DeepFace Model Error

- **Issue**: DeepFace fails to load or analyze
- **Solution**:
  - Ensure TensorFlow is installed: `pip install tensorflow`
  - Switch to FER model as fallback
  - Check internet connection (first run downloads models)

## 🔧 Technical Details

### Technologies Used

- **OpenCV**: Real-time computer vision and face detection
- **FER**: Lightweight emotion recognition library
- **DeepFace**: State-of-the-art facial analysis framework
- **TensorFlow**: Deep learning backend for DeepFace
- **NumPy**: Numerical computing for image processing
- **MTCNN**: Advanced face detection (optional with FER)

### How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade or MTCNN to detect face regions
2. **Face Extraction**: Extracts detected face regions with padding
3. **Emotion Analysis**: 
   - **FER**: Lightweight CNN model analyzes face directly
   - **DeepFace**: Advanced ensemble models for higher accuracy
4. **Emotion Classification**: Returns dominant emotion and confidence scores
5. **Visual Rendering**: Draws enhanced boxes with glow effects and corner accents
6. **Real-time Display**: Shows processed video feed with all overlays

### Performance Optimization

- Frame skipping for better FPS
- Efficient emotion caching
- Model warm-up on startup
- Optimized UI rendering with numpy operations

## 🎨 Visual Enhancements

### Glow Effects
- Layered bounding boxes with decreasing opacity
- Configurable intensity via `GLOW_INTENSITY`
- Toggle with `ENABLE_GLOW_EFFECT`

### Corner Accents
- Professional-looking corner highlights
- Dynamic sizing based on face dimensions
- Matches emotion color

### Color-Coded FPS
- 🟢 Green: 25+ FPS (Excellent)
- 🟡 Yellow: 15-24 FPS (Good)
- 🔴 Red: <15 FPS (Poor)

## 🚀 Future Enhancements

- 📊 Emotion timeline tracking and graphs
- 💾 Export emotion data to CSV/JSON
- 🎥 Record video with emotion overlays
- 🌐 Web interface for browser-based detection
- 📱 Mobile app version
- 🧠 Custom model training
- 👥 Face recognition + expression tracking

## 📄 License

This project is for educational and personal use. Please respect the licenses of the libraries used:
- FER: MIT License
- DeepFace: MIT License
- OpenCV: Apache 2.0 License
- TensorFlow: Apache 2.0 License

## 🙏 Acknowledgments

- **FER** library for fast emotion recognition
- **DeepFace** by SerengulLabs for facial analysis framework
- **OpenCV** community for computer vision tools
- Pre-trained models from FER2013 dataset contributors

---

**Created with ❤️ using Python, OpenCV, FER, and DeepFace**

For questions or issues, please refer to the troubleshooting section above.
