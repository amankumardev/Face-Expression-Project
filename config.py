"""
Configuration file for Face Expression Detection System
"""

# ============================================
# MODEL SELECTION
# ============================================
# Choose which model to use for emotion detection
# Options: 'FER' (faster, lighter) or 'DEEPFACE' (more accurate, heavier)
MODEL_TYPE = 'FER'  # Change to 'DEEPFACE' to use DeepFace model

# ============================================
# CAMERA SETTINGS
# ============================================
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# ============================================
# FACE DETECTION SETTINGS
# ============================================
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = (30, 30)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# ============================================
# FER MODEL SETTINGS
# ============================================
FER_MTCNN = True  # Use MTCNN face detector (more accurate but slower)

# ============================================
# DEEPFACE MODEL SETTINGS  
# ============================================
EXPRESSION_MODEL = 'emotion'  # DeepFace emotion detection
EXPRESSION_BACKEND = 'opencv'
EXPRESSION_ENFORCE_DETECTION = False
EXPRESSION_DETECTOR_BACKEND = 'opencv'

# ============================================
# UI SETTINGS
# ============================================
BOX_THICKNESS = 3
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_PADDING = 10

# Enhanced UI Effects
ENABLE_GLOW_EFFECT = True
GLOW_INTENSITY = 0.6
ENABLE_SMOOTH_BBOX = True  # Smooth bounding box transitions

# ============================================
# EXPRESSION COLORS (BGR format) - Enhanced Vibrant Palette
# ============================================
EXPRESSION_COLORS = {
    'happy': (30, 255, 30),       # Bright Green
    'sad': (255, 100, 50),         # Deep Blue
    'angry': (0, 50, 255),         # Bright Red
    'surprise': (0, 255, 255),     # Cyan/Yellow
    'fear': (180, 100, 180),       # Purple
    'disgust': (0, 140, 140),      # Teal/Brown
    'neutral': (200, 200, 200)     # Light Gray
}

# Glow colors (slightly transparent for overlay effect)
GLOW_COLORS = {
    'happy': (30, 255, 30, 100),
    'sad': (255, 100, 50, 100),
    'angry': (0, 50, 255, 100),
    'surprise': (0, 255, 255, 100),
    'fear': (180, 100, 180, 100),
    'disgust': (0, 140, 140, 100),
    'neutral': (200, 200, 200, 100)
}

# ============================================
# PERFORMANCE SETTINGS
# ============================================
SKIP_FRAMES = 2  # Process every Nth frame for better performance
ENABLE_GPU = False

# ============================================
# OUTPUT SETTINGS
# ============================================
SCREENSHOT_DIR = 'screenshots'
SCREENSHOT_PREFIX = 'expression_'
SCREENSHOT_FORMAT = 'jpg'

# ============================================
# DEBUG SETTINGS
# ============================================
SHOW_FPS = True
SHOW_CONFIDENCE = True
DEBUG_MODE = False
