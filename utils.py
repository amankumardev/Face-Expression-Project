"""
Utility functions for Face Expression Detection System
"""

import os
import cv2
import numpy as np
from datetime import datetime
from config import SCREENSHOT_DIR, SCREENSHOT_PREFIX, SCREENSHOT_FORMAT


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_screenshot(frame):
    """Save a screenshot with timestamp"""
    ensure_dir(SCREENSHOT_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_PREFIX}{timestamp}.{SCREENSHOT_FORMAT}"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    
    cv2.imwrite(filepath, frame)
    print(f"Screenshot saved: {filepath}")
    return filepath


def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess face image for expression detection"""
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to grayscale if needed
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized
    
    # Normalize pixel values
    face_normalized = face_gray / 255.0
    
    return face_normalized


def calculate_fps(prev_time, current_time):
    """Calculate frames per second"""
    time_diff = current_time - prev_time
    if time_diff > 0:
        fps = 1.0 / time_diff
        return fps
    return 0


def draw_text_with_background(frame, text, position, font, font_scale, 
                               text_color, bg_color, thickness=2, padding=5):
    """Draw text with a background rectangle for better visibility"""
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, text_color, thickness
    )
    
    return frame


def get_dominant_emotion(emotion_dict):
    """Get the dominant emotion from DeepFace results"""
    if not emotion_dict:
        return 'neutral', 0.0
    
    # Find emotion with highest confidence
    dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])
    return dominant_emotion[0], dominant_emotion[1]


def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence:.1f}%"


def is_valid_face(face_coords, min_size=(30, 30)):
    """Check if detected face meets minimum size requirements"""
    x, y, w, h = face_coords
    return w >= min_size[0] and h >= min_size[1]
