"""
Face Detection Module using OpenCV
"""

import cv2
import numpy as np
from config import (
    SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE,
    FACE_DETECTION_CONFIDENCE
)
from utils import is_valid_face


class FaceDetector:
    """Face detector using OpenCV's Haar Cascade"""
    
    def __init__(self):
        """Initialize face detector with Haar Cascade"""
        # Load pre-trained Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        print("Face detector initialized successfully")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            List of face coordinates as (x, y, w, h) tuples
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter valid faces
        valid_faces = [
            (x, y, w, h) for (x, y, w, h) in faces 
            if is_valid_face((x, y, w, h), MIN_FACE_SIZE)
        ]
        
        return valid_faces
    
    def extract_face_region(self, frame, face_coords):
        """
        Extract face region from frame
        
        Args:
            frame: Input frame
            face_coords: Face coordinates (x, y, w, h)
            
        Returns:
            Extracted face region as numpy array
        """
        x, y, w, h = face_coords
        
        # Add padding for better expression detection
        padding = int(0.1 * min(w, h))
        
        # Calculate padded coordinates
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract region
        face_region = frame[y1:y2, x1:x2]
        
        return face_region
    
    def get_face_center(self, face_coords):
        """Get center point of face"""
        x, y, w, h = face_coords
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)
