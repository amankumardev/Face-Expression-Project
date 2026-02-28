"""
FER-based Expression Classification Module
Uses the FER (Facial Emotion Recognition) library for fast, real-time emotion detection
"""

import cv2
import numpy as np
from fer import FER
from config import FER_MTCNN


class FERClassifier:
    """Expression classifier using FER library"""
    
    def __init__(self):
        """Initialize FER classifier"""
        self.model_loaded = False
        print("Initializing FER emotion detection model...")
        
        try:
            # Initialize FER detector
            # MTCNN is more accurate but slower, OpenCV is faster
            self.detector = FER(mtcnn=FER_MTCNN)
            self.model_loaded = True
            print(f"=> FER model loaded successfully (MTCNN: {FER_MTCNN})")
        except Exception as e:
            print(f"X Error loading FER model: {e}")
            self.detector = None
    
    def analyze_expression(self, face_img):
        """
        Analyze facial expression from face image
        
        Args:
            face_img: Face region image (BGR format from OpenCV)
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        try:
            # Validate input
            if face_img is None or face_img.size == 0:
                return 'neutral', 0.0
            
            if not self.model_loaded or self.detector is None:
                return 'neutral', 0.0
            
            # Convert BGR to RGB (FER expects RGB)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Detect emotions using FER
            result = self.detector.detect_emotions(face_rgb)
            
            # Check if any face detected
            if not result or len(result) == 0:
                return 'neutral', 0.0
            
            # Get the first face's emotions (FER returns list of faces)
            emotions = result[0]['emotions']
            
            # Find dominant emotion
            emotion, confidence = self._get_dominant_emotion(emotions)
            
            return emotion, confidence
            
        except Exception as e:
            # Return neutral on error
            if 'neutral' not in str(e).lower():
                print(f"FER analysis error: {e}")
            return 'neutral', 0.0
    
    def _get_dominant_emotion(self, emotion_dict):
        """
        Get the dominant emotion from emotion dictionary
        
        Args:
            emotion_dict: Dictionary of emotion scores
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        if not emotion_dict:
            return 'neutral', 0.0
        
        # Find emotion with highest score
        dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])
        emotion_label = dominant_emotion[0]
        confidence = dominant_emotion[1]
        
        return emotion_label, confidence
    
    def get_emotion_distribution(self, face_img):
        """
        Get full emotion distribution for a face
        
        Args:
            face_img: Face region image
            
        Returns:
            Dictionary of emotion scores
        """
        try:
            if face_img is None or face_img.size == 0:
                return {}
            
            if not self.model_loaded or self.detector is None:
                return {}
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            result = self.detector.detect_emotions(face_rgb)
            
            if not result or len(result) == 0:
                return {}
            
            return result[0]['emotions']
            
        except Exception as e:
            return {}
    
    def analyze_multiple_faces(self, frame):
        """
        Analyze expressions for all faces in a frame
        
        Args:
            frame: Full frame image (BGR format)
            
        Returns:
            List of dictionaries containing face box and emotions
        """
        try:
            if not self.model_loaded or self.detector is None:
                return []
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect all faces and their emotions
            results = self.detector.detect_emotions(frame_rgb)
            
            return results
            
        except Exception as e:
            print(f"Multi-face analysis error: {e}")
            return []
