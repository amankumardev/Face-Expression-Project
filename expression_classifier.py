"""
Expression Classification Module using DeepFace
"""

import cv2
import numpy as np
from deepface import DeepFace
from config import (
    EXPRESSION_MODEL, EXPRESSION_BACKEND,
    EXPRESSION_ENFORCE_DETECTION, EXPRESSION_DETECTOR_BACKEND
)
from utils import get_dominant_emotion


class ExpressionClassifier:
    """Expression classifier using DeepFace library"""
    
    def __init__(self):
        """Initialize expression classifier"""
        self.model_loaded = False
        self.last_emotions = {}  # Cache for smoothing
        print("Expression classifier initialized")
        
        # Warm up the model by running a dummy prediction
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model to avoid first-frame delay"""
        try:
            # Create a dummy image
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            
            # Run analysis once to load the model
            DeepFace.analyze(
                dummy_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=EXPRESSION_DETECTOR_BACKEND,
                silent=True
            )
            
            self.model_loaded = True
            print("Expression model loaded and ready")
        except Exception as e:
            print(f"Model warmup encountered an issue - will load on first use")
            print(f"Error details: {str(e)[:200]}")
            self.model_loaded = True  # Continue anyway
    
    def analyze_expression(self, face_img):
        """
        Analyze facial expression from face image
        
        Args:
            face_img: Face region image (BGR format)
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        try:
            # Ensure face image is valid
            if face_img is None or face_img.size == 0:
                return 'neutral', 0.0
            
            # Resize if too small
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                face_img = cv2.resize(face_img, (48, 48))
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=EXPRESSION_ENFORCE_DETECTION,
                detector_backend=EXPRESSION_DETECTOR_BACKEND,
                silent=True
            )
            
            # Handle both single result and list of results
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotion dictionary
            emotion_dict = result.get('emotion', {})
            
            # Get dominant emotion
            emotion, confidence = get_dominant_emotion(emotion_dict)
            
            return emotion, confidence
            
        except Exception as e:
            # Return neutral on error
            if "Face could not be detected" in str(e):
                return 'neutral', 0.0
            else:
                print(f"Expression analysis error: {e}")
                return 'neutral', 0.0
    
    def analyze_multiple_faces(self, frame, face_regions):
        """
        Analyze expressions for multiple faces
        
        Args:
            frame: Original frame
            face_regions: List of face region images
            
        Returns:
            List of (emotion, confidence) tuples
        """
        results = []
        
        for face_region in face_regions:
            emotion, confidence = self.analyze_expression(face_region)
            results.append((emotion, confidence))
        
        return results
    
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
            
            # Resize if too small
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                face_img = cv2.resize(face_img, (48, 48))
            
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=EXPRESSION_ENFORCE_DETECTION,
                detector_backend=EXPRESSION_DETECTOR_BACKEND,
                silent=True
            )
            
            # Handle both single result and list of results
            if isinstance(result, list):
                result = result[0]
            
            return result.get('emotion', {})
            
        except Exception as e:
            return {}
