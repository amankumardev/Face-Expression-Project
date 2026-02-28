"""
Unified Emotion Detector that supports both FER and DeepFace models
This module provides a simple interface to switch between different emotion detection backends
"""

from config import MODEL_TYPE


class EmotionDetector:
    """
    Unified emotion detector that automatically selects the appropriate model
    based on configuration
    """
    
    def __init__(self):
        """Initialize the emotion detector based on MODEL_TYPE setting"""
        print("\n" + "=" * 60)
        print("EMOTION DETECTION SYSTEM INITIALIZATION")
        print("=" * 60)
        print(f"Selected Model: {MODEL_TYPE}")
        print("-" * 60)
        
        self.model_type = MODEL_TYPE.upper()
        self.detector = None
        
        if self.model_type == 'FER':
            self._initialize_fer()
        elif self.model_type == 'DEEPFACE':
            self._initialize_deepface()
        else:
            print(f"⚠ Warning: Unknown MODEL_TYPE '{MODEL_TYPE}', defaulting to FER")
            self._initialize_fer()
        
        print("=" * 60 + "\n")
    
    def _initialize_fer(self):
        """Initialize FER model"""
        try:
            from fer_classifier import FERClassifier
            self.detector = FERClassifier()
            print("=> FER model ready for real-time detection")
        except ImportError:
            print("X FER library not installed. Install with: pip install fer")
            print("  Falling back to DeepFace...")
            self._initialize_deepface()
        except Exception as e:
            print(f"X Error initializing FER: {e}")
            print("  Falling back to DeepFace...")
            self._initialize_deepface()
    
    def _initialize_deepface(self):
        """Initialize DeepFace model"""
        try:
            from expression_classifier import ExpressionClassifier
            self.detector = ExpressionClassifier()
            print("=> DeepFace model ready for emotion analysis")
        except ImportError:
            print("X DeepFace library not installed. Install with: pip install deepface")
            raise RuntimeError("No emotion detection models available!")
        except Exception as e:
            print(f"X Error initializing DeepFace: {e}")
            raise
    
    def analyze_expression(self, face_img):
        """
        Analyze facial expression from face image
        
        Args:
            face_img: Face region image (BGR format)
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        if self.detector is None:
            return 'neutral', 0.0
        
        return self.detector.analyze_expression(face_img)
    
    def get_emotion_distribution(self, face_img):
        """
        Get full emotion distribution for a face
        
        Args:
            face_img: Face region image
            
        Returns:
            Dictionary of emotion scores
        """
        if self.detector is None:
            return {}
        
        return self.detector.get_emotion_distribution(face_img)
    
    def get_model_type(self):
        """
        Get the current model type being used
        
        Returns:
            String indicating the model type ('FER' or 'DEEPFACE')
        """
        return self.model_type
