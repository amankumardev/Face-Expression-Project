"""
AI Face Expression Detection System
Main application entry point - Supports both FER and DeepFace models
"""

import cv2
import time
import sys
from face_detector import FaceDetector
from emotion_detector import EmotionDetector  # New unified detector
from ui_overlay import UIOverlay
from utils import save_screenshot
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    FPS_TARGET, SKIP_FRAMES, MODEL_TYPE
)


class FaceExpressionApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        print("\n" + "=" * 50)
        print("  AI FACE EXPRESSION DETECTION SYSTEM")
        print("=" * 50)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()  # New unified detector
        self.ui_overlay = UIOverlay()
        
        # Initialize webcam
        self.camera = None
        self.frame_count = 0
        self.running = False
        
        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0
        
        print("\n=> Initialization complete!")
        print(f"=> Using {self.emotion_detector.get_model_type()} model for emotion detection")
        print("\n" + "-" * 50)
        print("KEYBOARD CONTROLS:")
        print("  'Q' - Quit application")
        print("  'S' - Save screenshot")
        print("=" * 50)
    
    def initialize_camera(self):
        """Initialize camera capture"""
        print(f"\nOpening camera (index: {CAMERA_INDEX})...")
        
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.camera.isOpened():
            print("ERROR: Could not open camera!")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        
        # Get actual properties
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera opened: {actual_width}x{actual_height}")
        
        return True
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with overlays
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Analyze expressions for each face
        faces_data = []
        
        for face_coords in faces:
            # Extract face region
            face_region = self.face_detector.extract_face_region(frame, face_coords)
            
            # Skip frame analysis for performance (process every Nth frame)
            if self.frame_count % SKIP_FRAMES == 0:
                # Analyze expression using unified detector
                emotion, confidence = self.emotion_detector.analyze_expression(face_region)
            else:
                # Use last known emotion
                emotion, confidence = 'neutral', 0.0
            
            faces_data.append((face_coords, emotion, confidence))
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.prev_time = current_time
        
        # Add UI overlay
        frame = self.ui_overlay.add_overlay(frame, faces_data, self.fps)
        
        return frame
    
    def handle_keyboard(self, key, frame):
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
            frame: Current frame
            
        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting application...")
            return False
        elif key == ord('s') or key == ord('S'):
            print("\nSaving screenshot...")
            save_screenshot(frame)
        
        return True
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        if not self.initialize_camera():
            return
        
        self.running = True
        
        print("\nApplication running...")
        print("Point your camera at a face to detect expressions!\n")
        
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    print("ERROR: Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Face Expression Detection', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard(key, processed_frame):
                        break
                
                # Increment frame counter
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        print("Application closed successfully")
        print("=" * 50)


def main():
    """Main entry point"""
    app = FaceExpressionApp()
    app.run()


if __name__ == "__main__":
    main()
