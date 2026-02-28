"""
Real-time Facial Emotion Recognition System
Quick start script with FER model (optimized for speed)
"""

import cv2
import time
import numpy as np
from fer import FER


def main():
    """Quick demo using FER library"""
    print("\n" + "=" * 60)
    print("  FACIAL EMOTION RECOGNITION - QUICK START")
    print("=" * 60)
    print("\nInitializing FER model...")
    
    # Initialize FER detector
    detector = FER(mtcnn=False)  # Use OpenCV for speed
    print("✓ FER model loaded\n")
    
    # Emotion colors (BGR)
    COLORS = {
        'angry': (0, 50, 255),      # Red
        'disgust': (0, 140, 140),   # Teal
        'fear': (180, 100, 180),    # Purple
        'happy': (30, 255, 30),     # Green
        'sad': (255, 100, 50),      # Blue
        'surprise': (0, 255, 255),  # Yellow
        'neutral': (200, 200, 200)  # Gray
    }
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    print("✓ Webcam opened\n")
    print("-" * 60)
    print("CONTROLS:")
    print("  'Q' - Quit")
    print("  'S' - Screenshot")
    print("=" * 60 + "\n")
    
    # FPS tracking
    prev_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze emotions every 2 frames for better performance
            if frame_count % 2 == 0:
                result = detector.detect_emotions(frame)
            
            # Draw results
            if result:
                for face_data in result:
                    # Get bounding box
                    box = face_data['box']
                    x, y, w, h = box
                    
                    # Get dominant emotion
                    emotions = face_data['emotions']
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    emotion_name = dominant_emotion[0]
                    confidence = dominant_emotion[1] * 100
                    
                    # Get color
                    color = COLORS.get(emotion_name, (255, 255, 255))
                    
                    # Draw bounding box with glow effect
                    for i in range(2, 0, -1):
                        alpha = 0.3 / i
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x-i, y-i), (x+w+i, y+h+i), color, 2+i)
                        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
                    
                    # Main box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Corner accents
                    corner_len = 25
                    cv2.line(frame, (x, y), (x+corner_len, y), color, 4)
                    cv2.line(frame, (x, y), (x, y+corner_len), color, 4)
                    cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, 4)
                    cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, 4)
                    cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, 4)
                    cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, 4)
                    cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, 4)
                    cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, 4)
                    
                    # Label
                    label = f"{emotion_name.upper()}: {confidence:.1f}%"
                    label_y = y - 15 if y > 40 else y + h + 30
                    
                    # Text background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x-5, label_y-text_h-10), (x+text_w+10, label_y+5), color, -1)
                    
                    # Text
                    cv2.putText(frame, label, (x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Draw FPS
            fps_color = (0, 255, 0) if fps >= 25 else (0, 255, 255) if fps >= 15 else (0, 100, 255)
            cv2.rectangle(frame, (5, 5), (150, 45), (30, 30, 30), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            
            # Display
            cv2.imshow('Facial Emotion Recognition - FER', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"emotion_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Application closed successfully")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
