"""
UI Overlay Module for displaying detection results
"""

import cv2
import numpy as np
from config import (
    BOX_THICKNESS, FONT, FONT_SCALE, FONT_THICKNESS,
    TEXT_PADDING, EXPRESSION_COLORS, SHOW_FPS, SHOW_CONFIDENCE
)
from utils import draw_text_with_background, format_confidence


class UIOverlay:
    """Handle all UI drawing operations"""
    
    def __init__(self):
        """Initialize UI overlay"""
        self.font = FONT
        self.font_scale = FONT_SCALE
        self.font_thickness = FONT_THICKNESS
        print("UI overlay initialized")
    
    def draw_face_box(self, frame, face_coords, emotion, confidence):
        """
        Draw enhanced bounding box with glow effects and label for a detected face
        
        Args:
            frame: Frame to draw on
            face_coords: Face coordinates (x, y, w, h)
            emotion: Detected emotion label
            confidence: Confidence score (0-100)
        """
        x, y, w, h = face_coords
        
        # Get color for this emotion
        color = EXPRESSION_COLORS.get(emotion.lower(), (200, 200, 200))
        
        # Import glow effect setting
        from config import ENABLE_GLOW_EFFECT, GLOW_INTENSITY
        
        # Draw glow effect (layered rectangles with decreasing opacity)
        if ENABLE_GLOW_EFFECT:
            overlay = frame.copy()
            for i in range(3, 0, -1):
                thickness = BOX_THICKNESS + (i * 2)
                alpha = GLOW_INTENSITY / (i + 1)
                cv2.rectangle(
                    overlay,
                    (x - i, y - i),
                    (x + w + i, y + h + i),
                    color,
                    thickness
                )
            # Blend overlay with original frame
            frame[:] = cv2.addWeighted(frame, 1 - GLOW_INTENSITY * 0.3, overlay, GLOW_INTENSITY * 0.3, 0)
        
        # Draw main bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            BOX_THICKNESS
        )
        
        # Draw corner accents for a more professional look
        corner_length = min(30, w // 4, h // 4)
        accent_thickness = BOX_THICKNESS + 1
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, accent_thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, accent_thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, accent_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, accent_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, accent_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, accent_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, accent_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, accent_thickness)
        
        # Prepare label text
        if SHOW_CONFIDENCE:
            label = f"{emotion.upper()}: {format_confidence(confidence)}"
        else:
            label = emotion.upper()
        
        # Draw label with background (above or below box)
        label_y = y - 15 if y > 40 else y + h + 25
        
        draw_text_with_background(
            frame,
            label,
            (x, label_y),
            self.font,
            self.font_scale,
            (255, 255, 255),  # White text
            color,  # Background matches box color
            self.font_thickness,
            TEXT_PADDING
        )
        
        return frame
    
    def draw_multiple_faces(self, frame, faces_data):
        """
        Draw boxes for multiple faces
        
        Args:
            frame: Frame to draw on
            faces_data: List of tuples (face_coords, emotion, confidence)
        """
        for face_coords, emotion, confidence in faces_data:
            self.draw_face_box(frame, face_coords, emotion, confidence)
        
        return frame
    
    def draw_fps(self, frame, fps):
        """Draw enhanced FPS counter with color coding"""
        if not SHOW_FPS:
            return frame
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Color code based on FPS (green = good, yellow = ok, red = poor)
        if fps >= 25:
            text_color = (0, 255, 0)  # Green
        elif fps >= 15:
            text_color = (0, 255, 255)  # Yellow
        else:
            text_color = (0, 100, 255)  # Red
        
        draw_text_with_background(
            frame,
            fps_text,
            (10, 30),
            self.font,
            0.7,
            text_color,
            (30, 30, 30),  # Dark gray background
            2,
            8
        )
        
        return frame
    
    def draw_face_count(self, frame, count):
        """Draw number of faces detected with enhanced styling"""
        if count == 0:
            count_text = "No faces detected"
            text_color = (100, 100, 255)  # Red
        elif count == 1:
            count_text = "1 Face"
            text_color = (100, 255, 100)  # Green
        else:
            count_text = f"{count} Faces"
            text_color = (255, 200, 100)  # Cyan
        
        draw_text_with_background(
            frame,
            count_text,
            (10, 70),
            self.font,
            0.7,
            text_color,
            (30, 30, 30),  # Dark gray background
            2,
            8
        )
        
        return frame
    
    def draw_instructions(self, frame):
        """Draw keyboard instructions"""
        height = frame.shape[0]
        
        instructions = [
            "Press 'Q' to quit",
            "Press 'S' to screenshot"
        ]
        
        y_offset = height - 60
        
        for i, instruction in enumerate(instructions):
            draw_text_with_background(
                frame,
                instruction,
                (10, y_offset + (i * 30)),
                self.font,
                0.5,
                (255, 255, 255),  # White text
                (0, 0, 0),         # Black background
                1,
                5
            )
        
        return frame
    
    def draw_emotion_bar(self, frame, emotion_dict, position=(10, 100)):
        """
        Draw emotion distribution bar chart
        
        Args:
            frame: Frame to draw on
            emotion_dict: Dictionary of emotion scores
            position: Starting position (x, y)
        """
        if not emotion_dict:
            return frame
        
        x, y = position
        bar_width = 150
        bar_height = 15
        spacing = 20
        
        # Sort emotions by confidence
        sorted_emotions = sorted(
            emotion_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        for i, (emotion, score) in enumerate(sorted_emotions[:5]):  # Show top 5
            # Calculate bar fill
            fill_width = int((score / 100) * bar_width)
            
            # Get color
            color = EXPRESSION_COLORS.get(emotion.lower(), (255, 255, 255))
            
            # Draw background bar
            cv2.rectangle(
                frame,
                (x, y + (i * spacing)),
                (x + bar_width, y + bar_height + (i * spacing)),
                (50, 50, 50),
                -1
            )
            
            # Draw filled bar
            cv2.rectangle(
                frame,
                (x, y + (i * spacing)),
                (x + fill_width, y + bar_height + (i * spacing)),
                color,
                -1
            )
            
            # Draw label
            label = f"{emotion}: {score:.1f}%"
            cv2.putText(
                frame,
                label,
                (x + bar_width + 10, y + 12 + (i * spacing)),
                self.font,
                0.4,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def add_overlay(self, frame, faces_data, fps, show_instructions=True):
        """
        Add complete overlay to frame
        
        Args:
            frame: Frame to draw on
            faces_data: List of (face_coords, emotion, confidence) tuples
            fps: Current FPS
            show_instructions: Whether to show keyboard instructions
        """
        # Draw all face boxes
        self.draw_multiple_faces(frame, faces_data)
        
        # Draw FPS counter
        self.draw_fps(frame, fps)
        
        # Draw face count
        self.draw_face_count(frame, len(faces_data))
        
        # Draw instructions
        if show_instructions:
            self.draw_instructions(frame)
        
        return frame
