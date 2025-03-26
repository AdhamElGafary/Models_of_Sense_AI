"""
# MultiTaskVideoAnalyzer: Advanced Video and Emotion Analysis Tool

## Overview
This Python script is a sophisticated video recording and analysis application that leverages multiple computer vision and machine learning libraries to perform real-time facial analysis, emotion detection, and recording.

## Key Features
1. **Multi-Modal Recording**
   - Simultaneously captures video and audio
   - Saves recordings with timestamped filenames
   - Supports manual recording start and stop

2. **Facial Analysis**
   - Uses MediaPipe for facial landmark detection
   - Tracks and visualizes facial landmarks in real-time
   - Overlays detected landmarks on video frames

3. **Emotion Recognition**
   - Employs DeepFace library for emotion analysis
   - Detects and displays dominant emotion during recording
   - Calculates emotion confidence levels

4. **Data Visualization**
   - Generates comprehensive analysis reports
   - Creates pie charts showing emotion distribution
   - Converts video recordings from AVI to MP4 format

## Technical Components
- Libraries Used:
  - OpenCV (cv2) for video processing
  - MediaPipe for facial landmark detection
  - DeepFace for emotion recognition
  - SoundDevice and SoundFile for audio recording
  - Matplotlib for data visualization
  - Torch and Transformers for potential future machine learning tasks

## Workflow
1. Initialize video and audio recording
2. Capture video frames
3. Detect facial landmarks
4. Analyze emotions in real-time
5. Overlay emotion and landmark information
6. Record video and audio
7. Generate analysis report and visualization
8. Convert and save recordings

## Unique Aspects
- Supports manual recording control
- Provides multi-dimensional analysis of facial expressions
- Creates informative visual reports
- Flexible and extensible architecture
"""
import cv2
import numpy as np
import mediapipe as mp
import dlib
import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from deepface import DeepFace
import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime
import matplotlib.pyplot as plt

class MultiTaskVideoAnalyzer:
    def __init__(self, output_dir='recordings'):
        # Preparing the analysis components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            min_detection_confidence=0.5
        )
        
        # Setting up audio and video recording
        self.video_writer = None
        self.audio_recorder = None
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.is_recording = False

    def _draw_facial_landmarks(self, frame, landmarks):
        """Draw facial landmarks on the frame"""
        for landmark in landmarks:
            cv2.circle(frame, 
                       (int(landmark[0]), int(landmark[1])), 
                       2, (0, 255, 0), -1)
        return frame

    def _overlay_emotion_info(self, frame, emotion, confidence):
        """Overlay emotion and confidence information on the frame"""
        # Background rectangle for text
        cv2.rectangle(frame, (10, 30), (frame.shape[1]-10, 70), 
                      (255, 255, 255), -1)
        
        # Emotion text
        emotion_text = f"Emotion: {emotion}"
        cv2.putText(frame, emotion_text, 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 2)
        
        # Confidence text
        confidence_text = f"Confidence: {confidence:.2f}%"
        cv2.putText(frame, confidence_text, 
                    (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 2)
        
        return frame

    def start_recording(self):
        """Start recording video and audio with the ability to stop manually"""
        # Setting the file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f'video_{timestamp}.avi')
        audio_path = os.path.join(self.output_dir, f'audio_{timestamp}.wav')

        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Make sure the camera is open
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return None, None
        
        # Video writer prepared
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, 20.0, 
            (frame_width, frame_height)
        )

        # Lists for storing data
        audio_frames = []
        emotion_results = []
        facial_landmarks = []

        # Preparing the audio recording
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                audio_frames.append(indata.copy())

        # Registration starts
        self.is_recording = True
        print("Recording has started. Press 'e' to stop.")

        # Start recording audio
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    break

                # Facial expression analysis
                try:
                    analysis = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        enforce_detection=False
                    )
                    emotion = analysis[0]['dominant_emotion']
                    emotion_confidence = analysis[0]['emotion'][emotion] * 100
                    emotion_results.append(emotion)
                except Exception as e:
                    emotion = 'unknown'
                    emotion_confidence = 0
                    emotion_results.append('unknown')

                # Facial feature extraction
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                # Draw facial landmarks if detected
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [(int(landmark.x * frame_width), 
                                  int(landmark.y * frame_height)) 
                                 for landmark in face_landmarks.landmark]
                    
                    # Draw landmarks on frame
                    frame = self._draw_facial_landmarks(frame, landmarks)
                    facial_landmarks.append(landmarks)
                
                # Overlay emotion information
                frame = self._overlay_emotion_info(frame, emotion, emotion_confidence)

                # Frame recording
                self.video_writer.write(frame)
                
                # Frame view
                cv2.imshow('Video Recording', frame)
                
                # Check the stop button
                key = cv2.waitKey(1) & 0xFF
                if key == ord('e'):
                    print("Recording is being paused...")
                    self.is_recording = False
                    break

        # Stop recording
        cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

        # Save audio
        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=0)
            sf.write(audio_path, audio_data, 44100)
        else:
            print("No sound was recorded.")
            audio_path = None

        # Convert AVI to MP4
        video_path = self._convert_avi_to_mp4(video_path)

        # Create an analysis report
        report_path = self._generate_analysis_report(
            video_path, 
            audio_path, 
            emotion_results, 
            facial_landmarks
        )

        # Visualize emotion analysis
        self._visualize_emotion_analysis(report_path)

        return video_path, audio_path

    # باقي الدوال تظل كما هي دون تغيير
    def _convert_avi_to_mp4(self, avi_path):
        """Convert AVI video to MP4"""
        # Create MP4 output path
        mp4_path = avi_path.replace('.avi', '.mp4')
        
        # Read the AVI video
        cap = cv2.VideoCapture(avi_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create VideoWriter object for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        
        # Read and write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Optional: Remove the original AVI file
        os.remove(avi_path)
        
        return mp4_path

    # باقي الدوال تظل كما هي (دوال التقرير والتصور)
    def _generate_analysis_report(self, video_path, audio_path, emotions, landmarks):
        """Create an analysis report"""
        report_path = os.path.join(
            self.output_dir, 
            f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        
        with open(report_path, 'w') as f:
            f.write("Video Analysis Report\n")
            f.write("====================\n\n")
            f.write(f"Video File: {video_path}\n")
            f.write(f"Audio File: {audio_path}\n\n")
            
            f.write("Emotion Analysis:\n")
            emotion_summary = {}
            for emotion in emotions:
                emotion_summary[emotion] = emotion_summary.get(emotion, 0) + 1
            
            for emotion, count in emotion_summary.items():
                f.write(f"- {emotion}: {count} times\n")
            
            f.write(f"\nTotal Facial Landmarks Tracked: {len(landmarks)}")
        
        return report_path

    def _visualize_emotion_analysis(self, report_path):
        """Create a pie chart of emotion distribution"""
        # Read the analysis report
        with open(report_path, 'r') as f:
            lines = f.readlines()
        
        # Extract emotion data
        emotions = {}
        for line in lines:
            if ':' in line and 'times' in line:
                emotion, count = line.split(':')
                emotion = emotion.strip()
                count = int(count.split()[0])
                emotions[emotion] = count
        
        # Create a pie chart
        plt.figure(figsize=(10, 7))
        plt.pie(
            emotions.values(), 
            labels=emotions.keys(), 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=plt.cm.Pastel1.colors
        )
        plt.title('Emotion Distribution During Recording', fontsize=15)
        plt.axis('equal')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f'emotion_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

if __name__ == "__main__":
    analyzer = MultiTaskVideoAnalyzer()
    video_file, audio_file = analyzer.start_recording()
    print(f"Recording completed. Files saved:\n{video_file}\n{audio_file}")

