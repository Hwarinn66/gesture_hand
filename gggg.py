import cv2
import mediapipe as mp
import numpy as np
import webbrowser
import time
import threading
import platform
import os
import sys

class HandGestureNavigator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Meningkatkan confidence threshold
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Definisi Website dengan gestur yang lebih jelas
        self.websites = {
            "peace": "https://www.youtube.com",          # Jari telunjuk dan tengah membentuk V
            "rock": "https://www.spotify.com",           # Telunjuk dan kelingking terangkat
            "thumbs_up": "https://www.google.com",       # Hanya jempol terangkat
            "open_palm": "https://www.facebook.com",     # Semua jari terbuka
            "twitter_gesture": "https://www.twitter.com", # Jempol dan kelingking
            "pinterest_gesture": "https://www.pinterest.com", # Jempol, telunjuk, dan kelingking
            "gmail_gesture": "https://mail.google.com",  # Jempol dan telunjuk membentuk L
            "drive_gesture": "https://drive.google.com"  # Jempol dan telunjuk
        }
        
        self.GESTURE_HOLD_TIME = 1.0  # Meningkatkan waktu tahan
        self.GESTURE_COOLDOWN = 2.0
        self.last_gesture = None
        self.gesture_start_time = 0
        self.last_website_time = 0
        
        # Tambahan untuk visualisasi debugging
        self.debug_info = []

    def calculate_finger_angles(self, landmarks):
        """Menghitung sudut untuk setiap jari"""
        angles = {}
        finger_tips = {
            "thumb": [4, 3, 2],
            "index": [8, 7, 6],
            "middle": [12, 11, 10],
            "ring": [16, 15, 14],
            "pinky": [20, 19, 18]
        }
        
        for finger, [tip, mid, base] in finger_tips.items():
            tip_pos = np.array([landmarks[tip].x, landmarks[tip].y])
            mid_pos = np.array([landmarks[mid].x, landmarks[mid].y])
            base_pos = np.array([landmarks[base].x, landmarks[base].y])
            
            vec1 = tip_pos - mid_pos
            vec2 = base_pos - mid_pos
            
            angle = np.degrees(np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2)))
            angles[finger] = abs(angle)
            
        return angles

    def is_finger_open(self, angle, vertical_dist):
        """Pengecekan jari terbuka dengan threshold yang disesuaikan"""
        return angle > 160 and vertical_dist > 0.04

    def detect_hand_gesture(self, hand_landmarks):
        try:
            landmarks = hand_landmarks.landmark
            angles = self.calculate_finger_angles(landmarks)
            
            # Deteksi status jari dengan threshold yang lebih ketat
            finger_states = {}
            for finger, points in {
                "thumb": (4, 3, 2),
                "index": (8, 7, 6),
                "middle": (12, 11, 10),
                "ring": (16, 15, 14),
                "pinky": (20, 19, 18)
            }.items():
                tip, mid, base = points
                vertical_dist = abs(landmarks[tip].y - landmarks[base].y)
                finger_states[finger] = self.is_finger_open(angles[finger], vertical_dist)
            
            # Update debug info
            self.debug_info = [f"{finger}: {'✓' if state else '✗'}" 
                             for finger, state in finger_states.items()]
            
            # Definisi gestur yang lebih presisi
            gestures = {
                "peace": (
                    finger_states["index"] and 
                    finger_states["middle"] and 
                    not any([finger_states[f] for f in ["thumb", "ring", "pinky"]])
                ),
                "rock": (
                    finger_states["index"] and 
                    finger_states["pinky"] and 
                    not any([finger_states[f] for f in ["thumb", "middle", "ring"]])
                ),
                "thumbs_up": (
                    finger_states["thumb"] and 
                    not any([finger_states[f] for f in ["index", "middle", "ring", "pinky"]])
                ),
                "open_palm": all(finger_states.values()),
                "twitter_gesture": (
                    finger_states["thumb"] and 
                    finger_states["pinky"] and 
                    not any([finger_states[f] for f in ["index", "middle", "ring"]])
                ),
                "pinterest_gesture": (
                    finger_states["thumb"] and 
                    finger_states["index"] and 
                    finger_states["pinky"] and 
                    not finger_states["middle"] and 
                    not finger_states["ring"]
                ),
                "gmail_gesture": (
                    finger_states["thumb"] and 
                    finger_states["index"] and 
                    not any([finger_states[f] for f in ["middle", "ring", "pinky"]])
                ),
                "drive_gesture": (
                    finger_states["thumb"] and 
                    finger_states["index"] and 
                    not any([finger_states[f] for f in ["middle", "ring", "pinky"]])
                )
            }
            
            # Deteksi gestur aktif
            active_gestures = [gesture for gesture, is_active in gestures.items() 
                             if is_active]
            
            return active_gestures[0] if active_gestures else None
            
        except Exception as e:
            print(f"Gesture Detection Error: {e}")
            return None

    def start_tracking(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            try:
                success, frame = cap.read()
                if not success:
                    print("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Tambahkan visualisasi status
                self._draw_gesture_guide(frame)
                
                current_time = time.time()
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Gambar landmark dengan style yang lebih jelas
                        self.mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        gesture = self.detect_hand_gesture(hand_landmarks)
                        
                        if gesture:
                            if gesture != self.last_gesture:
                                self.last_gesture = gesture
                                self.gesture_start_time = current_time
                            
                            # Visualisasi progress
                            hold_duration = current_time - self.gesture_start_time
                            if hold_duration < self.GESTURE_HOLD_TIME:
                                progress = int((hold_duration / self.GESTURE_HOLD_TIME) * 100)
                                self._draw_progress_bar(frame, progress)
                            
                            self.process_gesture(gesture, current_time)
                            
                            # Tampilkan gesture terdeteksi
                            cv2.putText(frame, 
                                      f"Gesture: {gesture.replace('_', ' ').title()}", 
                                      (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2)
                
                # Tampilkan debug info
                for i, info in enumerate(self.debug_info):
                    cv2.putText(frame, info, (10, 60 + i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Hand Gesture Navigator', frame)
                
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                    
            except Exception as e:
                print(f"Tracking Error: {e}")
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def _draw_gesture_guide(self, frame):
        """Menampilkan panduan gestur di layar"""
        height, width = frame.shape[:2]
        guide_text = [
            "Gestur:",
            "- Peace (V) = YouTube",
            "- Rock = Spotify",
            "- Jempol = Google",
            "- Telapak = Facebook",
            "- Jempol+Kelingking = Twitter",
            "- Jempol+Telunjuk+Kelingking = Pinterest",
            "- L Shape = Gmail/Drive"
        ]
        
        for i, text in enumerate(guide_text):
            cv2.putText(frame, text, 
                       (width - 300, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)

    def _draw_progress_bar(self, frame, progress):
        """Menampilkan progress bar untuk waktu tahan gestur"""
        height, width = frame.shape[:2]
        bar_width = 200
        bar_height = 20
        x = (width - bar_width) // 2
        y = height - 50
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     (100, 100, 100), -1)
        # Progress
        progress_width = int((progress / 100) * bar_width)
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height),
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(frame, f"{progress}%", 
                   (x + bar_width + 10, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)

    def process_gesture(self, gesture, current_time):
        # Implementasi process_gesture tetap sama
        if not gesture or gesture not in self.websites:
            return
            
        gesture_duration = current_time - self.gesture_start_time
        time_since_last_open = current_time - self.last_website_time
            
        if (gesture_duration >= self.GESTURE_HOLD_TIME and 
            time_since_last_open > self.GESTURE_COOLDOWN):
            threading.Thread(
                target=self._open_website,
                args=(gesture,),
                daemon=True
            ).start()
            self.last_website_time = current_time

    def _open_website(self, gesture):
        # Implementasi _open_website tetap sama
        try:
            website = self.websites.get(gesture)
            if website:
                webbrowser.open(website)
        except Exception as e:
            print(f"Error opening website: {e}")

def main():
    try:
        print("\n=== Hand Gesture Web Navigator ===")
        print("\nPanduan Penggunaan:")
        print("1. Posisikan tangan Anda di depan kamera")
        print("2. Buat gestur sesuai dengan website yang diinginkan:")
        print("   - Peace (V) = YouTube")
        print("   - Rock = Spotify")
        print("   - Jempol = Google")
        print("   - Telapak Terbuka = Facebook")
        print("   - Jempol + Kelingking = Twitter")
        print("   - Jempol + Telunjuk + Kelingking = Pinterest")
        print("   - Bentuk L = Gmail/Drive")
        print("3. Tahan gestur selama 1 detik")
        print("4. Tunggu cooldown 2 detik sebelum gestur berikutnya")
        print("\nTekan 'q' untuk keluar\n")
        
        navigator = HandGestureNavigator()
        navigator.start_tracking()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()