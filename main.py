import cv2
import pygame
import os
import random
import numpy as np
from collections import Counter
from deepface import DeepFace
import time
import math

# Initialize pygame mixer
pygame.mixer.init()

# Configuration
MUSIC_FOLDER = "music"
WINDOW_NAME = "Emotion Music Player"
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Initialize camera
cap = cv2.VideoCapture(0)

# Global variables
current_song = None
current_emotion = None
previous_emotion = None
song_list = []
song_index = 0
is_playing = False
camera_mode = True
music_mode = False
detection_active = True
emotion_stability_counter = 0
EMOTION_STABILITY_THRESHOLD = 8
last_emotion_time = time.time()
song_start_time = 0
song_paused = False
particles = []
rotation_angle = 0  # For animation

# Emotion colors for display
EMOTION_COLORS = {
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'angry': (0, 0, 255),      # Red
    'surprise': (255, 165, 0), # Orange
    'neutral': (200, 200, 200),# Gray
    'fear': (255, 0, 255),     # Purple
    'disgust': (0, 255, 0)     # Green
}

# Emotion emojis
EMOTION_EMOJIS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'neutral': '😐',
    'fear': '😨',
    'disgust': '🤢'
}

# Create window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)

class Particle:
    """For visual effects during music playback"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = 1.0
        self.color = color
        self.size = random.randint(2, 6)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.01
        return self.life > 0
    
    def draw(self, frame):
        if self.life > 0:
            alpha = int(255 * self.life)
            color = tuple(int(c * self.life) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, color, -1)

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness, radius=20):
    """Draw rectangle with rounded corners"""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Draw rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw circles for corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_filled_rounded_rect(img, top_left, bottom_right, color, alpha, radius=20):
    """Draw filled rectangle with rounded corners and transparency"""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Create overlay
    overlay = img.copy()
    
    # Draw filled rectangle
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    # Draw circles for corners
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    
    # Blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_music_button(img, center, radius, icon_type, color, is_active=True, rotation=0):
    """Draw a circular music control button"""
    x, y = center
    
    # Draw outer glow if active
    if is_active:
        for i in range(3, 0, -1):
            alpha = i * 0.3
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.circle(img, (x, y), radius + i*2, glow_color, 2)
    
    # Draw main circle
    cv2.circle(img, (x, y), radius, (50, 50, 50), -1)
    cv2.circle(img, (x, y), radius, color, 3)
    
    # Draw icon based on type
    if icon_type == "play":
        # Play triangle
        points = np.array([
            [x - 15, y - 15],
            [x - 15, y + 15],
            [x + 15, y]
        ], np.int32)
        cv2.fillPoly(img, [points], (255, 255, 255))
        
    elif icon_type == "pause":
        # Pause bars
        cv2.rectangle(img, (x - 15, y - 20), (x - 5, y + 20), (255, 255, 255), -1)
        cv2.rectangle(img, (x + 5, y - 20), (x + 15, y + 20), (255, 255, 255), -1)
        
    elif icon_type == "next":
        # Next (double right arrow)
        points1 = np.array([
            [x - 10, y - 15],
            [x - 10, y + 15],
            [x + 10, y]
        ], np.int32)
        points2 = np.array([
            [x + 5, y - 15],
            [x + 5, y + 15],
            [x + 25, y]
        ], np.int32)
        cv2.fillPoly(img, [points1], (255, 255, 255))
        cv2.fillPoly(img, [points2], (255, 255, 255))
        
    elif icon_type == "prev":
        # Previous (double left arrow)
        points1 = np.array([
            [x + 10, y - 15],
            [x + 10, y + 15],
            [x - 10, y]
        ], np.int32)
        points2 = np.array([
            [x - 5, y - 15],
            [x - 5, y + 15],
            [x - 25, y]
        ], np.int32)
        cv2.fillPoly(img, [points1], (255, 255, 255))
        cv2.fillPoly(img, [points2], (255, 255, 255))
        
    elif icon_type == "stop":
        # Stop square
        cv2.rectangle(img, (x - 15, y - 15), (x + 15, y + 15), (255, 255, 255), -1)

def get_song_list(emotion):
    """Get all songs for a specific emotion folder"""
    path = os.path.join(MUSIC_FOLDER, emotion)
    
    if not os.path.exists(path):
        print(f"⚠️ Folder not found: {path}")
        return []
    
    # Get all audio files
    songs = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            # Check all common audio formats
            if file.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.opus', '.aac')):
                songs.append(file_path)
            # Also try files without clear extension (WhatsApp files)
            elif '.' not in file or os.path.getsize(file_path) > 1000:  # Files > 1KB
                songs.append(file_path)
    
    if songs:
        print(f"✅ Found {len(songs)} songs for '{emotion}'")
        random.shuffle(songs)
    else:
        print(f"❌ No songs found for '{emotion}'")
    
    return songs

def play_song():
    """Play the current song"""
    global current_song, is_playing, song_start_time, detection_active, song_paused, particles
    
    if song_list and song_index < len(song_list):
        try:
            current_song = song_list[song_index]
            print(f"\n🎵 Playing: {os.path.basename(current_song)}")
            
            if os.path.exists(current_song):
                pygame.mixer.music.load(current_song)
                pygame.mixer.music.play()
                is_playing = True
                song_paused = False
                song_start_time = time.time()
                detection_active = False
                
                # Create particles for visual effect
                particles = []
                for _ in range(100):
                    x = random.randint(0, SCREEN_WIDTH)
                    y = random.randint(0, SCREEN_HEIGHT)
                    color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
                    particles.append(Particle(x, y, color))
                
                print("🔴 MUSIC PLAYING - Camera hidden")
            else:
                print(f"❌ File not found, trying next...")
                next_song()
                
        except Exception as e:
            print(f"❌ Error playing song: {e}")
            next_song()

def next_song():
    """Play next song in the list"""
    global song_index, is_playing, detection_active
    
    if song_list:
        song_index = (song_index + 1) % len(song_list)
        print(f"\n⏭️ Next song...")
        play_song()
    else:
        print("❌ No songs available")
        detection_active = True

def prev_song():
    """Play previous song in the list"""
    global song_index, is_playing, detection_active
    
    if song_list:
        song_index = (song_index - 1) % len(song_list)
        print(f"\n⏮️ Previous song...")
        play_song()
    else:
        print("❌ No songs available")
        detection_active = True

def stop_music():
    """Stop music and show camera"""
    global is_playing, music_mode, camera_mode, emotion_stability_counter, detection_active, song_paused, particles
    pygame.mixer.music.stop()
    is_playing = False
    song_paused = False
    music_mode = False
    camera_mode = True
    emotion_stability_counter = 0
    detection_active = True
    particles = []
    print("🔵 CAMERA SHOWING - Ready for new emotion")

def pause_music():
    """Pause/unpause music"""
    global is_playing, song_paused
    
    if is_playing:
        pygame.mixer.music.pause()
        is_playing = False
        song_paused = True
        print("\n⏸️ Music paused")
    else:
        pygame.mixer.music.unpause()
        is_playing = True
        song_paused = False
        print("\n▶️ Music resumed")

def detect_emotion(frame):
    """Detect emotions in frame using DeepFace - IMPROVED for multiple faces"""
    try:
        # Resize for faster processing
        small_frame = cv2.resize(frame, (640, 480))
        
        # Analyze face for emotion
        results = DeepFace.analyze(
            img_path=small_frame,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        
        if not results:
            return None, frame, []
        
        # Handle both single and multiple results
        if isinstance(results, dict):
            results = [results]
        
        emotions = []
        processed_frame = frame.copy()
        face_regions = []
        
        for i, face in enumerate(results):
            region = face.get("region", {})
            
            # Scale coordinates back to original frame size
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            x = int(region.get("x", 0) * scale_x)
            y = int(region.get("y", 0) * scale_y)
            w = int(region.get("w", 0) * scale_x)
            h = int(region.get("h", 0) * scale_y)
            
            emotion = face.get("dominant_emotion", "unknown")
            confidence = face.get("emotion", {}).get(emotion, 0)
            
            emotions.append(emotion)
            face_regions.append((x, y, w, h, emotion, confidence))
            
            # Draw face rectangle with emotion color
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw emotion label with background
            label = f"{EMOTION_EMOJIS.get(emotion, '')} {emotion} ({int(confidence)}%)"
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(processed_frame, (x, y-35), (x+label_w+10, y-5), (0, 0, 0), -1)
            cv2.rectangle(processed_frame, (x, y-35), (x+label_w+10, y-5), color, 1)
            
            # Label text
            cv2.putText(processed_frame, label, 
                       (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Face number
            cv2.putText(processed_frame, f"Face #{i+1}", 
                       (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Determine dominant emotion
        if emotions:
            emotion_counts = Counter(emotions)
            dominant_emotion, count = emotion_counts.most_common(1)[0]
            return dominant_emotion, processed_frame, face_regions
        
        return None, processed_frame, []
        
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
        return None, frame, []

def check_emotion_change(new_emotion):
    """Check if emotion has changed and is stable"""
    global current_emotion, emotion_stability_counter, previous_emotion, last_emotion_time
    
    if not new_emotion:
        return False
    
    # Reset counter if emotion changed
    if previous_emotion != new_emotion:
        emotion_stability_counter = 1
        previous_emotion = new_emotion
        print(f"🔍 New emotion detected: {new_emotion}")
    else:
        emotion_stability_counter += 1
    
    # Change music if emotion is stable AND no music playing
    if (emotion_stability_counter >= EMOTION_STABILITY_THRESHOLD and 
        new_emotion != current_emotion and not is_playing and not song_paused):
        
        last_emotion_time = time.time()
        emotion_stability_counter = 0
        print(f"✅ Emotion '{new_emotion}' stabilized! Starting music...")
        return new_emotion
    
    return False

def draw_camera_interface(frame, emotion, face_regions, detection_active, is_playing):
    """Draw camera interface - FACE FULLY VISIBLE, minimal overlays"""
    display = frame.copy()
    h, w = display.shape[:2]
    
    # === MINIMAL TOP BAR (Semi-transparent) ===
    draw_filled_rounded_rect(display, (0, 0), (w, 50), (0, 0, 0), 0.5, radius=10)
    
    # Status
    if detection_active:
        status_text = "🔵 FACE DETECTION ACTIVE"
        status_color = (0, 255, 0)
    else:
        status_text = "🔴 MUSIC PLAYING - Press S to stop"
        status_color = (0, 0, 255)
    
    cv2.putText(display, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Face count
    face_count = len(face_regions) if face_regions else 0
    cv2.putText(display, f"👥 {face_count} face(s)", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # === EMOTION INFO (Only when detection active) ===
    if detection_active and emotion:
        # Emotion badge at top right
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        emoji = EMOTION_EMOJIS.get(emotion, '')
        
        # Badge background
        badge_text = f"{emoji} {emotion.upper()}"
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        draw_filled_rounded_rect(display, (w-text_w-40, 60), (w-20, 100), (0, 0, 0), 0.7, radius=10)
        draw_rounded_rectangle(display, (w-text_w-40, 60), (w-20, 100), color, 2, radius=10)
        
        cv2.putText(display, badge_text, (w-text_w-30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Stability bar
        bar_x = w - 220
        bar_y = 110
        bar_width = 180
        progress = int((emotion_stability_counter / EMOTION_STABILITY_THRESHOLD) * bar_width)
        
        cv2.putText(display, "Stability:", (bar_x, bar_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        draw_filled_rounded_rect(display, (bar_x, bar_y), (bar_x+bar_width, bar_y+10), (100, 100, 100), 0.5, radius=3)
        draw_filled_rounded_rect(display, (bar_x, bar_y), (bar_x+progress, bar_y+10), color, 0.8, radius=3)
    
    # === MINIMAL CONTROLS (Bottom, small) ===
    draw_filled_rounded_rect(display, (10, h-90), (200, h-10), (0, 0, 0), 0.4, radius=15)
    
    controls = [
        "🎮 CONTROLS",
        "S - Stop music",
        "M - Music mode",
        "ESC - Exit"
    ]
    
    for i, text in enumerate(controls):
        y = h-70 + i*20
        color = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(display, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return display

def draw_music_visualizer(frame, emotion):
    """Draw full-screen music visualizer with animated round buttons"""
    global rotation_angle
    h, w = frame.shape[:2]
    display = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get emotion color
    main_color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    emoji = EMOTION_EMOJIS.get(emotion, '🎵')
    
    # Create gradient background
    for i in range(h):
        alpha = i / h
        color = tuple(int(c * (0.2 + alpha * 0.3)) for c in main_color)
        display[i, :] = color
    
    # Update rotation angle for animation
    rotation_angle += 2
    
    # Update particles
    global particles
    particles = [p for p in particles if p.update()]
    
    # Add new particles
    if len(particles) < 150:
        for _ in range(3):
            x = random.randint(0, w)
            y = random.randint(0, h)
            particles.append(Particle(x, y, main_color))
    
    # Draw particles
    for p in particles:
        p.draw(display)
    
    center_x, center_y = w // 2, h // 2
    
    # === ANIMATED ROUND CENTERPIECE ===
    # Rotating outer rings
    for i in range(3):
        radius = 180 + i * 20
        angle = rotation_angle + i * 30
        points = []
        for j in range(0, 360, 30):
            rad = math.radians(angle + j)
            x = int(center_x + radius * math.cos(rad))
            y = int(center_y + radius * math.sin(rad))
            points.append([x, y])
        
        for j in range(len(points)):
            cv2.line(display, tuple(points[j]), tuple(points[(j+1)%len(points)]), 
                    main_color, 2)
    
    # Main circle with glow
    for i in range(10, 0, -1):
        alpha = i / 10
        color = tuple(int(c * alpha) for c in main_color)
        cv2.circle(display, (center_x, center_y), 150 + i*3, color, 2)
    
    # Main circle
    cv2.circle(display, (center_x, center_y), 150, (255, 255, 255), 3)
    cv2.circle(display, (center_x, center_y), 145, main_color, 3)
    
    # Emotion emoji in center
    font_scale = 8.0
    (text_w, text_h), _ = cv2.getTextSize(emoji, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)
    cv2.putText(display, emoji, (center_x - text_w//2, center_y + text_h//2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, main_color, 5)
    
    # === MUSIC CONTROL BUTTONS (Round, below center) ===
    button_y = center_y + 250
    
    # Previous button
    draw_music_button(display, (center_x - 200, button_y), 40, "prev", main_color, True, rotation_angle)
    
    # Play/Pause button
    if is_playing:
        draw_music_button(display, (center_x, button_y), 50, "pause", main_color, True, rotation_angle)
    else:
        draw_music_button(display, (center_x, button_y), 50, "play", main_color, True, rotation_angle)
    
    # Next button
    draw_music_button(display, (center_x + 200, button_y), 40, "next", main_color, True, rotation_angle)
    
    # Stop button (smaller)
    draw_music_button(display, (center_x + 300, button_y - 50), 30, "stop", (0, 0, 255), True, rotation_angle)
    
    # Button labels
    cv2.putText(display, "PREV", (center_x - 215, button_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display, "PLAY/PAUSE", (center_x - 55, button_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display, "NEXT", (center_x + 185, button_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display, "STOP", (center_x + 285, button_y - 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # === NOW PLAYING INFO ===
    if current_song:
        song_name = os.path.basename(current_song)
        if len(song_name) > 40:
            song_name = song_name[:37] + "..."
        
        # Song title background
        draw_filled_rounded_rect(display, (center_x-300, center_y-350), (center_x+300, center_y-280), 
                                (0, 0, 0), 0.6, radius=20)
        
        cv2.putText(display, "🎵 NOW PLAYING", (center_x-150, center_y-320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, song_name, (center_x-280, center_y-280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2)
        
        # Time and progress
        elapsed = int(time.time() - song_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        # Progress bar background
        bar_width = 500
        bar_x = center_x - bar_width//2
        bar_y = center_y + 150
        
        draw_filled_rounded_rect(display, (bar_x-5, bar_y-5), (bar_x+bar_width+5, bar_y+25), 
                                (0, 0, 0), 0.5, radius=10)
        
        # Progress bar
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+bar_width, bar_y+20), (100, 100, 100), -1)
        
        # Fake progress (since we don't have song length)
        progress = (elapsed % 60) / 60 * bar_width
        draw_filled_rounded_rect(display, (bar_x, bar_y), (bar_x+int(progress), bar_y+20), 
                                main_color, 0.8, radius=5)
        
        # Time display
        cv2.putText(display, f"{minutes:02d}:{seconds:02d}", (center_x-40, bar_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Song count
        if song_list:
            cv2.putText(display, f"📀 {song_index + 1}/{len(song_list)}", (center_x + 200, bar_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # === EMOTION TAG ===
    emotion_text = f"{emoji} {emotion.upper()} MODE {emoji}"
    (text_w, text_h), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    
    draw_filled_rounded_rect(display, (center_x-text_w//2-20, 50), 
                            (center_x+text_w//2+20, 100), (0, 0, 0), 0.7, radius=20)
    draw_rounded_rectangle(display, (center_x-text_w//2-20, 50), 
                          (center_x+text_w//2+20, 100), main_color, 3, radius=20)
    
    cv2.putText(display, emotion_text, (center_x-text_w//2, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, main_color, 3)
    
    # === CONTROLS HINT (Bottom) ===
    draw_filled_rounded_rect(display, (0, h-60), (w, h), (0, 0, 0), 0.5, radius=0)
    
    controls = "SPACE: Play/Pause   |   N: Next   |   P: Previous   |   S: Stop/Camera   |   M: Music Mode   |   ESC: Exit"
    cv2.putText(display, controls, (w//2-450, h-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return display

def main():
    global current_emotion, song_list, song_index, is_playing, music_mode, camera_mode, cap, detection_active, song_paused
    
    print("=" * 70)
    print("🎭 EMOTION-BASED MUSIC PLAYER 🎭")
    print("=" * 70)
    
    # Check folder structure
    if not os.path.exists(MUSIC_FOLDER):
        os.makedirs(MUSIC_FOLDER)
        print(f"\n📁 Created '{MUSIC_FOLDER}' folder")
    
    # Create emotion folders
    for emotion in EMOTION_COLORS.keys():
        folder_path = os.path.join(MUSIC_FOLDER, emotion)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 Created: {folder_path}")
    
    print("\n📂 Folder Structure:")
    for emotion in ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']:
        folder = os.path.join(MUSIC_FOLDER, emotion)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            print(f"   {emotion}/ : {len(files)} files")
    
    print("\n🎯 NEW FEATURES:")
    print("   ✅ Animated round music controls in center")
    print("   ✅ Play/Pause/Next/Previous/Stop buttons")
    print("   ✅ Rotating rings animation")
    print("   ✅ Beautiful particle effects")
    print("\n🚀 Starting...")
    time.sleep(2)
    
    # Main loop
    while True:
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera error")
            break
        
        # Flip and resize
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # CAMERA MODE - Show camera fully
        if camera_mode:
            # Detect emotion ONLY if detection_active is True
            if detection_active:
                detected_emotion, processed_frame, face_regions = detect_emotion(frame)
                
                # Check for emotion change (only if no music playing)
                if not is_playing and not song_paused and detected_emotion:
                    emotion_to_play = check_emotion_change(detected_emotion)
                    
                    if emotion_to_play:
                        # Start playing music
                        current_emotion = emotion_to_play
                        song_list = get_song_list(current_emotion)
                        
                        if song_list:
                            song_index = 0
                            play_song()
                            # Automatically switch to music mode to show visualizer
                            music_mode = True
                            camera_mode = False
                            print(f"\n🎵 Started playing {current_emotion} songs")
                        else:
                            print(f"⚠️ No songs found for {current_emotion}")
            else:
                # Detection is OFF - just show plain frame
                processed_frame = frame.copy()
                detected_emotion = None
                face_regions = []
            
            # Draw camera interface (minimal overlays)
            display = draw_camera_interface(processed_frame if 'processed_frame' in locals() else frame, 
                                           detected_emotion if 'detected_emotion' in locals() else None,
                                           face_regions if 'face_regions' in locals() else [],
                                           detection_active, is_playing)
            cv2.imshow(WINDOW_NAME, display)
        
        # MUSIC MODE - Show full-screen visualizer with buttons
        elif music_mode:
            if current_emotion:
                # Draw full-screen music visualizer with buttons
                display = draw_music_visualizer(frame, current_emotion)
                cv2.imshow(WINDOW_NAME, display)
                
                # Auto-next when song finishes
                if not pygame.mixer.music.get_busy() and is_playing:
                    is_playing = False
                    print("🎵 Song finished, playing next...")
                    time.sleep(1)
                    next_song()
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Global controls
        if key == ord('m') or key == ord('M'):  # Toggle camera/music mode
            if music_mode:
                camera_mode = True
                music_mode = False
                print("\n📷 Switched to camera mode")
            elif camera_mode:
                if current_emotion:
                    music_mode = True
                    camera_mode = False
                    print("\n🎵 Switched to music mode")
        
        # COMMON CONTROLS
        if key == ord('s') or key == ord('S'):  # Stop music and show camera
            if is_playing or song_paused:
                stop_music()
                camera_mode = True
                music_mode = False
        
        if key == ord('r') or key == ord('R'):  # Force resume detection
            if is_playing or song_paused:
                stop_music()
                camera_mode = True
                music_mode = False
            elif not detection_active:
                detection_active = True
                print("\n🔵 Face detection resumed")
        
        if key == ord('n') or key == ord('N'):  # Next song
            if is_playing or song_paused:
                next_song()
        
        if key == ord('p') or key == ord('P'):  # Previous song
            if is_playing or song_paused:
                prev_song()
        
        if key == ord(' '):  # Spacebar - Play/Pause
            if camera_mode:
                # In camera mode: Force play current detected emotion
                if current_emotion and not is_playing:
                    song_list = get_song_list(current_emotion)
                    if song_list:
                        song_index = 0
                        play_song()
                        # Automatically switch to music mode
                        music_mode = True
                        camera_mode = False
            elif music_mode:
                # In music mode: Pause/Resume
                pause_music()
        
        # Music mode only controls
        if music_mode:
            # Manual emotion selection (1-7)
            if key == ord('1'):
                current_emotion = "happy"
                song_list = get_song_list("happy")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('2'):
                current_emotion = "sad"
                song_list = get_song_list("sad")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('3'):
                current_emotion = "angry"
                song_list = get_song_list("angry")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('4'):
                current_emotion = "surprise"
                song_list = get_song_list("surprise")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('5'):
                current_emotion = "neutral"
                song_list = get_song_list("neutral")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('6'):
                current_emotion = "fear"
                song_list = get_song_list("fear")
                if song_list:
                    song_index = 0
                    play_song()
            elif key == ord('7'):
                current_emotion = "disgust"
                song_list = get_song_list("disgust")
                if song_list:
                    song_index = 0
                    play_song()
        
        # Exit
        if key == 27:  # ESC
            print("\n👋 Exiting...")
            break
    
    # Cleanup
    cap.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()
    print("\n✅ Application closed!")

if __name__ == "__main__":
    main()