import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from gtts import gTTS
from playsound import playsound
import threading
import tempfile


class FingerCounter:
    def __init__(self):
        # Path to hand landmarker model
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                "Please download it from: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Finger tip IDs in MediaPipe (thumb, index, middle, ring, pinky)
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_base = [2, 6, 10, 14, 18]

        # Track last spoken number to avoid repeating
        self.last_spoken = None
        self.last_speak_time = 0

        # Create audio cache directory
        self.audio_cache_dir = tempfile.mkdtemp()

        # Pre-generate audio files for numbers 1-5 in Hebrew
        self.audio_files = {}
        self._generate_audio_files()

    def count_fingers(self, hand_landmarks, handedness_label):
        """Counts the number of raised fingers"""
        fingers_up = []

        # Check which hand (left or right)
        is_right_hand = handedness_label == "Right"

        # Thumb (special logic)
        if is_right_hand:
            # For right hand: if tip is to the left of base
            if hand_landmarks[self.finger_tips[0]].x < hand_landmarks[self.finger_base[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        else:
            # For left hand: if tip is to the right of base
            if hand_landmarks[self.finger_tips[0]].x > hand_landmarks[self.finger_base[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        # Other fingers (if tip is above base)
        for i in range(1, 5):
            if hand_landmarks[self.finger_tips[i]].y < hand_landmarks[self.finger_base[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        return sum(fingers_up), fingers_up

    def detect_gesture(self, hand_landmarks, handedness_label):
        """Detects specific hand gestures"""
        finger_count, fingers_up = self.count_fingers(hand_landmarks, handedness_label)

        # Peace sign (V) - only index and middle fingers up
        if fingers_up == [0, 1, 1, 0, 0]:
            return "Peace", finger_count

        # Thumbs up - only thumb up
        if fingers_up == [1, 0, 0, 0, 0]:
            return "Thumbs Up", finger_count

        # OK sign - thumb and index finger touching, others up
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        if distance < 0.05 and fingers_up[2:] == [1, 1, 1]:  # Close together + other fingers up
            return "OK Sign", finger_count

        # Pointing - only index finger up
        if fingers_up == [0, 1, 0, 0, 0]:
            return "Pointing", finger_count

        # Rock sign - index and pinky up
        if fingers_up == [0, 1, 0, 0, 1]:
            return "Rock", finger_count

        # Fist - all fingers down
        if finger_count == 0:
            return "Fist", finger_count

        # Open palm - all fingers up
        if finger_count == 5:
            return "Open Palm", finger_count

        # Default - just show finger count
        return f"{finger_count} Fingers", finger_count

    def _generate_audio_files(self):
        """Pre-generate audio files for Hebrew numbers"""
        number_words = {
            1: "אחת",
            2: "שתיים",
            3: "שלוש",
            4: "ארבע",
            5: "חמש"
        }

        print("Generating Hebrew audio files...")
        for number, text in number_words.items():
            try:
                audio_path = os.path.join(self.audio_cache_dir, f"{number}.mp3")
                tts = gTTS(text=text, lang='iw', slow=False)  # 'iw' is Hebrew language code
                tts.save(audio_path)
                self.audio_files[number] = audio_path
            except Exception as e:
                print(f"Error generating audio for {number}: {e}")

        print("Audio files generated successfully!")

    def speak_number(self, number):
        """Plays the Hebrew audio for the number in a separate thread"""
        current_time = time.time()
        # Only speak if number changed and at least 1 second has passed
        if number != self.last_spoken and (current_time - self.last_speak_time) > 1.0:
            self.last_spoken = number
            self.last_speak_time = current_time

            # Get audio file path
            audio_file = self.audio_files.get(number)
            if not audio_file:
                return

            # Run audio playback in separate thread to avoid blocking
            def play_audio():
                try:
                    playsound(audio_file)
                except Exception as e:
                    print(f"Error playing audio: {e}")

            thread = threading.Thread(target=play_audio)
            thread.daemon = True
            thread.start()

    def run(self):
        """Starts finger recognition from camera"""
        cap = cv2.VideoCapture(0)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev_time = 0

        print("Camera started! Show your fingers (1, 2 or 3)")
        print("Press 'q' to quit")

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image from camera")
                break

            # Flip image for more natural display
            img = cv2.flip(img, 1)

            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Detect hands
            detection_result = self.detector.detect(mp_image)

            finger_count = 0
            gesture_name = None

            # If hand is detected
            if detection_result.hand_landmarks and detection_result.handedness:
                hand_landmarks = detection_result.hand_landmarks[0]
                handedness = detection_result.handedness[0][0].category_name

                # Draw hand landmarks with connections
                h, w = img.shape[:2]

                # Draw connections between landmarks
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]

                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(img, start_point, end_point, (255, 0, 0), 2)

                # Draw landmarks as circles
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

                # Detect gesture
                gesture_name, finger_count = self.detect_gesture(hand_landmarks, handedness)

                # Speak the number
                if finger_count > 0:
                    self.speak_number(finger_count)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # Display gesture name and finger count
            if finger_count > 0 or gesture_name:
                # Display large number
                cv2.putText(img, str(finger_count), (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 15)

                # Display gesture name
                if gesture_name:
                    cv2.putText(img, gesture_name, (50, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

            # Display FPS
            cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Display instructions
            cv2.putText(img, "Press 'q' to quit", (10, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show image
            cv2.imshow("Finger Counter", img)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    counter = FingerCounter()
    counter.run()
