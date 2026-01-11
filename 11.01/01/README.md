# Finger Counter - Hand Gesture Recognition

A project for recognizing the number of raised fingers (1, 2, 3) using a webcam with voice feedback.

## Installation

### 1. Activate Virtual Environment

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Step-by-step:

1. Open Terminal

2. Navigate to project directory:
```bash
cd /Users/pavellubimov/Desktop/python/11.01/01
```

3. Activate virtual environment:
```bash
source venv/bin/activate
```

4. Run the program:
```bash
python finger_counter.py
```

### What happens on first run:
- The program will generate Hebrew audio files (requires internet)
- You'll see "Generating Hebrew audio files..."
- After generation: "Audio files generated successfully!"
- macOS will ask for camera permission - allow it
- Video window will open

## Usage

1. After starting, a video window will open showing your camera feed
2. Show your hand in front of the camera
3. Raise 1, 2, or 3 fingers
4. The program will display the recognized number in large green digits on screen
5. The program will speak the number out loud in Hebrew (אחת, שתיים, שלוש)
6. Press 'q' to quit

## Technologies

- **OpenCV** - for camera handling and video display
- **MediaPipe** - for hand detection and finger tracking
- **NumPy** - for mathematical operations
- **gTTS** - Google Text-to-Speech for Hebrew voice synthesis
- **playsound** - for audio playback

## Notes

- The program works with one hand at a time
- Best results are achieved with good lighting
- Background should contrast with your hand for better recognition
- Voice feedback plays when the finger count changes and after 1 second cooldown
- On first run, the program will generate Hebrew audio files (requires internet connection)
- Audio files are cached in a temporary directory for faster playback
