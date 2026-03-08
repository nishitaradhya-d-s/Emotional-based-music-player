
EMOTION BASED MUSIC PLAYER - RUN GUIDE

1. Install Python 3.10 or above
2. Open terminal / command prompt
3. Navigate to project folder

4. Create virtual environment (recommended):
   python -m venv venv
   venv\Scripts\activate  (Windows)
   source venv/bin/activate (Mac/Linux)

5. Install requirements:
   pip install -r requirements.txt

pip install tf-keras


6. Add your own MP3 songs inside:
   music/happy
   music/sad
   music/angry
   music/neutral

7. Run the project:
   python main.py

8. Press ESC to exit

Note:
- Webcam is required
- Supports multiple faces
- Music is chosen based on majority emotion
