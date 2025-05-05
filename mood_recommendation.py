from flask import Flask, render_template, Response, jsonify
import cv2
import random
import requests
import time
import re
import numpy as np
from deepface import DeepFace
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch
from difflib import SequenceMatcher

app = Flask(__name__)

# IMDb URLs mapped to emotions for Telugu movies
TELUGU_URLS = {
    "happy": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=comedy',
    "interesting": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=adventure',
    "sad": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=drama',
    "angry": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=action',
    "surprise": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=thriller',
    "neutral": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=documentary',
    "fear": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=horror',
    "disgust": 'https://www.imdb.com/search/title/?title_type=feature&primary_language=te&genres=fantasy',
}

# Imgflip API Credentials
IMGFLIP_USERNAME = "S.K.Tanusri"
IMGFLIP_PASSWORD = "Tanusri02"

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
last_emotion = "neutral"
last_detection_time = time.time()
detection_interval = 5  # Detect emotion every 5 seconds

# Cache for IMDb data
imdb_cache = {
    "last_updated": 0,
    "data": {}
}

# Track last 5 recommended movies
last_recommendations = []


# Helper function for similarity scoring
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Clean movie title
def clean_movie_title(title):
    cleaned_title = re.sub(r'[^a-zA-Z\s]', '', title).strip()
    return cleaned_title


def get_trending_memes():
    """Fetch the top trending meme templates from Imgflip."""
    url = "https://api.imgflip.com/get_memes"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            return data["data"]["memes"]

    return []


def generate_dynamic_text(emotion):
    """Generate dynamic meme text based on detected emotion."""
    text_options = {
        "happy": [
            "When you finally understand the topic! 🎉",
            "Watching Telugu comedies be like... 😎",
            "This movie actually makes sense! 🤯"
        ],
        "sad": [
            "When your favorite character dies in the movie... 😢",
            "That moment when the ending was sadder than expected...",
            "Why does this movie make me cry? 😭"
        ],
        "angry": [
            "When the villain wins in the first half... 😡",
            "Why did they kill off the best character?!",
            "Movie tickets be like: *All prices increasing* 😤"
        ],
        "surprise": [
            "Wait… THAT was the plot twist? 🤯",
            "When you didn't expect that ending!",
            "Why did no one warn me about this scene?!"
        ],
        "fear": [
            "Watching a horror movie alone like... 🤔",
            "When the jump scare happens but you were ready 😨",
            "Telugu horror movies got me like 😱"
        ],
        "neutral": [
            "Just another day watching Telugu movies... 😐",
            "Movie was average but the songs were good.",
            "When you finish a 3-hour movie and don't know how to feel."
        ],
        "disgust": [
            "When they add an unnecessary romance plot 🙄",
            "That awkward scene you weren't prepared for...",
            "Movie food prices got me like 😖"
        ]
    }

    return random.choice(text_options.get(emotion.lower(), ["Telugu movies for the win! 📚"]))


def generate_meme(emotion):
    """Generate a meme based on detected emotion."""
    trending_memes = get_trending_memes()

    if not trending_memes:
        return None

    selected_meme = random.choice(trending_memes)
    template_id = selected_meme["id"]

    top_text = generate_dynamic_text(emotion)

    url = "https://api.imgflip.com/caption_image"
    payload = {
        "template_id": template_id,
        "username": IMGFLIP_USERNAME,
        "password": IMGFLIP_PASSWORD,
        "text0": top_text,
        "text1": ""
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            return data["data"]["url"]

    return None


def fetch_movie_recommendations(emotion):
    """Fetch recommended movies based on emotion."""
    global imdb_cache, last_recommendations

    # Adjust mood if needed
    if emotion.lower() == "sad":
        emotion = "happy"  # For sad emotion, recommend happy movies
    elif emotion.lower() == "bored":
        emotion = "interesting"

    url = TELUGU_URLS.get(emotion.lower(), TELUGU_URLS["neutral"])

    # Check if cache is stale (older than 1 hour)
    if time.time() - imdb_cache["last_updated"] > 3600 or emotion not in imdb_cache["data"]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "lxml")

            # Extract movie titles
            titles = []
            for item in soup.select('.ipc-metadata-list-summary-item__t'):
                title = item.text.strip()
                if title and len(title) > 1:  # Ensure it's a real title
                    titles.append(title)

            # Shuffle the list for randomness
            random.shuffle(titles)

            # Update cache
            imdb_cache["data"][emotion] = titles
            imdb_cache["last_updated"] = time.time()
        except Exception as e:
            print(f"Error fetching movies: {e}")
            return "Error fetching movies."

    # Get a random movie from the cached list, excluding recent recommendations
    if emotion in imdb_cache["data"] and imdb_cache["data"][emotion]:
        available_titles = [title for title in imdb_cache["data"][emotion] if title not in last_recommendations]

        if available_titles:
            selected_movie = random.choice(available_titles)
            # Update last_recommendations (keep only the last 5)
            last_recommendations.append(selected_movie)
            if len(last_recommendations) > 5:
                last_recommendations.pop(0)
            return selected_movie
        else:
            # If all titles have been recommended recently, reset the list
            last_recommendations.clear()
            return random.choice(imdb_cache["data"][emotion])
    else:
        return "No movies found."


def search_youtube_trailer(movie_title):
    """Search YouTube for a movie trailer."""
    cleaned_title = clean_movie_title(movie_title)
    search_query = f"{cleaned_title} Telugu movie trailer"

    try:
        videos_search = VideosSearch(search_query, limit=1, region="IN")
        result = videos_search.result()

        if "result" in result and result["result"]:
            return result["result"][0]["link"]
    except Exception as e:
        print(f"Error searching YouTube: {e}")

    return None


def generate_frames():
    """Capture video frames and detect emotion."""
    global last_emotion, last_detection_time

    # Try to open the camera
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            # Return a default frame with error message
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
    except Exception as e:
        print(f"Camera error: {e}")
        # Create an error frame
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, f"Camera Error: {str(e)}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    while True:
        try:
            success, frame = cap.read()

            if not success:
                print("Failed to receive frame")
                # Create an error frame
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "Failed to receive frame", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            current_time = time.time()

            # Process frame for emotion detection every few seconds
            if current_time - last_detection_time >= detection_interval:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                    if len(faces) > 0:  # Only process if faces detected
                        for (x, y, w, h) in faces:
                            face_roi = frame[y:y + h, x:x + w]

                            # Analyze emotion
                            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            if isinstance(result, list) and len(result) > 0:
                                detected_emotion = result[0].get('dominant_emotion', 'neutral')
                                last_emotion = detected_emotion
                            elif isinstance(result, dict):
                                detected_emotion = result.get('dominant_emotion', 'neutral')
                                last_emotion = detected_emotion

                            # Draw rectangle and label
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, last_emotion, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    last_detection_time = current_time

                except Exception as e:
                    print(f"Error in emotion detection: {e}")

            # Always draw the latest emotion detection on the frame
            if last_emotion:
                cv2.putText(frame, f"Emotion: {last_emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Convert frame to bytes for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Frame processing error: {e}")
            # Create an error frame
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, f"Processing Error: {str(e)}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # Brief pause before retrying


@app.route('/')
def index():
    return render_template('here_movie.html')


@app.route('/health')
def health():
    return jsonify({"status": "ok", "emotion_detection": "running"})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_emotion_data')
def get_emotion_data():
    """Send detected emotion, movie and meme to frontend."""
    global last_emotion

    try:
        movie = fetch_movie_recommendations(last_emotion)
        trailer_url = search_youtube_trailer(movie)
        meme_url = generate_meme(last_emotion)

        return jsonify({
            "emotion": last_emotion,
            "movie": movie,
            "trailer": trailer_url,
            "meme": meme_url
        })
    except Exception as e:
        print(f"Error in get_emotion_data: {e}")
        return jsonify({
            "emotion": last_emotion,
            "movie": "Error fetching movie recommendations",
            "trailer": None,
            "meme": None,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')