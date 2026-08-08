import cv2
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from functools import wraps
from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# --- CONFIGURATION ---
USERNAME = "admin"
PASSWORD = "dna"  # Change this to your desired password
PORT = 5002
SAVE_DIR = "captures"
MIN_CONTOUR_AREA = 2500       # Sensitivity (lower = more sensitive)
COOLDOWN_SECONDS = 3          # Seconds between saving snapshots

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- AUTOMATIC PUBLIC TUNNEL FUNCTION ---
def start_public_tunnel():
    """Starts cloudflared in the background and prints the public URL to the screen."""
    def _tunnel_thread():
        try:
            cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Scan subprocess logs for the public trycloudflare.com URL
            for line in iter(process.stdout.readline, ''):
                if "trycloudflare.com" in line:
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        public_url = match.group(0)
                        print("\n" + "="*60)
                        print(f" [★] LIVE PUBLIC CAMERA URL:")
                        print(f"     {public_url}")
                        print("="*60 + "\n")
                        break
        except FileNotFoundError:
            print("\n[!] 'cloudflared' binary not found.")
            print("[!] Install it using: sudo apt install cloudflared -y\n")
        except Exception as e:
            print(f"\n[!] Tunnel error: {e}\n")

    # Run tunnel asynchronously so it doesn't block Flask
    threading.Thread(target=_tunnel_thread, daemon=True).start()

# --- HTTP BASIC AUTHENTICATION DECORATOR ---
def check_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != USERNAME or auth.password != PASSWORD:
            return Response(
                'Access Denied: Invalid Credentials\n', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'}
            )
        return f(*args, **kwargs)
    return decorated

camera = cv2.VideoCapture(0)

def generate_frames():
    avg_frame = None
    last_saved_time = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Motion detection processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_frame is None:
            avg_frame = gray.astype("float")
            continue

        cv2.accumulateWeighted(gray, avg_frame, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        status_text = "MOTION DETECTED" if motion_detected else "CLEAR"
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        current_time = time.time()
        if motion_detected and (current_time - last_saved_time) > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"motion_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[+] Motion captured! Saved to: {filename}")
            last_saved_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
@check_auth
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Starlink Security Camera</title>
            <style>
                body { font-family: sans-serif; background: #121212; color: #fff; text-align: center; margin-top: 2rem; }
                .card { display: inline-block; background: #1e1e1e; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
                img { max-width: 100%; height: auto; border-radius: 4px; border: 2px solid #333; }
            </style>
        </head>
        <body>
            <h2>Starlink Security Camera Feed</h2>
            <div class="card">
                <img src="/video_feed" alt="Live Feed">
            </div>
        </body>
        </html>
    ''')

@app.route('/video_feed')
@check_auth
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the tunnel thread right before launching Flask
    start_public_tunnel()
    app.run(host='0.0.0.0', port=PORT, debug=False)