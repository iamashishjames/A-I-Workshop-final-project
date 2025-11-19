import time
import argparse
import threading
import subprocess
import platform
import shutil
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf  # optional; script still works without a model

# Optional pyttsx3 — used if installed
try:
    import pyttsx3
    HAS_PYTTX3 = True
except Exception:
    HAS_PYTTX3 = False

# ---------------------- Config ----------------------
DEFAULT_EAR = 0.22
CLOSED_SECONDS_THRESHOLD = 2.0
EAR_CONSEC_FRAMES = 15
PERCLOS_WINDOW = 150
PERCLOS_THRESHOLD = 0.4
MAR_THRESHOLD = 0.6
YAWN_CONSEC_FRAMES = 10
ALERT_PERSIST_FRAMES = 1

# ---------------------- Geometry helpers ----------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(mouth):
    if mouth.shape[0] >= 20:
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[14] - mouth[18])
        C = np.linalg.norm(mouth[12] - mouth[16])
    elif mouth.shape[0] >= 8:
        A = np.linalg.norm(mouth[2] - mouth[8])
        B = np.linalg.norm(mouth[3] - mouth[7])
        C = np.linalg.norm(mouth[0] - mouth[6])
    else:
        return 0.0
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# ---------------------- Voice alarm (TTS) ----------------------
def _os_tts_say(text):
    """Try platform TTS: macOS 'say', Linux 'spd-say'/'espeak', Windows PowerShell System.Speech."""
    plat = platform.system()
    try:
        if plat == "Darwin":
            subprocess.run(["say", text], check=False)
            return True
        if plat == "Linux":
            if shutil.which("spd-say"):
                subprocess.run(["spd-say", text], check=False)
                return True
            if shutil.which("espeak"):
                subprocess.run(["espeak", text], check=False)
                return True
            return False
        if plat == "Windows":
            cmd = [
                "powershell", "-Command",
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
            ]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception:
        return False
    return False

def alarm_wake_up(repeat=2, text="Wake up"):
    """Try pyttsx3 first, then OS TTS, then fallback visual+beep."""
    # 1) pyttsx3 (preferred, cross-platform)
    if HAS_PYTTX3:
        try:
            engine = pyttsx3.init()
            for _ in range(repeat):
                engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass

    # 2) platform TTS
    for _ in range(repeat):
        ok = _os_tts_say(text)
        if ok:
            time.sleep(0.2)
        else:
            break

    # 3) visual + beep fallback if TTS not available or failed
    try:
        for _ in range(repeat):
            img = np.zeros((120, 480, 3), dtype=np.uint8)
            cv2.putText(img, text.upper(), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            cv2.imshow("ALERT", img)
            print("\a", end="", flush=True)
            cv2.waitKey(700)
            try:
                cv2.destroyWindow("ALERT")
            except Exception:
                pass
    except Exception:
        for _ in range(repeat):
            print("***", text.upper(), "***")

# ---------------------- MediaPipe Landmark Provider ----------------------
class LandmarkProvider:
    def __init__(self):
        # initialize MediaPipe face mesh once
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # these indices are for MediaPipe's face mesh
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]
        # mouth indices — keep them consistent with the face mesh indexing
        self.mouth_idx = [78,95,88,178,87,14,317,402,318,324,308,191,80,81,82,13,312,311,415,308]

    def get_landmarks(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm.landmark])
        # guard against index errors
        try:
            left_eye = pts[self.left_eye_idx]
            right_eye = pts[self.right_eye_idx]
            mouth = pts[self.mouth_idx]
        except IndexError:
            return None
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
            "face": pts
        }

# ---------------------- Drowsiness Detector ----------------------
class DrowsinessDetector:
    def __init__(self, lp, eye_model=None, show_window=True):
        self.lp = lp
        self.eye_model = eye_model
        self.show_window = show_window
        self.EAR_BASELINE = None
        self.adaptive_ear_threshold = DEFAULT_EAR
        self.perclos = deque(maxlen=PERCLOS_WINDOW)
        self.closed_frames = 0
        self.yawn_frames = 0
        self.alert = False
        self.alert_counter = 0
        self.closed_start_time = None

    def _update_threshold(self):
        if self.EAR_BASELINE is not None:
            self.adaptive_ear_threshold = max(0.12, 0.55 * self.EAR_BASELINE)
        else:
            self.adaptive_ear_threshold = DEFAULT_EAR

    def crop_eye_patch(self, frame, eye_points, pad=5):
        x_min = max(int(eye_points[:, 0].min()) - pad, 0)
        x_max = min(int(eye_points[:, 0].max()) + pad, frame.shape[1])
        y_min = max(int(eye_points[:, 1].min()) - pad, 0)
        y_max = min(int(eye_points[:, 1].max()) + pad, frame.shape[0])
        if x_max <= x_min or y_max <= y_min:
            return frame
        return frame[y_min:y_max, x_min:x_max]

    def classify_eye(self, patch):
        if self.eye_model is None:
            return None
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = patch
        resized = cv2.resize(gray, (34, 26)).astype("float32") / 255.0
        arr = resized.reshape((1, resized.shape[0], resized.shape[1], 1))
        pred = self.eye_model.predict(arr, verbose=0)[0][0]
        return float(pred)

    def analyze_frame(self, frame):
        lm = self.lp.get_landmarks(frame)
        if lm is None:
            self.closed_start_time = None
            return frame, False, "No face"

        left, right, mouth = lm["left_eye"], lm["right_eye"], lm["mouth"]
        leftEAR, rightEAR = eye_aspect_ratio(left), eye_aspect_ratio(right)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        self._update_threshold()

        # PERCLOS update
        self.perclos.append(1 if ear < self.adaptive_ear_threshold else 0)
        perclos_val = sum(self.perclos) / len(self.perclos) if self.perclos else 0.0

        # time-based closure
        now = time.time()
        if ear < self.adaptive_ear_threshold:
            if self.closed_start_time is None:
                self.closed_start_time = now
            closed_duration = now - self.closed_start_time
            self.closed_frames += 1
        else:
            closed_duration = 0.0
            self.closed_start_time = None
            self.closed_frames = 0

        # yawning
        if mar > MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            self.yawn_frames = 0

        # optional TF eye model nudging
        eye_open_prob = None
        if self.eye_model is not None:
            lp_patch = self.crop_eye_patch(frame, left)
            rp_patch = self.crop_eye_patch(frame, right)
            p1 = self.classify_eye(lp_patch)
            p2 = self.classify_eye(rp_patch)
            if p1 is not None and p2 is not None:
                eye_open_prob = (p1 + p2) / 2.0
                if eye_open_prob < 0.5:
                    ear = min(ear, self.adaptive_ear_threshold - 0.01)

        # fusion
        is_drowsy = False
        reasons = []
        if closed_duration >= CLOSED_SECONDS_THRESHOLD:
            is_drowsy = True; reasons.append(f"ClosedSec={closed_duration:.1f}")
        if self.closed_frames >= EAR_CONSEC_FRAMES:
            is_drowsy = True; reasons.append("LongClosure(frames)")
        if perclos_val > PERCLOS_THRESHOLD:
            is_drowsy = True; reasons.append(f"PERCLOS={perclos_val:.2f}")
        if self.yawn_frames >= YAWN_CONSEC_FRAMES:
            is_drowsy = True; reasons.append("Yawning")
        if mar > MAR_THRESHOLD and perclos_val > (PERCLOS_THRESHOLD * 0.7):
            is_drowsy = True; reasons.append("Yawn+PERCLOS")

        # smoothing / persistence
        self.alert_counter = self.alert_counter + 1 if is_drowsy else 0
        final_alert = self.alert_counter >= ALERT_PERSIST_FRAMES

        # visualization
        vis = frame.copy()
        for p in left:  cv2.circle(vis, tuple(p), 1, (0, 255, 0), -1)
        for p in right: cv2.circle(vis, tuple(p), 1, (0, 255, 0), -1)
        for p in mouth: cv2.circle(vis, tuple(p), 1, (255, 0, 0), -1)

        cv2.putText(vis, f"EAR={ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f"PERCLOS={perclos_val:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f"MAR={mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if eye_open_prob is not None:
            cv2.putText(vis, f"EyeOpenProb={eye_open_prob:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if closed_duration > 0:
            cv2.putText(vis, f"ClosedSec={closed_duration:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if final_alert:
            cv2.putText(vis, "DROWSINESS DETECTED", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        return vis, final_alert, ",".join(reasons)

    def run(self, src=0, cap=None):
        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(src)
            own_cap = True
        if not cap.isOpened():
            raise RuntimeError("Unable to open video source")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, is_drowsy, reasons = self.analyze_frame(frame)

                if is_drowsy and not self.alert:
                    threading.Thread(target=alarm_wake_up, kwargs={"repeat":2}, daemon=True).start()
                    self.alert = True
                if not is_drowsy:
                    self.alert = False

                if self.show_window:
                    cv2.imshow("DrowsinessDetector", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if own_cap:
                cap.release()
            cv2.destroyAllWindows()

# ---------------------- Calibration ----------------------
def calibrate_baseline(cap, lp, num_frames=80, show=False):
    print(f"Calibrating baseline for {num_frames} frames... please look at the camera")
    ear_vals = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        lm = lp.get_landmarks(frame)
        if lm is None:
            if show:
                cv2.putText(frame, "No face detected for calibration", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Calibration", frame); cv2.waitKey(1)
            continue
        leftEAR = eye_aspect_ratio(lm["left_eye"]); rightEAR = eye_aspect_ratio(lm["right_eye"])
        ear_vals.append((leftEAR + rightEAR) / 2.0)
        if show:
            cv2.imshow("Calibration", frame); cv2.waitKey(1)
    try:
        cv2.destroyWindow("Calibration")
    except Exception:
        pass
    if not ear_vals:
        print("Calibration failed: no usable frames")
        return None
    baseline = float(np.median(ear_vals))
    print(f"Calibration done. Baseline EAR = {baseline:.3f}")
    return baseline

# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser(description="Drowsiness detection (MediaPipe + OpenCV + optional TF model)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained eye classifier (.h5)")
    parser.add_argument("--no_calib", action="store_true", help="Skip calibration")
    parser.add_argument("--src", default=0, help="Video source (0 for webcam or path to video)")
    parser.add_argument("--no_window", action="store_true", help="Do not show OpenCV window")
    args = parser.parse_args()

    src = int(args.src) if str(args.src).isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Unable to open video source:", src); return

    lp = LandmarkProvider()

    baseline = None
    if not args.no_calib:
        baseline = calibrate_baseline(cap, lp, num_frames=80, show=not args.no_window)

    try:
        cap.release()
    except:
        pass
    # reopen same source (was previously hard-coded to 2); use src instead
    cap = cv2.VideoCapture(src)

    eye_model = None
    if args.model is not None:
        eye_model = tf.keras.models.load_model(args.model)
        print("Loaded eye model:", args.model)

    dd = DrowsinessDetector(lp, eye_model=eye_model, show_window=not args.no_window)
    if baseline is not None:
        dd.EAR_BASELINE = baseline

    dd.run(cap=cap)

if __name__ == "__main__":
    main()
