import os
import time
import argparse
import threading
import platform
from collections import deque

import cv2
import numpy as np

# try optional imports
try:
    import dlib
    HAS_DLIB = True
except Exception:
    HAS_DLIB = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except Exception:
    HAS_TF = False

# optional text-to-speech for alarm
try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    HAS_TTS = False

# ---------------------- Configuration & helpers ----------------------
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"  # optional
DEFAULT_MODEL_PATH = "eye_classifier.h5"

# thresholds and parameters (tweak these for your environment)
CLOSED_SECONDS_THRESHOLD = 2.0  # seconds eyes must stay closed to trigger long-closure alarm
EAR_CONSEC_FRAMES = 15          # legacy frame-based threshold (kept for compatibility)
EAR_THRESHOLD = 0.22            # fallback EAR threshold
PERCLOS_WINDOW = 150            # frames for PERCLOS window (~seconds * fps)
PERCLOS_THRESHOLD = 0.4         # percent of time eyes closed in window to trigger drowsiness
MAR_THRESHOLD = 0.6             # mouth aspect ratio threshold to consider yawning
YAWN_CONSEC_FRAMES = 10
ALERT_PERSIST_FRAMES = 1        # require this many consecutive drowsy checks before alarming

# convenience
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# EAR calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

# MAR calculation
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
        return 0
    mar = (A + B) / (2.0 * C) if C != 0 else 0
    return mar

# alarm: try text-to-speech "Wake up", fallback to visual flash + beep
def alarm_wake_up(repeat=1):
    if HAS_TTS:
        try:
            engine = pyttsx3.init()
            # speak asynchronously in this thread (pyttsx3 blocks runAndWait), so run here
            for _ in range(repeat):
                engine.say('Wake up')
            engine.runAndWait()
            return
        except Exception:
            pass
    # fallback: show visual alert window and short system beep(s)
    try:
        for _ in range(repeat):
            beep = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(beep, 'WAKE UP', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow('ALERT', beep)
            # cross-platform: try to trigger a short sound via ascii bell (may or may not play)
            print('\a', end='', flush=True)
            cv2.waitKey(700)
            try:
                cv2.destroyWindow('ALERT')
            except Exception:
                pass
    except Exception:
        # last-resort print
        for _ in range(repeat):
            print('*** WAKE UP ***')

# ---------------------- Landmark provider ----------------------
class LandmarkProvider:
    def __init__(self, use_dlib=False):
        self.use_dlib = use_dlib and HAS_DLIB and os.path.exists(DLIB_LANDMARK_PATH)
        if self.use_dlib:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
        else:
            if not HAS_MEDIAPIPE:
                raise RuntimeError("MediaPipe is required when dlib is not available. Install mediapipe.")
            self.mp_face = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   refine_landmarks=True,
                                                   min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)

    def get_landmarks(self, frame):
        h, w = frame.shape[:2]
        if self.use_dlib:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if len(rects) == 0:
                return None
            shape = self.predictor(gray, rects[0])
            coords = shape_to_np(shape)
            leftEye = coords[36:42]
            rightEye = coords[42:48]
            mouth = coords[48:68]
            return {"left_eye": leftEye, "right_eye": rightEye, "mouth": mouth, "face": coords}
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0]
            pts = []
            for p in lm.landmark:
                pts.append((int(p.x * w), int(p.y * h)))
            pts = np.array(pts)
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            mouth_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 415, 308]
            leftEye = pts[left_eye_idx]
            rightEye = pts[right_eye_idx]
            mouth = pts[mouth_idx]
            return {"left_eye": leftEye, "right_eye": rightEye, "mouth": mouth, "face": pts}

# ---------------------- Drowsiness Detector ----------------------
class DrowsinessDetector:
    def __init__(self, landmark_provider, eye_model=None, show_window=True):
        self.lp = landmark_provider
        self.eye_model = eye_model
        self.show_window = show_window
        self.frame_idx = 0
        self.EAR_BASELINE = None
        self.adaptive_ear_threshold = EAR_THRESHOLD
        self.leftEAR_deque = deque(maxlen=PERCLOS_WINDOW)
        self.closed_frames = 0
        self.yawn_frames = 0
        self.alert = False
        self.alert_counter = 0
        # time-based closed detection
        self.closed_start_time = None

    def crop_eye_patch(self, frame, eye_points, pad=5):
        x_min = max(np.min(eye_points[:,0]) - pad, 0)
        x_max = min(np.max(eye_points[:,0]) + pad, frame.shape[1])
        y_min = max(np.min(eye_points[:,1]) - pad, 0)
        y_max = min(np.max(eye_points[:,1]) + pad, frame.shape[0])
        patch = frame[y_min:y_max, x_min:x_max]
        return patch

    def classify_eye(self, patch):
        if self.eye_model is None or not HAS_TF:
            return None
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = patch
        resized = cv2.resize(gray, (34, 26))
        arr = resized.astype('float32') / 255.0
        arr = arr.reshape((1, resized.shape[0], resized.shape[1], 1))
        pred = self.eye_model.predict(arr, verbose=0)[0][0]
        return pred

    def _update_adaptive_threshold(self):
        if self.EAR_BASELINE is not None:
            # closed if less than a fraction of baseline; enforce sensible min
            self.adaptive_ear_threshold = max(0.12, 0.55 * self.EAR_BASELINE)
        else:
            self.adaptive_ear_threshold = EAR_THRESHOLD

    def analyze_frame(self, frame):
        lm = self.lp.get_landmarks(frame)
        if lm is None:
            # reset closed timer when no face
            self.closed_start_time = None
            return frame, False, "No face"
        left = lm['left_eye']
        right = lm['right_eye']
        mouth = lm['mouth']
        leftEAR = eye_aspect_ratio(left)
        rightEAR = eye_aspect_ratio(right)
        ear = (leftEAR + rightEAR) / 2.0

        # update adaptive threshold
        self._update_adaptive_threshold()

        # update PERCLOS deque
        self.leftEAR_deque.append(1 if ear < self.adaptive_ear_threshold else 0)
        perclos = sum(self.leftEAR_deque) / len(self.leftEAR_deque)

        # time-based sustained closure detection
        now = time.time()
        if ear < self.adaptive_ear_threshold:
            if self.closed_start_time is None:
                self.closed_start_time = now
            closed_duration = now - self.closed_start_time
        else:
            closed_duration = 0
            self.closed_start_time = None

        # legacy frame-based counter (kept but not primary)
        if ear < self.adaptive_ear_threshold:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        mar = mouth_aspect_ratio(mouth)
        if mar > MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            self.yawn_frames = 0

        # optional TF-based eye classifier (if available)
        eye_open_prob = None
        if self.eye_model is not None and HAS_TF:
            lp = self.crop_eye_patch(frame, left)
            rp = self.crop_eye_patch(frame, right)
            p1 = self.classify_eye(lp)
            p2 = self.classify_eye(rp)
            if p1 is not None and p2 is not None:
                eye_open_prob = (p1 + p2) / 2.0
                if eye_open_prob < 0.5:
                    ear = min(ear, self.adaptive_ear_threshold - 0.01)

        # decision fusion
        is_drowsy = False
        reasons = []

        # 1) long time-based closure (>= CLOSED_SECONDS_THRESHOLD)
        if closed_duration >= CLOSED_SECONDS_THRESHOLD:
            is_drowsy = True
            reasons.append(f'ClosedDuration={closed_duration:.1f}s')

        # 2) legacy frame closure (in case fps unknown)
        if self.closed_frames >= EAR_CONSEC_FRAMES:
            is_drowsy = True
            reasons.append('Long eye closure (frames)')

        # 3) PERCLOS
        if perclos > PERCLOS_THRESHOLD:
            is_drowsy = True
            reasons.append(f'PERCLOS={perclos:.2f}')

        # 4) yawning: immediate trigger
        if self.yawn_frames >= YAWN_CONSEC_FRAMES:
            is_drowsy = True
            reasons.append('Yawning')

        # combine yawning with medium PERCLOS
        if (mar > MAR_THRESHOLD) and (perclos > (PERCLOS_THRESHOLD * 0.7)):
            if 'Yawning' not in reasons:
                reasons.append('Yawning+PERCLOS')
            is_drowsy = True

        # smoothing: require persistence
        if is_drowsy:
            self.alert_counter += 1
        else:
            self.alert_counter = 0
        if self.alert_counter >= ALERT_PERSIST_FRAMES:
            final_alert = True
        else:
            final_alert = False

        # visualization
        vis = frame.copy()
        for pt in left:
            cv2.circle(vis, tuple(pt), 1, (0,255,0), -1)
        for pt in right:
            cv2.circle(vis, tuple(pt), 1, (0,255,0), -1)
        for pt in mouth:
            cv2.circle(vis, tuple(pt), 1, (255,0,0), -1)
        cv2.putText(vis, f'EAR={ear:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f'PERCLOS={perclos:.2f}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f'MAR={mar:.2f}', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if eye_open_prob is not None:
            cv2.putText(vis, f'EyeOpenProb={eye_open_prob:.2f}', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if closed_duration > 0:
            cv2.putText(vis, f'ClosedSec={closed_duration:.1f}', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if final_alert:
            cv2.putText(vis, 'DROWSINESS DETECTED', (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        return vis, final_alert, ','.join(reasons)

    def run(self, src=0, cap=None):
        # Accept either a cv2.VideoCapture (cap) or a source index/path
        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(src)
            own_cap = True
        if not cap.isOpened():
            raise RuntimeError('Unable to open video source')
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, is_drowsy, reasons = self.analyze_frame(frame)
                # trigger alarm when drowsiness confirmed
                if is_drowsy and not self.alert:
                    # launch TTS or fallback in separate thread so it doesn't block
                    threading.Thread(target=alarm_wake_up, kwargs={'repeat':2}, daemon=True).start()
                    self.alert = True
                if not is_drowsy:
                    self.alert = False
                if self.show_window:
                    cv2.imshow('DrowsinessDetector', out_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        finally:
            if own_cap:
                cap.release()
            cv2.destroyAllWindows()

# ---------------------- Calibration ----------------------

def calibrate_baseline(cap, lp, num_frames=80, show=False):
    """
    Capture num_frames and compute median EAR (open-eye baseline).
    Returns baseline_ear (float) or None if calibration failed.
    """
    ear_vals = []
    print(f"Calibrating baseline for {num_frames} frames... please look at the camera")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        lm = lp.get_landmarks(frame)
        if lm is None:
            if show:
                cv2.putText(frame, 'No face detected for calibration', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('Calibration', frame); cv2.waitKey(1)
            continue
        left = lm['left_eye']; right = lm['right_eye']
        leftEAR = eye_aspect_ratio(left); rightEAR = eye_aspect_ratio(right)
        ear = (leftEAR + rightEAR) / 2.0
        ear_vals.append(ear)
        if show:
            cv2.imshow('Calibration', frame); cv2.waitKey(1)
    if show:
        try:
            cv2.destroyWindow('Calibration')
        except Exception:
            pass
    if len(ear_vals) == 0:
        print('Calibration failed: no usable frames')
        return None
    baseline = float(np.median(ear_vals))
    print(f'Calibration done. Baseline EAR = {baseline:.3f}')
    return baseline

# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description='Drowsiness detection (no-training mode supported)')
    parser.add_argument('--use_dlib', action='store_true', help='use dlib landmarks (requires predictor file)')
    parser.add_argument('--model', type=str, default=None, help='path to trained eye classifier (.h5)')
    parser.add_argument('--no_calib', action='store_true', help='skip calibration and use default thresholds')
    parser.add_argument('--src', default=0, help='video source (0 for webcam or path to video)')
    parser.add_argument('--no_window', action='store_true', help='do not show OpenCV window')
    args = parser.parse_args()

    # prepare source
    src = int(args.src) if str(args.src).isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Unable to open video source', src)
        return

    use_dlib = args.use_dlib and HAS_DLIB and os.path.exists(DLIB_LANDMARK_PATH)
    if args.use_dlib and not use_dlib:
        print('dlib not available or predictor not found; falling back to MediaPipe.')
    lp = LandmarkProvider(use_dlib=use_dlib)

    # optional calibration
    baseline = None
    if not args.no_calib:
        baseline = calibrate_baseline(cap, lp, num_frames=80, show=not args.no_window)
    # if calibration produced a baseline, reuse it

    # if we've used the capture for calibration, reset it (some cameras need reopen)
    try:
        cap.release()
    except Exception:
        pass
    cap = cv2.VideoCapture(src)

    # load eye model if provided (optional)
    eye_model = None
    if args.model is not None:
        if not HAS_TF:
            print('TensorFlow not available; running heuristic-only detector.')
        else:
            eye_model = tf.keras.models.load_model(args.model)
            print('Loaded eye model', args.model)

    dd = DrowsinessDetector(lp, eye_model=eye_model, show_window=(not args.no_window))
    if baseline is not None:
        dd.EAR_BASELINE = baseline

    dd.run(cap=cap)

if __name__ == '__main__':
    main()
