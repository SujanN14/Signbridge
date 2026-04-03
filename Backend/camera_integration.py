import cv2
import joblib
import numpy as np
from collections import deque, Counter
import time
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf


class SignBridge:

    def __init__(self, model_path, landmark_model_path):
        # Load Keras MLP model
        self.model         = tf.keras.models.load_model(model_path)
        self.scaler        = joblib.load("feature_scaler.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")
        self.classes       = self.label_encoder.classes_

        print(f"[SignBridge] Model loaded  : {model_path}")
        print(f"[SignBridge] Input shape   : {self.model.input_shape}")
        print(f"[SignBridge] Classes ({len(self.classes)}): {self.classes}")

        # MediaPipe — 2 hands, EXACTLY like test_camera.py
        base_options = python.BaseOptions(model_asset_path=landmark_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Prediction stability
        self.prediction_buffer = deque(maxlen=7)
        self.confidence_buffer = deque(maxlen=7)
        self.top2_buffer       = deque(maxlen=15)
        self.current_word      = ""
        self.last_output       = None
        self.last_added_time   = 0

        # EMA smoother
        self.ema_probs = None
        self.ema_alpha = 0.4

        # Settings
        self.confidence_threshold = 0.70
        self.cooldown_time        = 1.2
        self.prev_landmarks       = None
        self.max_velocity         = 0.35
        self.debug_mode           = True

    # ── Feature Extraction — EXACTLY matches test_camera.py ─────────────────

    def _normalize_hand(self, hand_landmarks):
        coords  = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        wrist   = coords[0]
        mcp_mid = coords[9]  # landmark 9 — matches test_camera.py exactly
        scale   = float(np.linalg.norm(mcp_mid - wrist))
        if scale < 1e-6:
            return None
        return ((coords - wrist) / scale).flatten()

    def extract_features(self, hand_landmarks_list):
        feature_vector = np.zeros(126, dtype=np.float32)
        hands = sorted(hand_landmarks_list, key=lambda h: h[0].x)
        slot_offset = 0
        for hand in hands[:2]:
            normalized = self._normalize_hand(hand)
            if normalized is not None:
                feature_vector[slot_offset: slot_offset + 63] = normalized
            slot_offset += 63
        raw    = feature_vector.reshape(1, -1)
        scaled = self.scaler.transform(raw).astype(np.float32)
        return scaled

    # ── Velocity Gate ────────────────────────────────────────────────────────

    def _hand_velocity_ok(self, hand_landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        if self.prev_landmarks is None:
            self.prev_landmarks = pts
            return True
        vel = np.linalg.norm(pts - self.prev_landmarks, axis=1).mean()
        self.prev_landmarks = pts
        return vel < self.max_velocity

    # ── EMA Smoother ─────────────────────────────────────────────────────────

    def _smooth_probs(self, probs):
        if self.ema_probs is None or self.ema_probs.shape != probs.shape:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
        return self.ema_probs

    # ── Core Prediction ──────────────────────────────────────────────────────

    def predict_frame(self, frame):

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))

        if not result.hand_landmarks:
            self.prev_landmarks = None
            self.ema_probs      = None
            self._status(frame, "No hand detected", (100, 100, 100))
            return frame

        if not self._hand_velocity_ok(result.hand_landmarks[0]):
            self._status(frame, "MOTION — stabilising", (0, 165, 255))
            return frame

        # Draw skeleton
        h, w, _ = frame.shape
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (80, 180, 80), 1)
            for pt in pts:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        features   = self.extract_features(result.hand_landmarks)
        probs      = self.model.predict(features, verbose=0)
        probs      = self._smooth_probs(probs)
        confidence = float(np.max(probs))
        pred_idx   = int(np.argmax(probs))
        pred       = self.classes[pred_idx]
        sorted_idx = np.argsort(probs[0])[::-1]
        runner_up  = self.classes[sorted_idx[1]]
        runner_conf= float(probs[0][sorted_idx[1]])

        # HUD
        bar_w     = int(confidence * 200)
        bar_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 100, 255)
        cv2.rectangle(frame, (25, 105), (225, 125), (40, 40, 40), -1)
        cv2.rectangle(frame, (25, 105), (25 + bar_w, 125), bar_color, -1)
        cv2.putText(frame, f"Pred: {pred} ({confidence:.2f})",
                    (30, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
        cv2.putText(frame, f"2nd:  {runner_up} ({runner_conf:.2f})",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 0), 1)

        # Debug top-5
        if self.debug_mode:
            for rank, idx in enumerate(sorted_idx[:5]):
                lbl   = self.classes[idx]
                p     = float(probs[0][idx])
                y     = 200 + rank * 28
                bw    = int(p * 180)
                color = (0, 200, 100) if rank == 0 else (80, 80, 200)
                cv2.rectangle(frame, (30, y - 16), (30 + bw, y), color, -1)
                cv2.putText(frame, f"{lbl}: {p:.2f}", (35, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Weighted vote
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append(pred)
            self.confidence_buffer.append(confidence)
        self.top2_buffer.append((pred, runner_up))

        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            vote_weights = {}
            for p, c in zip(self.prediction_buffer, self.confidence_buffer):
                vote_weights[p] = vote_weights.get(p, 0.0) + c
            final_pred = max(vote_weights, key=vote_weights.get)

            candidates = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
            if len(candidates) > 1 and (candidates[0][1] - candidates[1][1]) < 0.5:
                top2_flat = [p for pair in self.top2_buffer for p in pair]
                best = Counter(top2_flat).most_common(1)
                if best:
                    final_pred = best[0][0]

            current_time = time.time()
            if (final_pred != self.last_output and
                    current_time - self.last_added_time > self.cooldown_time):
                self.current_word    += final_pred
                self.last_output      = final_pred
                self.last_added_time  = current_time

        return frame

    def _status(self, frame, msg, color):
        cv2.putText(frame, msg, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def main():
    MODEL_PATH     = "isl_twohand_mlp.h5"
    LANDMARK_MODEL = "hand_landmarker.task"

    signbridge = SignBridge(MODEL_PATH, LANDMARK_MODEL)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    fps_counter = deque(maxlen=30)
    prev_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = signbridge.predict_frame(frame)

        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Smart word display — auto-shrink font and wrap to fit screen
        word = signbridge.current_word
        frame_w = frame.shape[1]
        label = "Word: "
        full_text = label + word

        # Try decreasing font sizes until it fits
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        for scale in [1.0, 0.85, 0.7, 0.55, 0.45]:
            (tw, th), _ = cv2.getTextSize(full_text, font, scale, thickness)
            if tw < frame_w - 40:
                font_scale = scale
                break

        # If still too long, show only last N chars that fit
        display_text = full_text
        (tw, _), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        if tw > frame_w - 40:
            # Trim from the left, keep recent letters visible
            for i in range(len(full_text)):
                trimmed = "..." + full_text[i:]
                (tw, _), _ = cv2.getTextSize(trimmed, font, font_scale, thickness)
                if tw < frame_w - 40:
                    display_text = trimmed
                    break

        cv2.putText(frame, display_text,
                    (20, 50), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"FPS: {np.mean(fps_counter):.1f}",
                    (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "[C] Clear  [BKSP] Delete  [D] Debug  [Q] Quit",
                    (30, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("SignBridge - Master Edition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            signbridge.current_word = ""
            signbridge.prediction_buffer.clear()
            signbridge.confidence_buffer.clear()
            signbridge.top2_buffer.clear()
            signbridge.ema_probs   = None
            signbridge.last_output = None
        elif key == 8:
            signbridge.current_word = signbridge.current_word[:-1]
        elif key == ord('d'):
            signbridge.debug_mode = not signbridge.debug_mode
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()