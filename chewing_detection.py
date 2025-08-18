#!/usr/bin/env python3
"""Chewing detection and YouTube control via computer vision.

This example script demonstrates how one might build a simple application that
pauses or resumes YouTube playback depending on whether a person in front of the
camera appears to be chewing.  The focus is on clarity and portability rather
than production‑grade accuracy.

Major steps
===========
1. **Video capture** – OpenCV grabs frames from the default camera in real time.
2. **Facial landmark detection** – ``dlib`` provides a pre‑trained 68‑point
   landmark model.  We locate the mouth region for each frame.
3. **Chewing heuristic** – The Euclidean distance between the upper and lower
   inner lip landmarks is tracked over time.  Large, periodic changes in this
   distance indicate chewing.
4. **Bite detection** – A simple mouth‑opening ratio identifies when a bite is
   about to be taken.  Lack of mouth opening for 30 seconds triggers a pause.
5. **Pause/Play logic** – If chewing is absent for 5 seconds we send a pause
   command.  When chewing or a new bite occurs we send play.
6. **YouTube control** – On macOS the script simulates a spacebar key press via
   :mod:`pyautogui`.  For iPadOS we provide a stub where integration with a
   Safari extension or custom URL scheme would occur.
7. **UI and override controls** – The live video feed is displayed with OpenCV,
   overlaying landmarks and status text.  Keyboard shortcuts allow manual
   pause/play overrides.  For iPadOS a GUI would require a different framework,
   so this portion is left as a placeholder.

The script is heavily commented so that individual pieces can be swapped out or
extended.  For example, the chewing heuristic can be replaced with a more
sophisticated model, and the YouTube control backend can be adapted to other
platforms.

Prerequisites
-------------
* ``opencv-python`` for video capture and drawing utilities.
* ``dlib`` and the ``shape_predictor_68_face_landmarks.dat`` model file.
* ``numpy`` for numeric operations.
* ``pyautogui`` on macOS for keyboard simulation (optional).

The code attempts to import these libraries gracefully; if any are missing the
script will exit with a message explaining what functionality is unavailable.
"""

from __future__ import annotations

import collections
import os
import platform
import time
from dataclasses import dataclass
from typing import Deque, Optional

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------
try:  # Numeric operations
    import numpy as np
except Exception:  # pragma: no cover - executed when numpy is missing
    np = None

try:  # Computer vision and video capture
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # Facial landmark detection
    import dlib
except Exception:  # pragma: no cover
    dlib = None

try:  # Simulate key presses on macOS
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
MOUTH_POINTS = list(range(48, 68))  # indices of mouth landmarks in 68-point model
UPPER_LIP = 62  # inner upper lip index
LOWER_LIP = 66  # inner lower lip index
LEFT_LIP = 48  # left mouth corner
RIGHT_LIP = 54  # right mouth corner


def load_landmark_predictor() -> Optional["dlib.shape_predictor"]:
    """Load the 68-point facial landmark predictor.

    The model file ``shape_predictor_68_face_landmarks.dat`` must be present in
    the working directory.  The function returns ``None`` if ``dlib`` is not
    available or the file is missing.
    """

    if dlib is None:
        return None
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(
            f"Missing '{predictor_path}'. Download it from:\n"
            "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
        return None
    return dlib.shape_predictor(predictor_path)


# ---------------------------------------------------------------------------
# Chewing detection logic
# ---------------------------------------------------------------------------
@dataclass
class ChewingDetector:
    """Classifies chewing based on mouth opening variability.

    Parameters
    ----------
    window_size:
        Number of recent mouth-opening measurements to retain.
    threshold:
        Standard deviation above which chewing is assumed.
    """

    window_size: int = 20
    threshold: float = 2.0

    def __post_init__(self) -> None:
        self.history: Deque[float] = collections.deque(maxlen=self.window_size)

    def update(self, distance: float) -> bool:
        """Update detector with the latest mouth opening distance.

        Returns ``True`` when chewing is detected.
        """

        self.history.append(distance)
        if len(self.history) < self.window_size:
            return False  # Not enough data yet
        if np is None:
            return False
        std = float(np.std(np.array(self.history)))
        return std > self.threshold


def is_mouth_open(coords: "np.ndarray", threshold: float = 0.6) -> bool:
    """Return ``True`` when the mouth is open wide enough to take a bite.

    A simple ratio of vertical lip distance to mouth width is used.  Values above
    ``threshold`` are considered an "open" mouth.
    """

    upper = coords[UPPER_LIP]
    lower = coords[LOWER_LIP]
    left = coords[LEFT_LIP]
    right = coords[RIGHT_LIP]
    vertical = float(np.linalg.norm(upper - lower))
    horizontal = float(np.linalg.norm(left - right))
    if horizontal == 0:
        return False
    return (vertical / horizontal) > threshold


# ---------------------------------------------------------------------------
# YouTube control
# ---------------------------------------------------------------------------
class YouTubeController:
    """Handle pause/play actions across platforms."""

    def __init__(self) -> None:
        self._playing = True
        self.platform = platform.system().lower()

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        if self.platform == "darwin" and pyautogui is not None:
            pyautogui.press("space")
        else:  # iPadOS or other platforms
            print("[PAUSE] (no-op)")

    def play(self) -> None:
        if self._playing:
            return
        self._playing = True
        if self.platform == "darwin" and pyautogui is not None:
            pyautogui.press("space")
        else:  # iPadOS or other platforms
            print("[PLAY] (no-op)")

# ---------------------------------------------------------------------------
# Video processing loop
# ---------------------------------------------------------------------------
def process_video() -> None:
    """Main loop that captures video and performs chewing detection."""

    if cv2 is None or dlib is None or np is None:
        print("Required libraries (cv2, dlib, numpy) are missing.")
        return

    predictor = load_landmark_predictor()
    if predictor is None:
        return

    detector = dlib.get_frontal_face_detector()
    chew_detector = ChewingDetector()
    controller = YouTubeController()
    last_chew_time = time.time()
    last_open_time = time.time()
    manual_override: Optional[bool] = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access camera.")
        return

    try:
        while True:  # pragma: no cover - requires camera input
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            chewing = False
            open_mouth = False

            for face in faces:
                shape = predictor(gray, face)
                coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                upper = coords[UPPER_LIP]
                lower = coords[LOWER_LIP]
                distance = float(np.linalg.norm(upper - lower))
                chewing = chew_detector.update(distance)
                open_mouth = is_mouth_open(coords)

                for i in MOUTH_POINTS:
                    x, y = coords[i]
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                break  # only process the first face

            now = time.time()
            if open_mouth:
                last_open_time = now

            if manual_override is not None:
                if manual_override:
                    controller.pause()
                    status_text = "Paused (manual override)"
                else:
                    controller.play()
                    status_text = "Playing (manual override)"
            elif chewing:
                last_chew_time = now
                controller.play()
                status_text = "Chewing"
            elif open_mouth:
                controller.play()
                status_text = "Mouth open"
            else:
                chew_inactive = now - last_chew_time > 5
                bite_inactive = now - last_open_time > 30
                if chew_inactive or bite_inactive:
                    controller.pause()
                    if chew_inactive and bite_inactive:
                        status_text = "Paused (no chewing/no bite)"
                    elif chew_inactive:
                        status_text = "Paused (no chewing)"
                    else:
                        status_text = "Paused (no bite)"
                else:
                    controller.play()
                    status_text = "Not chewing"

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if chewing else (0, 0, 255),
                2,
            )
            cv2.imshow("Chewing Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                manual_override = True
                controller.pause()
            elif key == ord("o"):
                manual_override = False
                controller.play()
            elif key == ord("a"):
                manual_override = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_video()
