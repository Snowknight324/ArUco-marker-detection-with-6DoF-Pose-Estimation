import cv2
import numpy as np
import time

from modules.detector import ArucoDetector
from modules.distance import DistanceCalculator
from modules.orientation import OrientationCalculator
from modules.pose_estimation import PoseEstimator
from modules.utils import (
    draw_center_line,
    draw_crosshair,
    draw_marker_center,
    draw_pose_information,
    get_marker_center,
)

# --------------------------------------------------
# Load Camera Calibration
# --------------------------------------------------

data = np.load("calibration/calibration_data.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["distortion"]

MARKER_SIZE = 0.05  # Marker size in meters

# --------------------------------------------------
# Initialize Modules
# --------------------------------------------------

detector = ArucoDetector()

pose_estimator = PoseEstimator(camera_matrix, dist_coeffs, MARKER_SIZE)

# --------------------------------------------------
# Open Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# --------------------------------------------------
# FPS Variables
# --------------------------------------------------

prev_time = time.perf_counter()

fps = 0.0
# FPS smoothing factor
# Lower value  -> smoother FPS
# Higher value -> faster response
FPS_ALPHA = 0.1

# --------------------------------------------------
# Main Loop
# --------------------------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------
    # FPS Calculation
    # --------------------------------------------------

    current_time = time.perf_counter()

    elapsed_time = current_time - prev_time

    prev_time = current_time

    if elapsed_time > 0:
        instantaneous_fps = 1.0 / elapsed_time

        # Exponential moving average
        fps = FPS_ALPHA * instantaneous_fps + (1.0 - FPS_ALPHA) * fps

    # --------------------------------------------------
    # Draw FPS
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------
    # Draw Image Center
    # --------------------------------------------------

    draw_crosshair(frame)

    # --------------------------------------------------
    # Detect Markers
    # --------------------------------------------------

    corners, ids, rejected = detector.detect(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker in enumerate(corners):
            # ------------------------------------------
            # Pose Estimation
            # ------------------------------------------

            success, rvec, tvec = pose_estimator.estimate(marker)

            marker_id = int(np.asarray(ids[i]).item())

            if not success:
                continue

            # ------------------------------------------
            # Draw Coordinate Axes
            # ------------------------------------------

            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE
            )

            # ------------------------------------------
            # Distance Calculations
            # ------------------------------------------

            distance = DistanceCalculator.euclidean_distance(tvec)

            # ------------------------------------------
            # Orientation Calculations
            # ------------------------------------------

            roll, pitch, yaw = OrientationCalculator.euler_angles(rvec)

            # ------------------------------------------
            # Visualization
            # ------------------------------------------

            center = get_marker_center(marker[0])

            draw_marker_center(frame, center)

            draw_center_line(frame, center)

            draw_pose_information(frame, center, marker_id, distance, roll, pitch, yaw)

    # --------------------------------------------------
    # Show Output
    # --------------------------------------------------

    cv2.imshow("ArUco Pose Estimation", frame)

    # --------------------------------------------------
    # Keyboard Input
    # --------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
