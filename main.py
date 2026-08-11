import cv2
import numpy as np

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
# Main Loop
# --------------------------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Draw image center
    draw_crosshair(frame)

    # Detect markers
    corners, ids, rejected = detector.detect(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker in enumerate(corners):
            success, rvec, tvec = pose_estimator.estimate(marker)
            marker_id = int(np.asarray(ids[i]).item())

            if not success:
                continue

            # Draw coordinate axes
            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE
            )

            # -----------------------------------------
            # Distance Calculations
            # -----------------------------------------

            distance = DistanceCalculator.euclidean_distance(tvec)

            # -----------------------------------------
            # Orientation Calculations
            # -----------------------------------------

            roll, pitch, yaw = OrientationCalculator.euler_angles(rvec)

            # -----------------------------------------
            # Visualization
            # -----------------------------------------

            center = get_marker_center(marker[0])

            draw_marker_center(frame, center)

            draw_center_line(frame, center)

            draw_pose_information(frame, center, marker_id, distance, roll, pitch, yaw)

    # Show output
    cv2.imshow("ArUco Pose Estimation", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
