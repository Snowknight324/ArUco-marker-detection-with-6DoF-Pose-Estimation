import cv2
import numpy as np


def get_marker_center(corners):
    """
    Returns the pixel coordinates of the center of an ArUco marker.

    Parameters:
        corners (ndarray): Marker corners returned by ArUco detector.

    Returns:
        tuple: (center_x, center_y)
    """

    center_x = int(np.mean(corners[:, 0]))
    center_y = int(np.mean(corners[:, 1]))

    return center_x, center_y


def draw_crosshair(frame):
    """
    Draw a crosshair at the image center.
    """

    h, w = frame.shape[:2]

    cx = w // 2
    cy = h // 2

    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)


def draw_marker_center(frame, center):
    """
    Draws a small circle at the marker center.
    """

    cv2.circle(frame, center, 5, (0, 0, 255), -1)


def draw_center_line(frame, center):
    """
    Draws a line from the image center to the marker center.
    """

    h, w = frame.shape[:2]

    image_center = (w // 2, h // 2)

    cv2.line(frame, image_center, center, (255, 255, 0), 2)


def draw_pose_information(frame, center, marker_id, distance, roll, pitch, yaw):
    """
    Draws marker ID, distance and orientation on the frame.
    """

    x, y = center

    cv2.putText(
        frame,
        f"ID: {marker_id}",
        (x, y - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Distance: {distance:.2f} m",
        (x, y - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Roll : {roll:.1f}",
        (x, y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Pitch: {pitch:.1f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Yaw  : {yaw:.1f}",
        (x, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )
