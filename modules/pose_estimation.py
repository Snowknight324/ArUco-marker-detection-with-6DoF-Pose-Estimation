import cv2
import numpy as np


class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, marker_size):

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size

    def estimate(self, corners):

        success, rvec, tvec = cv2.solvePnP(
            self._object_points(),
            corners.reshape(-1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        return success, rvec, tvec

    def _object_points(self):

        s = self.marker_size / 2

        return np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )
