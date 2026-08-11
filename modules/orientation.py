import cv2
import numpy as np


class OrientationCalculator:
    @staticmethod
    def rotation_matrix(rvec):

        R, _ = cv2.Rodrigues(rvec)
        return R

    @staticmethod
    def euler_angles(rvec):

        R, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])

        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.degrees([roll, pitch, yaw])
