import numpy as np


class DistanceCalculator:
    @staticmethod
    def euclidean_distance(tvec):

        return np.linalg.norm(tvec)

    @staticmethod
    def x_distance(tvec):

        return float(tvec[0])

    @staticmethod
    def y_distance(tvec):

        return float(tvec[1])

    @staticmethod
    def z_distance(tvec):

        return float(tvec[2])
