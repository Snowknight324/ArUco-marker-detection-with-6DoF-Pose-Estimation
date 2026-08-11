A modular computer vision project using Python and OpenCV for detecting ArUco markers and estimating their 6-DoF pose relative to a camera.

The project includes laptop webcam calibration using a chessboard pattern to obtain the camera matrix and distortion coefficients, followed by ArUco marker detection and pose estimation. The system calculates the marker's 3D position, distance from the camera, and orientation (roll, pitch, and yaw).

The implementation is structured into separate modules for camera calibration, marker detection, distance estimation, and pose estimation, making the code easier to understand, reuse, and extend for robotics and computer vision applications.

Key Features
Laptop webcam calibration using a chessboard calibration pattern
Camera intrinsic parameter and distortion estimation
ArUco marker detection using OpenCV
3D position and distance estimation
6-DoF pose estimation
Roll, pitch, and yaw calculation
Modular Python implementation
