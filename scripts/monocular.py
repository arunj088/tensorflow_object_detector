#!/usr/bin/env python

import numpy as np
from math import sin, cos, radians
from functools import reduce
from scipy.linalg import inv


# call the class with Monocular(intrinsic, height, pitch, yaw, roll, sensor_location)

class Monocular:

    def __init__(self, intrinsic, height, pitch, yaw, roll, sensor_location):
        self.intrinsic = intrinsic
        self.height = height
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.sensorLocation = sensor_location
        # world units is considered to be meters

    def rotX(self, x):
        x = float(radians(x))
        # Rotation matrix around X-axis
        R = np.array([[1.0, 0.0, 0.0], [0.0, cos(x), -sin(x)], [0.0, sin(x), cos(x)]])
        return R

    def rotY(self, y):
        y = float(radians(y))
        # Rotation matrix around Y-axis
        R = np.array([[cos(y), 0.0, sin(y)], [0.0, 1.0, 0.0], [-sin(y), 0.0, cos(y)]])
        return R

    def rotZ(self, z):
        z = float(radians(z))
        # Rotation matrix around Z-axis
        R = np.array([[cos(z), -sin(z), 0.0], [sin(z), cos(z), 0.0], [0.0, 0.0, 1.0]])
        return R

    @property
    def rotationMatrix(self):
        rot = np.dot(np.dot(np.dot(np.dot(self.rotY(180.0), self.rotZ(-90.0)),
                                   self.rotZ(-self.yaw)), self.rotX(90.0 - self.pitch)), self.rotZ(self.roll))
        return rot

    def translationVector(self):
        rot = reduce(np.dot, [self.rotZ(-self.yaw), self.rotX(90 - self.pitch),
                              self.rotZ(self.roll)])
        sl = self.sensorLocation
        translationInWorldUnits = np.array([sl[1], sl[0], self.height])
        translation = np.dot(translationInWorldUnits, rot)
        return translation

    def forwardAffineTransform(self, T, v1, v2):  # substitute for 'transformPointForward'
        u = np.array([v1, v2, 1.0])  # pad to make homogenous
        transform = np.dot(u, T)
        return transform

    def rawTformToImage3D(self):
        translation = self.translationVector()
        rotation = self.rotationMatrix
        tform = np.dot(np.vstack((rotation, translation)), self.intrinsic)
        return tform

    def tformToImage(self):
        camMatrix = self.rawTformToImage3D()
        # print(camMatrix)
        tform2D = np.array([camMatrix[0, :], camMatrix[1, :], camMatrix[3, :]])
        return tform2D  # haven't created an object like matlab. Using the T matrix directly

    def tformToVehicle(self):
        vehicletoimageTform = self.tformToImage()
        tform = inv(vehicletoimageTform)
        return tform  # inverse of T matrix is computed here. Need to be double if runs into issue

    def imageToVehicle(self, imagePoints):
        # check & report error if the image point value is not < 0.5 or > imagesize[1]+0.5 or > imagesize[0]+0.5
        vehiclePoints = self.forwardAffineTransform(self.tformToVehicle(), imagePoints[0], imagePoints[1])
        vehiclePoints = vehiclePoints[0:2]/vehiclePoints[2]
        vehiclePoints = [vehiclePoints[0], vehiclePoints[1]]
        return vehiclePoints

    def vehicleToImage(self, vehiclePoints):
        # check column size of vehiclePoints. Should be 2 or 3 i.e, [x,y] or [x,y,z] in vehicle coordinates
        size = np.size(vehiclePoints)
        if size == 2.0:
            # the vehicle points are 2D [x,y]
            imagePoints = self.forwardAffineTransform(self.tformToImage(),
                                                      np.array(vehiclePoints[0]), np.array(vehiclePoints[1]))
            imagePoints[0:2] = imagePoints[0:2] / imagePoints[2]  # need to make 2D compatible
            imagePoints = imagePoints[0:2]
            return imagePoints
        elif size == 3.0:
            # validate the vehicle points as real 2D single or double value
            pts = np.array([vehiclePoints[0], vehiclePoints[1], vehiclePoints[2], 1.0])
            imagePoints = np.dot(pts, self.rawTformToImage3D())
            if imagePoints.size == 0:
                imagePoints = np.zeros(2)
            else:
                imagePoints[0:2] = imagePoints[0:2] / imagePoints[2]  # need to make 2D compatible
                imagePoints = imagePoints[0:2]
            return imagePoints
        else:
            raise ValueError('World Coordinate is more than 3-Dimensional!')


"""
# test code for class 

# focal_length = np.array([800.0, 800.0])
# principal_Point = np.array([320.0, 240.0])
# image_size = np.array([480.0, 640.0], float)
h = 1.5
p = 0.0
world_cordinates = np.array([0.0, 0.0])
# assume we have camera intrinsic without calibration
# camera_intrinsic = np.array([[focal_length[0], 0.0, 0.0],
#                              [0.0, focal_length[1], 0.0],
#                              [principal_Point[0], principal_Point[1], 1.0]])
m = Monocular(np.array([[860.463418, 0.000000, 311.608199],
                        [0.000000, 869.417896, 287.737199],
                        [0.000000, 0.000000, 1.000000]]).T, h, p, 0.0, 0.0, world_cordinates)
value = m.imageToVehicle(np.array([50.0, 60.0], float))
pixel = m.vehicleToImage(np.array([20,0]))
homography = m.tformToImage()
rotation = m.rotationMatrix
translation = m.translationVector()
print(value)
print(homography)
print([int(homography[0,0]/homography[0,2]), int(homography[0,1]/homography[0,2])])  # horizon
"""