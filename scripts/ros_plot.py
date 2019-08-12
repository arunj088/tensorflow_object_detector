#!/usr/bin/env python
import rospy
from math import sin, cos, radians
import numpy as np
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from barc.msg import Lanepoints
import monocular as mono



xy_loc = np.zeros((10,2))
lane_loc = np.zeros((10,4))
vehicle_pos = np.zeros((10,2))



def animate(xy_loc, fig, ax):
    global lane_loc, vehicle_pos
    # circle3 = plt.Circle((0, 0), 13, color='r', clip_on=True, fill = False)
    ax.clear()
    # ax.scatter(xy_loc[:, 1], xy_loc[:, 0], s=30, c=np.array([[1,0,0]]))
    if not vehicle_pos[0,0] < 0:
        ax.add_patch(patches.Rectangle(((vehicle_pos[0,1]+(vehicle_pos[0,1]//2)),vehicle_pos[0,0]),2.0,2.0, fill=True, alpha =2))
    # ax.plot(vehicle_pos[:,1],vehicle_pos[:,0], 'bo')
    if 0 < vehicle_pos[0,0] <= 7.5:
        car_length = 2.0
        for i in range(len(lane_loc)):
            if (lane_loc[i,0]>=vehicle_pos[0,0]-car_length) and (lane_loc[i,0] <= (vehicle_pos[0,0] + car_length*2)):
                lane_loc[i,1] = lane_loc[i,1]+3.5 # left lane
            if (lane_loc[i,0]>=vehicle_pos[0,0]) and ((lane_loc[i,0]) <= (vehicle_pos[0,0] + car_length)):
                lane_loc[i,3] = lane_loc[i,3] + 3.5 # right lane
        ax.plot(lane_loc[:, 1], lane_loc[:, 0], lane_loc[:, 3], lane_loc[:, 2])
    else:
        ax.plot(lane_loc[:,1], lane_loc[:,0], lane_loc[:,3], lane_loc[:,2])
    # ax.add_artist(circle3)
    # ax.plot([0,-1],[0,5])
    ax.set_xlim((7, -7))
    # ax.set_ylim((0, 80))


def lanelocCallback(msg):
    row = msg.rows
    col = msg.cols
    global lane_loc
    image_pts = np.array(msg.loc).reshape((row, col))
    lane_loc = image_pts

def vehicleptsCallback(msg):
    row = msg.rows
    col = msg.cols
    image_pts = np.array(msg.loc).reshape((row, col))
    global vehicle_pos
    vehicle_pos = image_pts

#
# def laserCallback(data):
#     global xy_loc
#     max = data.range_max
#     min = data.range_min
#     angle_min = data.angle_min
#     angle_max = data.angle_max
#     angle_increment = data.angle_increment
#     time_increment = data.time_increment
#     length = int((angle_max-angle_min)/angle_increment)+1
#     xy_location = []
#     angle_inc = radians(360/length)
#     for i in range(length):
#         if min <= data.ranges[i] <= max:
#             xy_location.append((data.ranges[i]*cos(i*angle_inc), data.ranges[i] * sin(i * angle_inc)))
#     xy_location = np.array(xy_location).reshape((-1,2))
#     xy_loc = xy_location


if __name__ == '__main__':
    rospy.init_node('laser_scanner')
    hz = 10
    # bridge = CvBridge()
    rate = rospy.Rate(hz)
    # m = mono.Monocular(np.array([[860.463418, 0.000000, 311.608199],
    #                              [0.000000, 869.417896, 287.737199],
    #                              [0.000000, 0.000000, 1.000000]]).T, 1.2, 3.55, 0.0, 0.0, np.array([0.0, 0.0]))
    # fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    # rospy.Subscriber('/scan', LaserScan, laserCallback, queue_size=1)
    rospy.Subscriber('/lane_loc',Lanepoints, lanelocCallback, queue_size=1)
    rospy.Subscriber('/vehicle_loc',Lanepoints, vehicleptsCallback, queue_size=1)

    while not rospy.is_shutdown():
        animate(xy_loc, fig, ax)
        fig.canvas.draw()
        plt.show()
        rate.sleep()



