#!/usr/bin/env python
'''
try:
     import cv2.cv2 as cv2
except ImportError:
    pass
'''
import yaml, time
import numpy as np
import matplotlib.pyplot as plt
import monocular as monocular
import cv2
import rospy
from barc.msg import Lanepoints

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridgeError, CvBridge


class ImageProcess:
    def __init__(self):
        rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.image_proc, queue_size=1)
        self.pub = rospy.Publisher('/proc_image', Image, queue_size=1)
        self.lane_pub = rospy.Publisher('/lane_loc',Lanepoints, queue_size=10)
        self.Bridge = CvBridge()
        # self.C_PATH = '/home/aj/Curved-Lane-Lines/calibrationdata_logitech/head_camera.yaml'  # yaml file created by ros node
        # with open(self.C_PATH, 'r') as stream:
        #     data = yaml.safe_load(stream)
        #     mtx = data['camera_matrix']
        #     dist = data['distortion_coefficients']
        #     (rows, cols, data) = (mtx['rows'], mtx['cols'], mtx['data'])
        #     mtx = np.array(data, dtype=float).reshape(rows, cols)
        #     (rows, cols, data) = (dist['rows'], dist['cols'], dist['data'])
        #     dist = np.array(data, dtype=float).reshape(rows, cols)
        #     self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 1, (640, 480))
        # self.mono = monocular.Monocular(np.array([[860.463418, 0.000000, 311.608199],
        #                          [0.000000, 869.417896, 287.737199],
        #                          [0.000000, 0.000000, 1.000000]]).T, 1.2, 7.2, 0.6, 0, np.array([0.0, 0.0], dtype=np.float))

    def h_transform(self,u,v,H):
        tx = (H[0,0]*u + H[0,1]*v + H[0,2])
        ty = (H[1,0]*u + H[1,1]*v + H[1,2])
        tz = (H[2,0]*u + H[2,1]*v + H[2,2])
        px = (tx / tz)
        py = (ty / tz)
        Z = (1 / tz)
        return (px, py)

    def image_proc(self, data):
        # C_PATH = '/home/aj/Curved-Lane-Lines/calibrationdata_logitech/head_camera.yaml'  # yaml file created by ros node
        # with open(C_PATH, 'r') as stream:
        #     data = yaml.safe_load(stream)
        #     mtx = data['camera_matrix']
        #     dist = data['distortion_coefficients']
        #     (rows, cols, data) = (mtx['rows'], mtx['cols'], mtx['data'])
        #     mtx = np.array(data, dtype=float).reshape(rows, cols)
        #     (rows, cols, data) = (dist['rows'], dist['cols'], dist['data'])
        #     dist = np.array(data, dtype=float).reshape(rows, cols)
        #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 1, (640, 480))
        # mono = monocular.Monocular(newcameramtx.T,1.2,3.2,0,0,np.array([0.0, -0.02], dtype=np.float))
        t0 = time.time()
        try:
            cv_img = self.Bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv_img = cv2.imread('/home/aj/Curved-Lane-Lines/test_images/prescan2.jpg')
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # cv_img = cv2.undistort(cv_img, mtx, dist, None, mtx)
        hls = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HLS)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        r_channel = cv_img[:,:,0]
        g_channel = cv_img[:,:,1]
        b_channel = cv_img[:,:,2]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        h_channel = hls[:,:,0]

        # cv2.imshow('l_channel', l_channel)
        # cv2.imshow('s_channel', s_channel)
        # cv2.imshow('h_channel', h_channel)
        # cv2.imshow('canny',canny)
        # cv2.waitKey(0)
        # l_channel[()]

        s_channel[(s_channel <= 45)] = 0
        s_channel[:s_channel.shape[0]//2+5,:] = 3

        l_channel[(l_channel <= 40) | (l_channel >= 55)] = 0
        l_channel[:l_channel.shape[0] // 2+25 , :] = 0
        canny = cv2.Canny(r_channel, 30, 95)
        canny[:canny.shape[0]//2,:] = 0


        # cv2.imshow('canny',canny)
        # cv2.waitKey(0)
        # h_channel[(h_channel != 0)] = 1
        height, width = cv_img.shape[0], cv_img.shape[1]
        # gray
        s_binary = np.zeros_like(gray)
        s_binary[(s_channel == 0)] = 1  # filter the region of interest from the image
        s_binary_3 = np.dstack((s_binary,s_binary,s_binary))
        test = cv_img*s_binary_3
        # test = gray*s_binary
        test1d = np.multiply(gray, s_binary)
        new_binary = np.zeros_like(test)
        new_binary[(test > 160)] = 255
        # cv2.imshow('s_channel', s_channel)
        # cv2.imshow('l_channel', new_binary)
        # cv2.imshow('s_channel', cv_img)
        # cv2.waitKey(3)
        # warping
        src = np.float32([(0.4281*width, 0.442*height), (0.5703*width,0.442*height),
                          (0.0109* width,0.7916* height), (0.942*width,0.7916*height)])
        dst = np.float32([(0.4281*width, 0.442*height), (0.5703*width,0.442*height),
                          (0.4281* width,0.7916* height), (0.5703*width,0.7916*height)])
        mat = cv2.getPerspectiveTransform(src, dst)
        inv_mat = cv2.getPerspectiveTransform(dst,src)
        warped_img = cv2.warpPerspective(cv_img,mat, (cv_img.shape[1], cv_img.shape[0]))
        warped_mask = cv2.warpPerspective(new_binary ,mat, (cv_img.shape[1], cv_img.shape[0]))
        warped_canny = cv2.warpPerspective(canny ,mat, (cv_img.shape[1], cv_img.shape[0]))
        hist = np.sum(warped_mask[:,:,1][:warped_mask.shape[0],],axis=0)
        # cv2.imshow('warped_mask', warped_canny)
        # cv2.waitKey(0)
        # plt.plot(hist)
        # plt.show()
        midpoint = int(hist.shape[0]/2)
        left_x_base = midpoint//3 + np.argmax(hist[midpoint//3:midpoint])
        right_x_base = np.argmax(hist[midpoint:]) + midpoint
        #sliding window & curve fitting
        left_a, left_b, left_c = [],[],[]
        right_a, right_b, right_c = [],[],[]
        nwindows = 9
        margin = 40
        minpix = 1
        draw_windows=True
        left_fit_ = np.empty(3)
        right_fit_ = np.empty(3)
        windows_height = np.int(warped_mask.shape[0]/nwindows)
        nonzero = warped_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_x_current = left_x_base
        right_x_current = right_x_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped_mask.shape[0] - (window+1)*windows_height
            win_y_high = warped_mask.shape[0] - window*windows_height
            win_xleft_low = left_x_current - margin
            win_xleft_high = left_x_current + margin
            win_xright_low = right_x_current - margin
            win_xright_high = right_x_current + margin
            if draw_windows:
                cv2.rectangle(warped_mask, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                              (100, 255, 255), 3)
                cv2.rectangle(warped_mask, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (100, 255, 255), 3)
            # identify the nonzero pixels in x and y within windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_x_current = np.int(np.mean(nonzerox[good_right_inds]))


        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # Generate x and y values for plotting
        if max(righty) >= max(lefty):
            limit = max(righty)
        else:
            limit = max(lefty)
        ploty = np.linspace(0, limit - 1, limit)  # righty[0] == lefty[0]
        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        # plt.plot(left_fitx, ploty, right_fitx, ploty)
        # plt.show()
        warped_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        warped_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
        color_img = np.zeros_like(warped_img)
        lane_img = color_img
        for a in range(len(left_fitx)):
            cv2.circle(lane_img, (int(left_fitx[a]), int(ploty[a])), 5, (255, 0, 0), thickness=-1)
        for b in range(len(right_fitx)):
            cv2.circle(lane_img, (int(right_fitx[b]), int(ploty[b])), 5, (0, 255, 0), thickness=-1)
        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((left, right))
        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
        inv_p_wrap = cv2.warpPerspective(color_img, inv_mat, (cv_img.shape[1], cv_img.shape[0]))
        inv_lane_warp = cv2.warpPerspective(lane_img, inv_mat, (cv_img.shape[1], cv_img.shape[0]))
        inv_p_wrap = cv2.addWeighted(cv_img, 1, inv_p_wrap, 0.7, 0)
        inv_p_wrap = cv2.addWeighted(cv_img, 1, inv_lane_warp, 0.5, 0)
        inv_p_wrap = cv2.cvtColor(inv_p_wrap, cv2.COLOR_BGR2RGB)
        img_plane_midpoint = np.dot(inv_mat, np.float32([(right_fitx[-1]-left_fitx[-1])/2+left_fitx[-1], ploty[-1], 1]).T)
        img_plane_midpoint = np.array([img_plane_midpoint[0]/img_plane_midpoint[2], img_plane_midpoint[1]/img_plane_midpoint[2]],dtype=int)
        offset = cv_img.shape[1]//2 - img_plane_midpoint[0]
        index_y_ref = height - 40
        homography = self.mono.tformToImage()
        horizon = [int(homography[0,0]/homography[0,2]), int(homography[0,1]/homography[0,2])]
        cv2.rectangle(inv_p_wrap, (400,100), (640,0), (255,255,255), thickness = -1, lineType = 8, shift = 0)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(inv_p_wrap, "information box", (465, 20), fontFace, .5,(0,0,0))
        cv2.putText(inv_p_wrap, "pixel offset: "+str(offset) + " pxls", (440, 50), fontFace, .5,(0,0,0))
        cv2.line (inv_p_wrap, (img_plane_midpoint[0], index_y_ref-35),(img_plane_midpoint[0], index_y_ref+35), (255,255,255),3)
        cv2.line(inv_p_wrap, (width//2, index_y_ref-5-35),(width//2, index_y_ref-5+35), (0,0,0),3)
        cv2.line(inv_p_wrap, (0, horizon[1]), (width, horizon[1]), (255, 255, 255), 1)  # horizon line ######################
        cv2.line(inv_p_wrap, (horizon[0], 0), (horizon[0], height), (255, 255, 255),1)  # horizon line ######################
        cv2.putText(inv_p_wrap, str(offset), (width//2, 255), fontFace, .5, (0, 0, 255))
        if offset > 0:
            cv2.arrowedLine(inv_p_wrap, (width//2 + 40, 250), (width//2 + 90, 250), (255, 0, 0), 2, 8, 0, 0.5)
        elif offset < 0:
            cv2.arrowedLine(inv_p_wrap, (width//2 - 10, 250), (width//2 - 60, 250), (255, 0, 0), 2, 8, 0, 0.5)
        else:
            cv2.arrowedLine(inv_p_wrap, (width//2, 230), (width//2, 180), (255, 0, 0), 2, 8, 0, 0.5)
        image_out = self.Bridge.cv2_to_imgmsg(inv_p_wrap,"bgr8")
        self.pub.publish(image_out)
        # To Display Image
        cv2.imshow('new_img', inv_p_wrap)
        cv2.waitKey(3)
        ### world cordinate calculation ###

        image_lane_loc = []
        image_pts = []
        for idx in range(len(ploty)):
            llane = self.h_transform(left_fitx[idx], ploty[idx], inv_mat)
            w_llane = self.mono.imageToVehicle(llane)
            rlane = self.h_transform(right_fitx[idx], ploty[idx], inv_mat)
            w_rlane = self.mono.imageToVehicle(rlane)
            if len(image_lane_loc) is 0:
                image_lane_loc = np.array([llane,rlane]).reshape(-1)
                image_pts = np.array([w_llane, w_rlane]).reshape(-1)
            else:
                image_lane_loc = np.vstack((image_lane_loc,np.array([llane,rlane]).reshape(-1)))
                image_pts = np.vstack((image_pts,np.array([w_llane,w_rlane]).reshape(-1)))
        (row, col) = (image_lane_loc.shape[0], image_lane_loc.shape[1])
        # plt.plot(image_lane_loc[:,0], 480-image_lane_loc[:,1], image_lane_loc[:,2],480-image_lane_loc[:,3])
        # plt.show()
        # plt.plot(image_pts[:,1], image_pts[:,0], image_pts[:,3], image_pts[:,2])
        # plt.xlim((5,-5))
        # plt.show()
        ros_msg = image_lane_loc.reshape(-1)
        msg = Lanepoints()
        msg.rows = int(row)
        msg.cols = int(col)
        msg.loc = ros_msg
        self.lane_pub.publish(msg)
        tf = time.time()
        fps = 1/(tf-t0)
        print(fps)
        # plt.plot(image_lane_loc[:, 0], height - image_lane_loc[:, 1], image_lane_loc[:, 2], height - image_lane_loc[:, 3])
        # plt.xlim((5, -5))
        # plt.show()
        # plt.plot(left_fitx, ploty, right_fitx, ploty)
        # plt.show()


def startup():
    rospy.init_node('Image_proc')
    ImageProcess()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        startup()
    except rospy.ROSInterruptException:
        pass
