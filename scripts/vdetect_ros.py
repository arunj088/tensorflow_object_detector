#!/usr/bin/env python
# Purpose: Ros node to detect objects using tensorflow

import os
import sys
import cv2
import numpy as np
import monocular as mono
import time
import matplotlib.pyplot as plt
# ROS related imports
import rospy
from barc.msg import Lanepoints
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
#from geometry_msgs.msg import Vector3
# Object detection module imports
import object_detection
import tensorflow as tf
'''
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)
'''
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.7

######### Set model here ############
# MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'data','models' , MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
# LABEL_NAME = 'labelmap.pbtxt'
LABEL_NAME = 'mscoco_label_map.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]),'data','labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 90  # 90||2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION


# lane detection #######################################################################################################


#######################################################################################################################

# Detection

class Detector:

    def __init__(self):
        self.image_pub = rospy.Publisher("debug_image",Image, queue_size=1)
        self.map_pub = rospy.Publisher("world_lane",Image, queue_size=1)
        self.object_pub = rospy.Publisher("objects", Detection2DArray, queue_size=1)
        self.loc_pub = rospy.Publisher("vehicle_loc",Lanepoints, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image,
                                          self.image_cb, queue_size=1, buff_size=2**24)
        self.sess = tf.Session(graph=detection_graph,config=config)
        # intrinsic matrix is transposed as monocular class expects a transposed matrix
        # monocular class can be modified to make the matrix transposed
        self.m = mono.Monocular(np.array([[860.463418, 0.000000, 311.608199],
                                         [0.000000, 869.417896, 287.737199],
                                         [0.000000, 0.000000, 1.000000]]).T, 1.2, 3.55, 0.0, 0.0, np.array([0.0, 0.0]))

    # def draw_lines(self, img, edges, color=[0, 0, 255], thickness=3):
    #
    #     height, width = edges.shape
    #     offset = 0
    #     y_llimit = 130
    #     y_ulimit = 225
    #     for y_base in range(y_llimit, y_ulimit, 1):
    #         # y_base = 150 #20 # this represents 3" in front of the car
    #         index_y = height - y_base
    #         index_x = width // 2
    #         if index_x >= width:
    #             index_x = width - 1
    #         if index_x < 0:
    #             index_x = 0
    #         x_left, x_right = self.find_offset_in_lane(edges, index_x, index_y, width)
    #         midpoint = (x_right + x_left) // 2
    #
    #         if y_base == y_llimit:
    #             pts0 = np.array([(x_left, index_y), (x_right, index_y)], dtype=np.int32)
    #
    #         if y_base % 2 == 0:
    #             pts = np.array([(x_left, index_y), (x_right, index_y)], dtype=np.int32)
    #         else:
    #             pts = np.array([(x_right, index_y), (x_left, index_y)], dtype=np.int32)
    #
    #         if y_base == y_llimit:
    #             offset = midpoint - width // 2
    #             midpoint_ref = midpoint
    #             index_y_ref = index_y
    #
    #         pts0 = np.concatenate((pts0, pts))
    #         ###########################################################
    #         cv2.circle(img, (x_right, index_y), 3, (0, 255, 255), -1)
    #         cv2.circle(img, (x_left, index_y), 3, (0, 255, 255), -1)
    #     ###############################################################
    #     # cv2.fillPoly(img, [pts0], (0, 255, 0))
    #     ###############################################################
    #     lane_loc = pts0.reshape((-1, 4))
    #     for row_idx in range(lane_loc.shape[0]):
    #         if lane_loc[row_idx][0] > lane_loc[row_idx][2]:
    #             temp0 = lane_loc[row_idx][0]
    #             temp2 = lane_loc[row_idx][2]
    #             (lane_loc[row_idx][0], lane_loc[row_idx][2]) = (temp2, temp0)
    #
    #     # plt.plot(lane_loc[:, 0], 480-lane_loc[:, 1], lane_loc[:, 2], 480-lane_loc[:, 3])
    #     # plt.xlim(0, 640)
    #     # plt.ylim(0, 480)
    #     # plt.show()
    #     world = []
    #     for idx in range(lane_loc.shape[0]):
    #         points = [self.m.imageToVehicle((lane_loc[idx, 0], lane_loc[idx, 1])),
    #                   self.m.imageToVehicle((lane_loc[idx, 2], lane_loc[idx, 3]))]
    #         points = np.reshape(points, (1, 4)).astype(np.float32)
    #         if len(world) is 0:
    #             world = points
    #         world = np.vstack((world, points))
    #     #########################################################################################
    #     '''
    #     fig = plt.figure()
    #     plt.plot(world[:, 1], world[:, 0], world[:, 3], world[:, 2])
    #     plt.xlim((5, -5))
    #     fig.canvas.draw()
    #     # convert canvas to image
    #     plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    #                          sep='')
    #     plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    #     # img is rgb, convert to opencv's default bgr
    #     plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
    #     # plt.show(fig)
    #     '''
    #     #########################################################################################
    #     fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    #     cv2.putText(img, str(offset), (midpoint_ref, 255), fontFace, .5, (0, 0, 255))
    #     if offset > 0:
    #         cv2.arrowedLine(img, (midpoint_ref + 40, 250), (midpoint_ref + 90, 250), (255, 0, 0), 2, 8, 0, 0.5)
    #     elif offset < 0:
    #         cv2.arrowedLine(img, (midpoint_ref - 10, 250), (midpoint_ref - 60, 250), (255, 0, 0), 2, 8, 0, 0.5)
    #     else:
    #         cv2.arrowedLine(img, (midpoint_ref, 230), (midpoint_ref, 180), (255, 0, 0), 2, 8, 0, 0.5)
    #
    #     cv2.line(img, (midpoint_ref, index_y_ref - 35), (midpoint_ref, index_y_ref + 35), (255, 255, 255), 3)
    #     cv2.line(img, (width // 2, index_y_ref - 5 - 35), (width // 2, index_y_ref - 5 + 35), (0, 0, 0), 3)
    #
    #     return offset, img, pts0, world
    #
    # def find_offset_in_lane(self, img, x, y, width):
    #     """
    #     Returns the difference in x and y positions
    #     operates on pixels. Return value is pixel offset from nominal
    #     """
    #     x_left = x
    #     x_right = x
    #     while (x_left > 0 and not img[y, x_left]):
    #         x_left = x_left - 1
    #     while (x_right < width and not img[y, x_right]):
    #         x_right = x_right + 1
    #     return (x_left, x_right)

    def image_cb(self, data):
        objArray = Detection2DArray()
        time_i = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # ROI
        ypixel_offset_l = image.shape[0]//2-50
        ypixel_offset_u = image.shape[0] - 200
        roi_image = image[ypixel_offset_l:ypixel_offset_u,:,:]
        image_np = np.asarray(roi_image)
        image_np_real = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np_real, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        min_score = 0.2
        category_id = 3  # 3 for mobilnet/ 1 for kitte

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        (boxes, scores, classes) = self.filter_boxes(min_score, np.squeeze(boxes),
                                                     np.squeeze(scores), np.squeeze(classes).astype(np.int32), [category_id])
        # ht, wdt, chn = image[ypixel_offset_l:ypixel_offset_u, :, :].shape
        # (boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]) = ((boxes[:,0]*ht+ypixel_offset_l), boxes[:,1]*image.shape[1],
        #                                                     (boxes[:,2]*ht+ypixel_offset_l), boxes[:,3]*image.shape[1])  # [y_min, x_min, y_max, x_max]
        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        #############  new logic
        ht, wdt, chn = image.shape
        #############

        objArray.detections = []
        objArray.header = data.header
        object_count = 1
        bbox = []
        x_mid = []
        y_mid = []
        image_loc = []
        xy_range = []

        for i in range(len(objects)):
            object_count+=1
            objArray.detections.append(self.object_predict(objects[i],data.header,image_np,cv_image))
            bbox.append(objects[i][2])   # [y_min, x_min, y_max, x_max]
            x_mid.append(int((bbox[i][3] - bbox[i][1])*wdt/2) + int((bbox[i][1])*wdt))
            y_mid.append(int((bbox[i][2])*ht))
        image_loc = np.hstack((np.array(x_mid), np.array(y_mid))).reshape((-1, 2))
        for i in range(len(objects)):
            xy_range.append(np.array(self.m.imageToVehicle(np.array([x_mid[i],y_mid[i]]))))
        xy_range = np.array(xy_range).reshape((-1,2))

        fps = 1/(time.time() - time_i)
        #############################################
        # Inserting range value into the images
        # convert array into image to draw
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(objects)):
            cv2.putText(image, "x= %.2f" % xy_range[i][0] + " y= %.2f" % xy_range[i][1],
                        (x_mid[i], y_mid[i]), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "fps=%.2f" % fps,
                    (0, ht - 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if len(objects):
            msg = Lanepoints()
            msg.rows = image_loc.shape[0]
            msg.cols = image_loc.shape[1]
            msg.loc = image_loc.reshape((-1))
            #############################################
            self.loc_pub.publish(msg)
        self.object_pub.publish(objArray)

        img=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        ###############################################
        # # world cordinate plot
        # fig = plt.figure()
        # plt.plot(world[:, 1], world[:, 0], world[:, 3], world[:, 2])
        # plt.plot(xy_range[:,1],xy_range[:,0],'ro',markersize=22)
        # plt.xlim((5, -5))
        # fig.canvas.draw()
        # # convert canvas to image
        # plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        #                      sep='')
        # plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #
        # # img is rgb, convert to opencv's default bgr
        # plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        # plt.close(fig)
        # # plt.show(fig)
        # cv2.fillPoly(img, [pts0], (0, 255, 0))
        ##############################################
        image_out = Image()
        try:
            image_out = self.bridge.cv2_to_imgmsg(image,"bgr8")
         #   map_out = self.bridge.cv2_to_imgmsg(plot,"bgr8")
        except CvBridgeError as e:
            print(e)
        image_out.header = data.header
        self.image_pub.publish(image_out)
        # self.map_pub.publish(map_out)
        cv2.imshow("output_image",image)
        cv2.waitKey(3)

    def filter_boxes(self, min_score, boxes, scores, classes, categories):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] in categories and scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    def object_predict(self,object_data, header, image_np,image):
        image_height,image_width,channels = image.shape
        obj=Detection2D()
        obj_hypothesis= ObjectHypothesisWithPose()

        object_id=object_data[0]
        object_score=object_data[1]
        dimensions=object_data[2]

        obj.header=header
        obj_hypothesis.id = object_id
        obj_hypothesis.score = object_score
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((dimensions[2]-dimensions[0])*image_height)
        obj.bbox.size_x = int((dimensions[3]-dimensions[1])*image_width)
        obj.bbox.center.x = int((dimensions[1] + dimensions [3])*image_height/2)
        obj.bbox.center.y = int((dimensions[0] + dimensions[2])*image_width/2)

        return obj


def main(args):
    rospy.init_node('detector_node')
    obj=Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)