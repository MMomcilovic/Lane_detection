#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

def middle_lane_point(lines):
    y_const = 350
    x_const = 320
    x_right_list = []
    x_left_list = []

    x_right = 640
    x_left = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_check = (x1+x2)/2
        y_check = (y1+y2)/2
        check_x = x_check - x_const
        y_mean = max([y1,y2])
        if y_mean>250 and y_mean<400 :
            if check_x>0 and x_check<x_right :
                _x = int((x_check + x_right)/2)
                x_right_list.append(_x)
                x_right = np.average(x_right_list)

            elif check_x<0 and x_check>x_left:
                _x = int((x_check + x_left)/2)
                x_left_list.append(_x)
                x_left = np.average(x_left_list)
            # error_y = abs(y1-y_check)
    if len(x_left_list) <= 1:
        x_left = -100
    if len(x_right_list) <= 1:
        x_right = 600
    x= int((x_right+x_left)/2)
    return (x, y_const)

def lane_tracking(edges):
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                2, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=60, # Min number of votes for valid line
                minLineLength=20, # Min allowed length of line
                maxLineGap=4# Max allowed gap between line for joining them
                )
    # print(lines)
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        lines_list.append([x1,y1,x2,y2])
    (x,y) = middle_lane_point(lines)
    return lines_list, (x,y)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intersect(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2  = line;
        y_mean = max([y1,y2])
        if y_mean>250:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if (len(left_fit) == 0 or len(right_fit) == 0):
        return -1
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def region_of_interest(edges):
    height = edges.shape[0]
    width = edges.shape[1]
    mask = np.zeros_like(edges)
    triangle = np.array([[(0,height),(320,50), (640, height),]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def process_image(image):
    src_img = cv2.resize(image,(640,480))
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ksize = (5, 5)
    blur_img = cv2.blur(gray_img, ksize, cv2.BORDER_DEFAULT) 
    edges = cv2.Canny(blur_img,190,230,None, 3)
    return edges

class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.pre_x = 0
        rospy.init_node('LineDetection', anonymous=True)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        self.coord_pub = rospy.Publisher("/control/coord", String, queue_size=10)
        self.rate = rospy.Rate(20)
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        edges = process_image(self.cv_image)
        # cv2.imshow("Image with edges",edges)
        roi = region_of_interest(edges)
        # cv2.imshow("Image with roi",roi)
        lines, (x,y) = lane_tracking(roi)
        avg_lines = average_slope_intersect(self.cv_image, lines)
        if (avg_lines != -1):
            # cv2.imshow("Image with avg",avg_lines)
            line_img = display_lines(self.cv_image, avg_lines)
            # cv2.imshow("Image with line",line_img)
            comboImg = addWeighted(self.cv_image, line_img)
            cv2.imshow("Image with comb",comboImg)
            image = self.cv_image
            if x > 300 and x < 340:
                x = 320
            if abs(x-self.pre_x) < 20:
                x = self.pre_x
            self.pre_x = x
            for line in lines:
                x1,y1,x2,y2=line
                y_mean = max([y1,y2])
                if y_mean>250:
                    cv2.line(self.cv_image,(x1,y1),(x2,y2),(0,255,0),2)
        image = cv2.circle(self.cv_image, (x,y), radius=1, color=(0, 0, 255), thickness=4)
        cv2.imshow("Image with lines",image)
        self.coord_pub.publish(str(x))

        key = cv2.waitKey(1)
    
            
if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass