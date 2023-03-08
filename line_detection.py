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
import matplotlib.pyplot as plt

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

# def perspective_warp(img,
#                      dst_size=(640,250),
#                      src=np.float32([(0,1),(0.95,1),(0,0),(1,0)]),
#                      dst=np.float32([(0.5,1), (0.6, 1), (0,0), (1,0)])):
#     img_size = np.float32([(img.shape[1],img.shape[0])])
#     src = src* img_size
#     # For destination points, I'm arbitrarily choosing some points to be
#     # a nice fit for displaying our warped result
#     # again, not exact, but close enough for our purposes
#     dst = dst * np.float32(dst_size)
#     # Given src and dst points, calculate the perspective transform matrix
#     M = cv2.getPerspectiveTransform(src, dst)
#     # Warp the image using OpenCV warpPerspective()
#     warped = cv2.warpPerspective(img, M, dst_size)
#     return warped

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary

def perspective_warp(img, 
                     dst_size=(640,170),
                     src=np.float32([(0,1),(1,1),(0,0),(1,0)]),
                     dst=np.float32([(0.3,1), (0.7, 1), (0,0), (1,0)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, 
                     dst_size=(640,170),
                     dst=np.float32([(0,1),(0.95,1),(0,0),(1,0)]),
                     src=np.float32([(0.45,1), (0.6, 1), (0,0), (1,0)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=50, minpix = 30, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img[70:170, 0:640])
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int32(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    if len(lefty) != 0 and len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = [0,0,0]
    if len(righty) != 0 and len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = [0,0,0]
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 50]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 50, 255]
    
    return out_img, (ltx, rtx), (left_fit, right_fit), ploty

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.pre_x = 320
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
        # cv2.imshow("img", perspective_warp(self.cv_image[230:480, 0:640]))

        img = self.cv_image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pipe = pipeline(img)
        # cv2.imshow('first', img)
        # cv2.imshow('pipeline', pipe)
        perspective = perspective_warp(pipe[330:470, 0:640])
        # cv2.imshow("perspektiva", perspective)

        gray = cv2.cvtColor(perspective, cv2.COLOR_BGR2GRAY)

        out_img, curves, lanes, ploty = sliding_window(gray)
        # cv2.imshow('sliding', out_img)
        # img_ = draw_lanes(img[300:470, 0:640], curves[0], curves[1])
        meanx = int(lanes[0][2] + abs(lanes[0][2] - lanes[1][2]) / 2)
        if abs(meanx - self.pre_x) < 20 and abs(meanx - self.pre_x) > 250:
            meanx = self.pre_x
        self.pre_x = meanx
        image = cv2.circle(self.cv_image, (meanx,300), radius=1, color=(0, 0, 255), thickness=4)
        # cv2.imshow('slicica',img_)
        cv2.imshow('tackica', image)

        # edges = process_image(self.cv_image)
        # # cv2.imshow("Image with edges",edges)
        # roi = region_of_interest(edges)
        # # cv2.imshow("Image with roi",roi)
        # lines, (x,y) = lane_tracking(roi)
        # avg_lines = average_slope_intersect(self.cv_image, lines)
        # if (avg_lines != -1):
        #     # cv2.imshow("Image with avg",avg_lines)
        #     line_img = display_lines(self.cv_image, avg_lines)
        #     # cv2.imshow("Image with line",line_img)
        #     comboImg = addWeighted(self.cv_image, line_img)
        #     cv2.imshow("Image with comb",comboImg)
        #     image = self.cv_image
        #     if x > 300 and x < 340:
        #         x = 320
        #     if abs(x-self.pre_x) < 20:
        #         x = self.pre_x
        #     self.pre_x = x
        #     for line in lines:
        #         x1,y1,x2,y2=line
        #         y_mean = max([y1,y2])
        #         if y_mean>250:
        #             cv2.line(self.cv_image,(x1,y1),(x2,y2),(0,255,0),2)
        # image = cv2.circle(self.cv_image, (x,y), radius=1, color=(0, 0, 255), thickness=4)
        # cv2.imshow("Image with lines",image)
        self.coord_pub.publish(str(meanx))

        key = cv2.waitKey(1)
    
            
if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass
