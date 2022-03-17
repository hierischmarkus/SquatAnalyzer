from keyword import kwlist
from operator import index
import re
from turtle import color, pos
from typing import Mapping
from webbrowser import get
from cv2 import threshold, transform
import numpy as np
import math
import VectorsPY
from queue import Queue
import cv2
import keyboard
import csv


class SquatAnalyzer:

    def __init__(self, joint_names, joint_connections):
        with open('config.txt') as f:
            text = f.read()
        bone_data = re.search('BoneNames:\n(.*)\n', text)
        self.bone_convention = bone_data.group(1).split(';')

        self.joint_names = joint_names
        self.joint_connections = joint_connections
        self._pose_data = [VectorsPY.Vector3(
            0, 0, 0)] * joint_names.size
        self.max_distance_standing = 0.0

        self.transform = VectorsPY.Vector3(float(re.search('transform_x: (.*)\n', text).group(1)), float(
            re.search('transform_y: (.*)\n', text).group(1)), float(re.search('transform_z: (.*)\n', text).group(1)))

        # if is in start positon
        self.start_position_angle = float(
            re.search('start_position_angle: (.*)\n', text).group(1))
        self.is_startposition = False

        # correct depth threshold
        self.check_depth = re.search('Depth:\n(.*)\n', text).group(1) == 'True'
        self.depth_threshold = float(
            re.search('depth_threshold: (.*)\n', text).group(1))
        self._correct_depth = True

        # knee in right position
        self.check_knee_dir = re.search(
            'KneeDir:\n(.*)\n', text).group(1) == 'True'
        self.deviation_knee_angle_low = float(
            re.search('deviation_knee_angle_low: (.*)\n', text).group(1))
        self.deviation_knee_angle_high = float(
            re.search('deviation_knee_angle_high: (.*)\n', text).group(1))
        self.reference_point_knee_direction_left = VectorsPY.Vector3(0, 0, 0)
        self.reference_point_knee_direction_right = VectorsPY.Vector3(0, 0, 0)
        self._right_knee_in_right_position = True
        self._left_knee_in_right_position = True

        # rounded back angle parameters
        self.check_back = re.search(
            'ArchedBack:\n(.*)\n', text).group(1) == 'True'
        self.min_roundedback_angle_low = float(
            re.search('min_roundedback_angle_low: (.*)\n', text).group(1))
        self.max_roundedback_angle_low = float(
            re.search('max_roundedback_angle_low: (.*)\n', text).group(1))
        self.min_roundedback_angle_high = float(
            re.search('min_roundedback_angle_high: (.*)\n', text).group(1))
        self.max_roundedback_angle_high = float(
            re.search('max_roundedback_angle_high: (.*)\n', text).group(1))
        self._good_back_posture = True

        # hips shoot up
        self.check_hips = re.search(
            'HipsShootUp:\n(.*)\n', text).group(1) == 'True'
        self.last_pelvis_values = []
        self.depth_angle = 0.0
        self.depth_angle_threshold = float(
            re.search('depth_angle_threshold: (.*)\n', text).group(1))
        self._no_pelvis_shoot_up = True

        # line of bar
        self.check_upline = re.search(
            'ComingUpLine:\n(.*)\n', text).group(1) == 'True'
        self._line_of_bar = True
        self.reference_point = 0.0
        self.reference_angle_line = 0.0
        self.line_bar_threshold_high = float(
            re.search('line_bar_threshold_high: (.*)', text).group(1))
        self.line_bar_threshold_low = float(
            re.search('line_bar_threshold_low: (.*)', text).group(1))

        # recording the detection
        self.do_record = re.search(
            'RecordData:\n(.*)\n', text).group(1) == 'True'
        file = open('detection_analysis', 'w')
        self.file_writer = csv.writer(file)
        self.file_writer.writerow('huan')
        self.in_squat = False
        self.squat_count = 0
        self.depth_in_squat_reached = False
        self.record_data = ''

    # pose and parameter setup
    def set_pose_data(self, pose_data):
        transformed_pose_data = [VectorsPY.Vector3(
            0, 0, 0)] * self.joint_names.size
        if(np.array(self._pose_data).size == self.joint_names.size):
            new_zero = pose_data[self.get_bone_index('pelvis')][0]
            new_zero[1] = (pose_data[self.get_bone_index('left_ankle')][0]
                           [1]+pose_data[self.get_bone_index('right_ankle')][0][1])/2
            for index, pose in enumerate(pose_data):
                transformed_pose_data[index] = VectorsPY.Vector3(
                    (pose[0] - new_zero[0])/self.transform.x, (pose[1] - new_zero[1])/self.transform.y, (pose[2] - new_zero[2])/self.transform.z)
            left_hip = self.get_bone('left_hip')
            left_knee = self.get_bone('left_knee')
            left_ankle = self.get_bone('left_ankle')
            if(left_hip.x != 0.0):
                self.is_startposition = (self.get_angle_between_3_points(
                    left_ankle, left_knee, left_hip) > self.start_position_angle)
                if(self.is_startposition):
                    self.max_distance_standing = left_hip.y - left_knee.y + self.depth_threshold
            pelvis = self.get_bone('pelvis')
            if(len(self.last_pelvis_values) > 10):
                self.last_pelvis_values.pop()
            self.last_pelvis_values.insert(0, pelvis.y)
        self._pose_data = transformed_pose_data

    # helper functions
    def mapping(self, value, min_start, max_start, min_end, max_end):
        return min_end + (max_end - min_end) * (value - min_start) / (max_start - min_start)

    def mapping_cut(self, value, min_start, max_start, min_end, max_end):
        if(min_start < max_start):
            if(value < min_start):
                return min_end
            if(value > max_start):
                return max_end
        if(min_start > max_start):
            if(value < max_start):
                return max_end
            if(value > min_start):
                return min_end
        return self.mapping(value, min_start, max_start, min_end, max_end)

    def get_bone(self, name):
        index_bone = self.get_bone_index(name)[0]
        return self._pose_data[index_bone]

    def get_bone_index(self, name):
        bone = [s for s in self.bone_convention if name in s][0]
        index_bone = np.where(self.joint_names ==
                              re.search('"(.*)"', bone).group(1))[0]
        return index_bone

    def current_depth(self):
        left_knee = self.get_bone('left_knee')
        left_hip = self.get_bone('left_hip')
        return left_hip.y - left_knee.y + self.depth_threshold

    def get_vector_length(self, vector):
        return vector.magnitude()

    def get_angle_between_3_points(self, point1, middle_point, point2):
        vector1 = VectorsPY.Vector(middle_point - point1)
        vector2 = VectorsPY.Vector(middle_point - point2)

        angle = math.acos((vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z) / (
            self.get_vector_length(vector1) * self.get_vector_length(vector2)))*180/math.pi
        return angle

    # mistake detections
    def correct_depth(self):
        if(self.current_depth() < 0):
            return True
        else:
            return False

    def knee_in_right_position(self, toe, knee, ankle):

        threshold = self.mapping_cut(self.current_depth(), 0.0, self.max_distance_standing-self.max_distance_standing/10,
                                     self.deviation_knee_angle_low, self.deviation_knee_angle_high)

        pelvis_no_y = VectorsPY.Vector3(self.get_bone(
            'pelvis').x, 0, self.get_bone('pelvis').z)
        ankle_no_y = VectorsPY.Vector3(ankle.x, 0, ankle.z)
        knee_no_y = VectorsPY.Vector3(knee.x, 0, ankle.z)
        toe_no_y = VectorsPY.Vector3(toe.x, 0, toe.z)

        angle1 = self.get_angle_between_3_points(
            pelvis_no_y, ankle_no_y, knee_no_y)
        angle2 = self.get_angle_between_3_points(
            pelvis_no_y, ankle_no_y, toe_no_y)

        # print(angle1-angle2)

        if(self.is_startposition or self.current_depth() > self.max_distance_standing-self.max_distance_standing/2):
            return True
        else:
            if(angle1-angle2 < threshold):
                return False
            else:
                return True

    def left_knee_in_right_position(self):
        left_knee = self.get_bone('left_knee')
        left_toe = self.get_bone('left_toe')
        left_ankle = self.get_bone('left_ankle')
        return self.knee_in_right_position(left_toe, left_knee, left_ankle)

    def right_knee_in_right_position(self):
        right_knee = self.get_bone('right_knee')
        right_toe = self.get_bone('right_toe')
        right_ankle = self.get_bone('right_ankle')
        return self.knee_in_right_position(right_toe, right_knee, right_ankle)

    def good_back_posture(self):
        pelvis = self.get_bone('pelvis')
        spine_chest = self.get_bone('spine_chest')
        neck = self.get_bone('neck')
        reference_angle_min = self.mapping_cut(self.current_depth(), 0.0, self.max_distance_standing,
                                               self.min_roundedback_angle_low, self.min_roundedback_angle_high)
        reference_angle_max = self.mapping_cut(self.current_depth(), 0.0, self.max_distance_standing,
                                               self.max_roundedback_angle_low, self.max_roundedback_angle_high)
        angle = self.get_angle_between_3_points(pelvis, spine_chest, neck)
        # print(angle)
        if(angle > reference_angle_min and angle < reference_angle_max):
            return True
        else:
            return False

    def no_pelvis_shoot_up(self):
        pelvis = self.get_bone('pelvis')
        spine_chest = self.get_bone('spine_chest')
        right_ankle = self.get_bone('right_ankle')
        # left ankle = (0,0,0)
        middle_point_bottom = VectorsPY.Vector3(right_ankle.x/2, 0.0, 0.0)
        angle = self.get_angle_between_3_points(
            middle_point_bottom, pelvis, spine_chest)
        if(self.correct_depth()):
            self.depth_angle = angle

        #print(angle - self.depth_angle)
        if(angle - self.depth_angle < self.depth_angle_threshold and pelvis.y >= sum(self.last_pelvis_values)/len(self.last_pelvis_values)):
            return False
        else:
            return True

    def line_of_bar(self):
        left_shoulder = self.get_bone('left_shoulder')
        left_ankle = self.get_bone('left_ankle')
        left_toe = self.get_bone('left_toe')

        threshold = self.mapping_cut(self.current_depth(), 0.0, self.max_distance_standing/5,
                                     self.line_bar_threshold_low, self.line_bar_threshold_high)

        p1 = VectorsPY.Vector3(
            left_shoulder.x, left_shoulder.y, left_shoulder.z)
        p2 = VectorsPY.Vector3(left_ankle.x, left_ankle.y, left_ankle.z)
        p3 = VectorsPY.Vector3(left_ankle.x, left_ankle.y, left_toe.z)
        current_angle = self.get_angle_between_3_points(p1, p2, p3)

        if(self.is_startposition):
            self.reference_angle_line = self.get_angle_between_3_points(
                p1, p2, p3)

        if(current_angle - self.reference_angle_line < threshold):
            return False
        else:
            return True

    def make_detections(self):
        if(keyboard.is_pressed('r')):
            self.do_record = True
        if(keyboard.is_pressed('p')):
            self.do_record = False

        if(self.check_depth):
            self._correct_depth = self.correct_depth()
        if(self.check_knee_dir):
            self._left_knee_in_right_position = self.left_knee_in_right_position()
            self._right_knee_in_right_position = self.right_knee_in_right_position()
        if(self.check_back):
            self._good_back_posture = self.good_back_posture()
        if(self.check_hips):
            self._no_pelvis_shoot_up = self.no_pelvis_shoot_up()
        if(self.check_upline):
            self._line_of_bar = self.line_of_bar()

        if(self.do_record):
            self.recording()

    def recording(self):
        if(self.is_startposition):
            if(self.depth_in_squat_reached == True):
                print("back again")
                self.in_squat = False
                self.depth_in_squat_reached = False
            if(self.in_squat == False):
                self.in_squat = True
                self.squat_count += 1
                row = 'squat: ' + str(self.squat_count) + '\n'
                self.record_data += row
        if(self.in_squat == True and self.correct_depth()):
            print("depth reached")
            self.depth_in_squat_reached = True

        if(self.in_squat):
            if(self._left_knee_in_right_position == False):
                self.record_data += "Left Knee wrong in depth: " + \
                    str(self.current_depth()) + '\n'
            if(self._right_knee_in_right_position == False):
                self.record_data += "Right Knee wrong in depth: " + \
                    str(self.current_depth()) + '\n'
            if(self._good_back_posture == False):
                self.record_data += "Bend Back in depth: " + \
                    str(self.current_depth()) + '\n'
            if(self._no_pelvis_shoot_up == False):
                self.record_data += "Hips shoot up in depth: " + \
                    str(self.current_depth()) + '\n'
            if(self._line_of_bar == False):
                self.record_data += "Not Line of bar in depth: " + \
                    str(self.current_depth()) + '\n'
            print(self.record_data)

    def visualization(self, frame, poses2d, show_infotext):

        # for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        font_scale_numbers = 2
        thickness_numbers = 3

        for joint_edge in self.joint_connections:
            x1 = int(poses2d[joint_edge[0]][0])
            x2 = int(poses2d[joint_edge[1]][0])
            y1 = int(poses2d[joint_edge[0]][1])
            y2 = int(poses2d[joint_edge[1]][1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), radius=3,
                       color=(0, 255, 0), thickness=-1)

        if(self._correct_depth):
            pos = poses2d[self.get_bone_index('pelvis')][0]
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=20,
                       color=(0, 255, 0), thickness=-1)
            frame = cv2.putText(frame, '1', (int(pos[0])-20, int(pos[1])+10), font,
                                font_scale_numbers, (255, 255, 255), thickness_numbers, cv2.LINE_AA)

        if(self._good_back_posture == False):
            pelvis_pos = poses2d[self.get_bone_index('pelvis')][0]
            navel_pos = poses2d[self.get_bone_index('spine_naval')][0]
            chest_pos = poses2d[self.get_bone_index('spine_chest')][0]
            neck_pos = poses2d[self.get_bone_index('neck')][0]
            cv2.line(frame, (int(pelvis_pos[0]), int(pelvis_pos[1])), (int(
                navel_pos[0]), int(navel_pos[1])), (0, 0, 255), 5)
            cv2.line(frame, (int(navel_pos[0]), int(navel_pos[1])), (int(
                chest_pos[0]), int(chest_pos[1])), (0, 0, 255), 5)
            cv2.line(frame, (int(chest_pos[0]), int(chest_pos[1])), (int(
                neck_pos[0]), int(neck_pos[1])), (0, 0, 255), 5)
            frame = cv2.putText(frame, '3', (int(chest_pos[0])-50, int(chest_pos[1])+10), font,
                                font_scale_numbers, (255, 255, 255), thickness_numbers, cv2.LINE_AA)

        if(self._no_pelvis_shoot_up == False):
            pelvis_pos = poses2d[self.get_bone_index('pelvis')][0]
            cv2.arrowedLine(frame, (int(pelvis_pos[0]), int(pelvis_pos[1])), (int(
                pelvis_pos[0]+30), int(pelvis_pos[1])-40), (0, 0, 255), 10)
            frame = cv2.putText(frame, '4', (int(pelvis_pos[0]+30), int(pelvis_pos[1])-40), font,
                                font_scale_numbers, (255, 255, 255), thickness_numbers, cv2.LINE_AA)

        if(self._left_knee_in_right_position == False):
            left_knee_pos = poses2d[self.get_bone_index('left_knee')][0]
            left_toe_pos = poses2d[self.get_bone_index('left_toe')][0]
            cv2.line(frame, (int(left_knee_pos[0]), int(left_knee_pos[1])), (int(
                left_toe_pos[0]), int(left_toe_pos[1])), (0, 0, 255), 5)
            cv2.circle(frame, (int(left_knee_pos[0]), int(left_knee_pos[1])), radius=8,
                       color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (int(left_toe_pos[0]), int(left_toe_pos[1])), radius=8,
                       color=(0, 0, 255), thickness=-1)
            frame = cv2.putText(frame, '2', (int(left_knee_pos[0]+30), int(left_knee_pos[1])), font,
                                font_scale_numbers, (255, 255, 255), thickness_numbers, cv2.LINE_AA)

        if(self._right_knee_in_right_position == False):
            right_knee_pos = poses2d[self.get_bone_index('right_knee')][0]
            right_toe_pos = poses2d[self.get_bone_index('right_toe')][0]
            cv2.line(frame, (int(right_knee_pos[0]), int(right_knee_pos[1])), (int(
                right_toe_pos[0]), int(right_toe_pos[1])), (0, 0, 255), 5)
            cv2.circle(frame, (int(right_knee_pos[0]), int(right_knee_pos[1])), radius=8,
                       color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (int(right_toe_pos[0]), int(right_toe_pos[1])), radius=8,
                       color=(0, 0, 255), thickness=-1)
            frame = cv2.putText(frame, '2', (int(right_knee_pos[0]-30), int(right_knee_pos[1])), font,
                                font_scale_numbers, (255, 255, 255), thickness_numbers, cv2.LINE_AA)

        if(self._line_of_bar == False):
            left_shoulder_pos = poses2d[self.get_bone_index(
                'left_shoulder')][0]
            right_shoulder_pos = poses2d[self.get_bone_index(
                'right_shoulder')][0]
            frame = cv2.putText(frame, '5', (int(left_shoulder_pos[0])+30, int(left_shoulder_pos[1])), font,
                                font_scale_numbers, font_color, thickness_numbers, cv2.LINE_AA)
            cv2.line(frame, (int(left_shoulder_pos[0]+50), int(left_shoulder_pos[1])), (int(
                left_shoulder_pos[0]+50), int(left_shoulder_pos[1])+100), (0, 0, 255), 5)
            cv2.line(frame, (int(right_shoulder_pos[0]-50), int(right_shoulder_pos[1])), (int(
                right_shoulder_pos[0]-50), int(right_shoulder_pos[1])+100), (0, 0, 255), 5)

        if(show_infotext):
            frame = cv2.rectangle(frame, (0, 0), (400, 115), (0, 0, 0), -1)
            font_scale = 0.5
            thickness = 1
            frame = cv2.putText(frame, '1.) Correct Depth reached', (20, 20), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, '2.) Knees should be same direction as toes', (20, 40), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, '3.) Bad back posture', (20, 60), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, '4.) Hips are shooting up from the bottom', (20, 80), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, '5.) Bar is not moving in a straight line', (20, 100), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)

        return frame
