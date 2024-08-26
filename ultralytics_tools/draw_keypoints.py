import cv2
class draw_keypoints():
    """
        0 - nose
        1 - left eye
        2 - right eye
        3 - left ear
        4 - right ear
        5 - left shoulder
        6 - right shoulder
        7 - left elbow
        8 - right elbow
        9 - left wrist
        10 - right wrist
        11 - left hip
        12 - right hip
        13 - left knee
        14 - right knee
        15 - left ankle
        16 - right ankle
    """
    def __init__(
        self, 
        scene: list,
        results: list,
        ofset: tuple = (0, 0)
    ) -> None:
        """
            Draw keypoints based on the results of the YOLO pose estimation model.
        """
        if results[0].keypoints.xy.numpy().size > 0:
            keypoints = results[0].keypoints.xy.numpy().tolist()[0]
            for i, point in enumerate(keypoints):
                if results[0].keypoints.conf.numpy()[0][i] > 0.5:
                    keypoints[i][0] += ofset[0]
                    keypoints[i][1] += ofset[1]
        
            self.keypoints = keypoints
            self.scene = scene
            self.ofset = ofset
            
            self.__draw_head()
            self.__draw_body()
            self.__draw_legs()
            self.__draw_points()
        else:
            self.keypoints = None
        
    def __draw_head(self):
        """
            0 - nose
            1 - left eye
            2 - right eye
            3 - left ear
            4 - right ear
        """
        for begin, end in [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]]:
            if not (self.keypoints[begin] == [0,0] or self.keypoints[begin] == [self.ofset[0], self.ofset[1]]) and not (self.keypoints[end] == [0,0] or self.keypoints[end] == [self.ofset[0], self.ofset[1]]):
                cv2.line(self.scene, (int(self.keypoints[begin][0]), int(self.keypoints[begin][1])), (int(self.keypoints[end][0]), int(self.keypoints[end][1])), (0, 255, 0), 2)
    
    def __draw_body(self):
        """
            5 - left shoulder
            6 - right shoulder
            7 - left elbow
            8 - right elbow
            9 - left wrist
            10 - right wrist
            11 - left hip
            12 - right hip
            13 - left knee
            14 - right knee
            15 - left ankle
            16 - right ankle
        """
        for begin, end in [[5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12]]:
            if not (self.keypoints[begin] == [0,0] or self.keypoints[begin] == [self.ofset[0], self.ofset[1]]) and not (self.keypoints[end] == [0,0] or self.keypoints[end] == [self.ofset[0], self.ofset[1]]):
                cv2.line(self.scene, (int(self.keypoints[begin][0]), int(self.keypoints[begin][1])), (int(self.keypoints[end][0]), int(self.keypoints[end][1])), (0, 0, 255), 2)
                
    def __draw_legs(self):
        """
            11 - left hip
            13 - left knee
            15 - left ankle
            12 - right hip
            14 - right knee
            16 - right ankle
        """
        for begin, end in [[11, 13], [13, 15], [12, 14], [14, 16]]:
            if not (self.keypoints[begin] == [0,0] or self.keypoints[begin] == [self.ofset[0], self.ofset[1]]) and not (self.keypoints[end] == [0,0] or self.keypoints[end] == [self.ofset[0], self.ofset[1]]):
                cv2.line(self.scene, (int(self.keypoints[begin][0]), int(self.keypoints[begin][1])), (int(self.keypoints[end][0]), int(self.keypoints[end][1])), (255, 0, 255), 2)
                
    def __draw_points(self):
        for _, point in enumerate(self.keypoints):
            if not (point == [0,0] or point == [self.ofset[0], self.ofset[1]]):
                cv2.circle(self.scene, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        
from typing import List, Tuple
import numpy as np

class draw_pose_keypoints():
    """
        Openpose keypoints based drawing:
        0: nose, 1: neck, 2: right shoulder, 3: right elbow, 
        4: right wrist, 5: left shoulder, 6: left elbow, 
        7: left wrist, 8: right hip, 9: right knee, 
        10: right ankle, 11: left hip, 12: left knee, 
        13: left ankle, 14: right eye, 15: left eye, 
        16: right ear, 17: left ear
    """
    def __init__(
        self,
        image: np.ndarray,
        keypoints: List,
        ofset: Tuple = (0,0),
        draw: bool = True
    ):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        
        self.keypoints = keypoints[0].body[0]
        
        self.ofsetX = ofset[0]
        self.ofsetY = ofset[1]
        
        self._draw_head()
        self._draw_torso()
        self._draw_arms()
        self._draw_legs()
    
    
    def _draw_head(self) -> None:
        """
            Head points: 0: nose, 1: neck, 14: right eye, 15: left eye, 16: right ear, 17: left ear
        """ 
        for start, end in [(0, 1), (1, 14), (1, 15), (14, 16), (15, 17)]:
            try:
                start = [int(self.keypoints[start][0] * self.width + self.ofsetX), int(self.keypoints[start][1] * self.height + self.ofsetY)]
                end = [int(self.keypoints[end][0] * self.width + self.ofsetX), int(self.keypoints[end][1] * self.height + self.ofsetY)]
                
                cv2.line(self.image, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), 2)
            except TypeError:
                continue
            
    
    def _draw_torso(self) -> None:
        """
            Torso points: 1: neck, 8: right hip, 11: left hip, 2: right shoulder, 5: left shoulder
        """
        for start, end in [(2, 8), (5, 11), (1, 2), (1, 5), (8, 11)]:
            try:
                start = [int(self.keypoints[start][0] * self.width + self.ofsetX), int(self.keypoints[start][1] * self.height + self.ofsetY)]
                end = [int(self.keypoints[end][0] * self.width + self.ofsetX), int(self.keypoints[end][1] * self.height + self.ofsetY)]
                
                cv2.line(self.image, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), 2)
            except TypeError:
                continue
                
    def _draw_arms(self) -> None:
        """
            Arms points: 2: right shoulder, 3: right elbow, 4: right wrist, 5: left shoulder, 6: left elbow, 7: left wrist
        """
        for start, end in [(2, 3), (3, 4), (5, 6), (6, 7)]:
            try:
                start = [int(self.keypoints[start][0] * self.width + self.ofsetX), int(self.keypoints[start][1] * self.height + self.ofsetY)]
                end = [int(self.keypoints[end][0] * self.width + self.ofsetX), int(self.keypoints[end][1] * self.height + self.ofsetY)]
                
                cv2.line(self.image, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), 2)
            except TypeError:
                continue
                
    def _draw_legs(self) -> None:
        """
            Legs points: 8: right hip, 9: right knee, 10: right ankle, 11: left hip, 12: left knee, 13: left ankle
        """
        for start, end in [(8, 9), (9, 10), (11, 12), (12, 13)]:
            try: 
                start = [int(self.keypoints[start][0] * self.width + self.ofsetX), int(self.keypoints[start][1] * self.height + self.ofsetY)]
                end = [int(self.keypoints[end][0] * self.width + self.ofsetX), int(self.keypoints[end][1] * self.height + self.ofsetY)]
                
                cv2.line(self.image, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), 2)
            except TypeError:
                continue
        