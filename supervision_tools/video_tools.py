import cv2
import os
from supervision import get_video_frames_generator, VideoInfo

def generate_video_frames(video_path):
    frame_generator = get_video_frames_generator(video_path)
    frame = next(frame_generator)
    for frame in frame_generator:
        yield frame
        
def video_info(video_source):
    return VideoInfo.from_video_path(video_source)
    
class save_video():
    def __init__(
        self,
        fps,
        height,
        width
    ):
        if not os.path.exists('output'):
            os.makedirs('output')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
    def update_frame(
        self,
        frame
    ):
        self.out.write(frame)
        
    def __call__(
        self,
        frame
    ):
        self.update_frame(frame)