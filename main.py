"""
    TESTE COMPLETO DE TODAS AS FERRAMENTAS EM VIDEO!
"""

import supervision_tools as svt
import ultralytics_tools as ult
import cv2
from transformers import pipeline
import mivolo.predictor as mivolo
from PIL import Image
import torch
import numpy as np
from types import SimpleNamespace
import pandas as pd
from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def filter_emotion_tolerance(emotions:dict) -> str:
    emotions = pd.DataFrame(emotions)
    if (emotions[emotions['label'] == 'neutral']['score'].iloc[0] - emotions.iloc[0]['score']) > 0.3:
        label = 'neutral'
    label = emotions.iloc[0]['label']
    return label

def main(url):
    # emotions model - huggingface
    pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    pose_model = ult.run_yolo_model('yolov8n', task='pose', format='onnx', device=device)
    mivolo_weights = 'src/models/yolo_weights.pt'
    mivolo_checkpoint = 'src/models/mivolo_checkpoints.tar'

    # face-person - mivolo model
    config = {
        'detector_weights': mivolo_weights,
        'checkpoint': mivolo_checkpoint,
        'device': device,
        'with_persons': True,
        'disable_faces': False,
        'draw': True,
    }
    config = SimpleNamespace(**config)
    face_person_model = mivolo.Predictor(config, verbose=False)

    # instantiate the objects
    annotation = svt.annotations()
    tracker = svt.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=60000,
            minimum_matching_threshold=0.9,
        )
    tracker.reset()
    timers = svt.FPSBasedTimer()

    video_url = url
    video_info = svt.video_info(video_url)
    video_writer = svt.save_video(fps=video_info.fps, height=video_info.height, width=video_info.width)
    log = svt.log()

    frame_count = 0
    # video frames generating
    for frame in svt.generate_video_frames(video_url):
        
        if frame is not None:
            frame_count+=1
            annotated_frame = frame.copy()
            
            start = datetime.now()
            # face_person_model detects and person and faces in the frame
                # also, it estimates gender and ages
            face_person_results, output_image = face_person_model.recognize(annotated_frame)
            face_to_person_map = face_person_results.face_to_person_map
            face_to_person_map_keys = list(face_to_person_map.keys())
            face_person_results.associate_faces_with_persons()
            person_bbox_idx = face_person_results.get_bboxes_inds("person")
            
            genders = face_person_results.genders
            ages = face_person_results.ages
            
            yolo_results = face_person_results.yolo_results
            detections = svt.get_ultralytics_detections(yolo_results[person_bbox_idx], tracker=tracker)
            times = timers.tick(detections)
             
            end = datetime.now()
            print(f'Face - person inference time: {(end-start).total_seconds():.2f}')
            
            # enumerate all detections bounding boxes (detections are filter by person class)
            for idx, (person_bbox, tracker_id, time) in enumerate(zip(detections.xyxy, detections.tracker_id, times)):
                
                person_idx = person_bbox_idx[idx]
                
                # openpose process
                # person_croped = face_person_results.crop_object(frame,person_idx)
                # if person_croped is not None:
                #     person_croped_pose_results = pose_model(person_croped)
                #     ult.draw_keypoints(annotated_frame, person_croped_pose_results, ofset=(int(person_bbox[0]), int(person_bbox[1])))
                # else:
                #     person_croped_pose_results = None
                person_croped_pose_results = None
                
                # try is used because when it gets IndexError, means that is a person without face
                start = datetime.now()
                try:
                    face_idx = int([k for k in face_to_person_map_keys if face_to_person_map[k] == person_bbox_idx[idx]][0])
                    face_bbox = face_person_results.get_bbox_by_ind(face_idx)
                    
                    # emotion process
                    face_croped = face_person_results.crop_object(frame,face_idx)
                    face_croped = cv2.cvtColor(face_croped, cv2.COLOR_BGR2RGB)
                    PIL_image_croped = Image.fromarray(face_croped)
                    emotion = pipe(PIL_image_croped)
                    emotion_label = filter_emotion_tolerance(emotion)
                    
                    cv2.rectangle(annotated_frame, (int(face_bbox[0]), int(face_bbox[1])), (int(face_bbox[2]), int(face_bbox[3])), (0, 255, 0), 2)
                
                except IndexError:
                    emotion_label = None
                    pass
                person_bbox = face_person_results.get_bbox_by_ind(person_idx)
                gender = genders[person_idx]
                age = ages[person_idx]
                
                end = datetime.now()
                print(f'Emotion inference time: {(end-start).total_seconds():.2f}')
                
                
                label = f'#{tracker_id} {gender} {int(age)}yo {emotion_label} {time:.2f}s'
                cv2.rectangle(annotated_frame, (int(person_bbox[0]), int(person_bbox[1])), (int(person_bbox[2]), int(person_bbox[3]),), (0, 0, 255), 2)
                cv2.putText(annotated_frame, label, (int(person_bbox[0]), int(person_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
                
                # logging process
                log.custom_logging(
                    {
                        f'Pessoa {tracker_id}': {
                            'gender': gender,
                            'age': age,
                            'emotion': emotion_label,
                            'time_in_frame': f'{time:2}',
                            'last_time_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                )
                log.populate_csv(tracker_id, frame_count, gender, age, emotion_label, person_bbox, person_croped_pose_results)
                log.save_log()
                video_writer(annotated_frame)
        
        
        cv2.imshow('frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    url = '/Users/thiagomachado/Documents/Programação/Python/src/Videos/retail.mp4'
    main(url)