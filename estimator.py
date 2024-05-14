import sys 
import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, Signal 
from PySide6.QtGui import QImage 
from body import BodyState
from body.const import IMAGE_HEIGHT, IMAGE_WIDTH

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


BG_COLOR = (192, 192, 192) # gray

class Cv2Thread(QThread):
    update_frame = Signal(QImage) # connecting with a slot that takes a QImage as an argument
    update_state = Signal(dict) 

    def __init__(
            self, parent = None, mp_config = None, body_config = None, events_config = None
    ):
        QThread.__init__(self, parent)
        self.status = True 
        self.cap = True 
        self.body = BodyState(body_config, events_config)
        self.mp_config = mp_config 
    
    def run(self):
        checkpoint_path = 'models/checkpoint_iter_5000_new.pth'
        height_size = 256
        # video = 0
        video = 'data/dancing.mp4'
        images = 'data/hor.jpg'
        cpu = True 
        track = 1 
        smooth = 1

        if video == '' and images == '':
            raise ValueError('Either --video or --image has to be provided')

        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)

        if video == '':
            track = 0

        cap = cv2.VideoCapture(0)
        while cap.isOpened() and self.status:
            
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading video, use 'break' instead of 'continue'.
                continue 

            # Recolor image to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame.flags.writeable = False 
            
            # Make detection
            pose_coordinations, drawn_frame = run_demo(net, frame, height_size, cpu, track, smooth)

            # Recolor back to BGR
            frame.flags.writeable = True 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # The calculate() function calculates the body landmarks of a given image
            self.body.calculate(drawn_frame, pose_coordinations)

            # Reading the image in RGB to display it 
            image = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)

            # Creating and scalling
            h, w, ch = image.shape 
            image = QImage(image.data, w, h, ch*w, QImage.Format_RGB888)
            image = image.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, Qt.KeepAspectRatio)

            # Emit signal
            self.update_frame.emit(image)
            self.update_state.emit(dict(body = self.body))

            delay = 1
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                break
            elif key == 112:  # 'p'
                if delay == 1:
                    delay = 0
                else:
                    delay = 1

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


# def run_demo(net, image_provider = None, height_size, cpu, track, smooth):
def run_demo(net, img, height_size, cpu, track, smooth):

    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    
    # for img in image_provider:
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    pose_coords = []
    for pose in current_poses:
        pose_coords.append(pose.keypoints)
    if not pose_coords:
        print("Warning: No human pose detected in the frame")
        return np.empty((0,)), img
    if track:
        track_poses(previous_poses, current_poses, smooth=smooth)
        previous_poses = current_poses
    for pose in current_poses:
        # Draw pose for each person
        pose.draw(img)
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        if track:
            cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
    pose_coords = np.array(pose_coords)
    # n = pose_coords.shape[0]
    random_scores = np.random.uniform(0.6, 1, size = (18, 1))
    pose_coords = np.concatenate((pose_coords[0], random_scores), axis = 1)
    return pose_coords, img

# if __name__ == '__main__':
#     checkpoint_path = 'models/checkpoint_iter_5000_new.pth'
#     height_size = 256
#     # video = 0
#     video = 'data/dancing.mp4'
#     images = 'data/hor.jpg'
#     cpu = True 
#     track = 1 
#     smooth = 1

#     if video == '' and images == '':
#         raise ValueError('Either --video or --image has to be provided')

#     net = PoseEstimationWithMobileNet()
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     load_state(net, checkpoint)

#     if video == '':
#         track = 0

#     cap = cv2.VideoCapture(video)
#     while cap.isOpened():
        
#         _, frame = cap.read()
#         pose_coordinations, image = run_demo(net, frame, height_size, cpu, track, smooth)
#         print(pose_coordinations.shape)
#         delay = 1
#         key = cv2.waitKey(delay)
#         if key == 27:  # esc
#             break
#         elif key == 112:  # 'p'
#             if delay == 1:
#                 delay = 0
#             else:
#                 delay = 1