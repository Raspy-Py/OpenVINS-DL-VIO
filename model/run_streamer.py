import os
import cv2
import time
import torch
import argparse
import onnxruntime
import numpy as np
import pandas as pd
from torch import nn 


class Streamer(object):
    def __init__(self, host='10.42.0.1', w=256, h=256, fps=30, port=5000):
        gst_out = (
            f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! '
            f'rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port} auto-multicast=true'
        )
        self.w, self.h = w, h
        self.fps = fps
        self.out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, self.fps, (self.w, self.h), True)

    def __del__(self):
        self.out.release()

    def send(self, frame):
        self.out.write(frame)

class CameraReader(object):
    def __init__(self, device="video2", w=256, h=256, fps=30):
        self.device_path = os.path.join("/dev", device)
        self.cap = cv2.VideoCapture(self.device_path)

        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.h, self.w = h, w
        self.offset_h = (cam_h - h) // 2
        self.offset_w = (cam_w - w) // 2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise IOError(f"Can't open: {self.device_path}")
        
    def get_frame(self):
        ret, img = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")

        return img[
            self.offset_h:self.offset_h + self.h, 
            self.offset_w:self.offset_w + self.w, 
            :]



class DKD(nn.Module):
    def __init__(self, radius=2, top_k=500, scores_th=0.5, n_limit=5000, detector="default"):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.detector_type = detector
        self.keypoint_detector = self.tiled_detect_keypoints

    def tiled_detect_keypoints(self, scores_map):
        def reshape_split(image, kernel):
            img_h, img_w = image.shape
            image = image.reshape(img_h // kernel, kernel, img_w // kernel, kernel).swapaxes(1, 2)
            flattened_tiles = image.reshape(img_h // kernel, img_w // kernel, kernel * kernel)
            return flattened_tiles
        
        _, _, h, w = scores_map.shape
        scores_map[:, :, :self.radius + 1, :] = 0
        scores_map[:, :, :, :self.radius + 1] = 0
        scores_map[:, :, h - self.radius:, :] = 0
        scores_map[:, :, :, w - self.radius:] = 0

        kernel = 4

        reshaped_image = reshape_split(scores_map.squeeze(0).squeeze(0), kernel)
        num_tiles_h, num_tiles_w, _ = reshaped_image.shape
    
        argmax_indices = np.argmax(reshaped_image, axis=2)
        values = np.take_along_axis(reshaped_image, argmax_indices[:, :, np.newaxis], axis=2).squeeze()

        # global indices
        tile_row_starts = np.arange(num_tiles_h) * kernel
        tile_col_starts = np.arange(num_tiles_w) * kernel

        global_row_indices = tile_row_starts[:, np.newaxis] + argmax_indices // kernel
        global_col_indices = tile_col_starts + argmax_indices % kernel
        flat_indices = np.argsort(values.ravel())[-self.top_k:]
        top_values = values.ravel()[flat_indices]
        top_row_indices = global_row_indices.ravel()[flat_indices]
        top_col_indices = global_col_indices.ravel()[flat_indices]

        keypoints_xy = np.vstack((top_col_indices, top_row_indices)).T
        #keypoints_xy = keypoints_xy / keypoints_xy.new_tensor([w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

        return keypoints_xy, top_values
    
    def sample_descriptor(self, descriptor_map, kpts, bilinear_interp=False):
        """
        :param descriptor_map: CxHxW
        :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
        :param bilinear_interp: bool, whether to use bilinear interpolation
        :return: descriptors: list, len=B, each is NxD
        """
        _, _, height, width = descriptor_map.shape

        kptsi = kpts # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = torch.nn.functional.grid_sample(descriptor_map.unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            #kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            #kptsi = kptsi.long()
            descriptors_ = descriptor_map[:, :, kptsi[:, 1], kptsi[:, 0]]  # CxN
        descriptors_ = descriptors_ / np.linalg.norm(descriptors_, axis=1)

        return descriptors_.T

    def forward(self, scores_map, descriptor_map):
        """
        :param scores_map:  1xHxW
        :param descriptor_map: CxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        keypoints, kptscores = self.tiled_detect_keypoints(scores_map)
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
        return keypoints, descriptors, kptscores

def preprocess_image(raw):
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32), raw

class ALIKEPipeline:
    def __init__(self, shape="480x640", frames=1, n_desc=500):
        # Feature Extractor
        self.onnx_path = "./model.onnx"

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list   
        self.input_shape = (1, 3, self.w, self.h)

        # DKD
        self.sort = False
        self.dkd = DKD(radius=2, top_k=n_desc, scores_th=0.5, n_limit=5000)

        # Data
        self.input_node = "image"

        # TIDL runtime
        self.inference_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )


    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }


    def forward(self, image):
        descriptors, scores_map = self.inference_session.run(None, {self.input_node: image})
        # ==================== extract keypoints
        with torch.no_grad():
            keypoints, descriptors, scores = self.dkd(scores_map, descriptors)
            #keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[self.w - 1, self.h - 1]])


            if self.sort:
                indices = torch.argsort(scores, descending=True)
                keypoints = keypoints[indices]
                descriptors = descriptors[indices]
                scores = scores[indices]

        return keypoints, descriptors, scores


        # keypoints
        # descriptors
        # scores

#class SimpleTracker(object):


if __name__ == "__main__":
    camera_reader = CameraReader()
    video_streamer = Streamer()
    #alike_pipeline = ALIKEPipeline()
    #tracker = SimpleTracker()

    while True:
        image = camera_reader.get_frame()
        video_streamer.send(image)
