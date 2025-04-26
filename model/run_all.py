import os
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd
from torch import nn 
import time

import torch
torch.set_num_threads(1)

# Making it global to display in the end
total_time = 0
feature_extrator_time = 0
detect_keypoints_time = 0
sample_descriptor_time = 0
simple_nms_time = 0
for_loop_time = 0
top_k_time = 0
stacking_time = 0
full_dkd_time = 0

class DKD(nn.Module):
    def __init__(self, radius=2, top_k=500, scores_th=0.5, n_limit=5000, detector="tiled"):
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
        if detector == "maxpool":
            self.keypoint_detector = self.detect_keypoints
        else:
            self.keypoint_detector = self.tiled_detect_keypoints

    def tiled_detect_keypoints(self, scores_map):
        _, _, h, w = scores_map.shape
        scores = scores_map.squeeze()
        
        # Zero out borders
        r = self.radius + 1
        scores[:r, :] = scores[-r:, :] = scores[:, :r] = scores[:, -r:] = 0

        kernel = 4
        num_tiles_h, num_tiles_w = h // kernel, w // kernel
        
        reshaped = scores.reshape(num_tiles_h, kernel, num_tiles_w, kernel).swapaxes(1, 2)
        reshaped = reshaped.reshape(num_tiles_h, num_tiles_w, -1)
        argmax_indices = reshaped.argmax(axis=2)
        values = reshaped[np.arange(num_tiles_h)[:, None], np.arange(num_tiles_w), argmax_indices]

        row_indices = (np.arange(num_tiles_h) * kernel)[:, None] + argmax_indices // kernel
        col_indices = (np.arange(num_tiles_w) * kernel) + argmax_indices % kernel

        flat_indices = np.argpartition(values.ravel(), -self.top_k)[-self.top_k:]
        top_values = values.ravel()[flat_indices]
        top_row_indices = row_indices.ravel()[flat_indices]
        top_col_indices = col_indices.ravel()[flat_indices]

        keypoints_xy = np.column_stack((top_col_indices, top_row_indices))

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
        descriptors_ = descriptors_.squeeze(0)
        descriptors_ = descriptors_ / np.linalg.norm(descriptors_, axis=0)
        return descriptors_.T

    def forward(self, scores_map, descriptor_map):
        """
        :param scores_map:  1xHxW
        :param descriptor_map: CxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """
        global detect_keypoints_time
        global sample_descriptor_time

        start_time = time.time_ns()
        keypoints, kptscores = self.tiled_detect_keypoints(scores_map)
        end_time = time.time_ns()
        detect_keypoints_time = end_time - start_time

        start_time = time.time_ns()
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
        end_time = time.time_ns()
        sample_descriptor_time = end_time - start_time

        # keypoints: B M 2
        # descriptors: B M D
        # scoredispersitys:
        return keypoints, descriptors, kptscores
    
# Inference code
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


class ALIKEPipeline:
    def __init__(self, shape="480x640", frames=1, n_desc=500):
        # Feature Extractor
        self.onnx_path = "./model.onnx"
        self.reports_folder = "./reports"
        self.artifacts_folder = "./artifacts"

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list   
        self.input_shape = (1, 3, self.w, self.h)

        # DKD
        self.sort = False
        self.dkd = DKD(radius=2, top_k=n_desc, scores_th=0.5, n_limit=5000)

        # Data
        self.input_node = "image"
        self.inference_frames = int(frames)
        self.data = self.get_calibration_tensors()

        self.warm_up_iterations = 10


    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }

    def get_calibration_tensors(self):
        calibration_dataset_path = "/opt/model_zoo/alike-data/"
        data = pd.read_csv(os.path.join(calibration_dataset_path, "data.csv"))
        data["image_path"] = (
            f"{calibration_dataset_path}/"
            + data["image_folder_path"]
            + "/"
            + data["filename"]
        )

        preprocessed_images = []
        for row_index, row in data.iterrows():
            if len(preprocessed_images) >= self.inference_frames:
                break
            image = cv2.imread(row["image_path"])
            if image is None:
                print(f"Image not found: {row['image_path']}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocessed_images.append(preprocess_image(image)[:,:,:self.w,:self.h])
        
        return preprocessed_images


    def run_tidl(self):
        global feature_extrator_time, total_time, full_dkd_time

        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )

        # Warm up the device before measuring time 
        print(f"Running {self.warm_up_iterations} warmup iterations...")
        for _ in range(self.warm_up_iterations):
            inference_tidl_session.run(None, {self.input_node: self.data[0]})

        print(f"Measuring time...")
        start_time = time.time_ns()
        descriptors, scores_map = inference_tidl_session.run(None, {self.input_node: self.data[0]})
        end_time = time.time_ns()
        feature_extrator_time = end_time - start_time 
        # ==================== extract keypoints
        with torch.no_grad():
            full_dkd_start = time.time_ns()
            keypoints, descriptors, scores = self.dkd(scores_map, descriptors)
            full_dkd_end = time.time_ns()

            full_dkd_time = full_dkd_end - full_dkd_start

        end_total = time.time_ns()
        total_time = end_total - start_time




if __name__ == "__main__":
    Fire(ALIKEPipeline)
    print(f"{'Full pipeline time: '}            {total_time * 1e-6:.1f}ms.")
    print(f" -{'Feature extractor time: '}      {feature_extrator_time * 1e-6:.1f}ms.")
    print(f" -{'Full DKD time: '}               {full_dkd_time * 1e-6:.1f}ms.")
    print(f" --{'Detect keypoints time: '}      {detect_keypoints_time * 1e-6:.1f}ms.")
    print(f" --{'Sample descriptor time: '}     {sample_descriptor_time * 1e-6:.1f}ms.")
    print(f" -{'Amortization time: '}           {(total_time - feature_extrator_time - sample_descriptor_time - detect_keypoints_time) * 1e-6:.1f}ms.")
    print("Done")
