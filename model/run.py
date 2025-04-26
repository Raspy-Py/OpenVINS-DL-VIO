import os
import shutil
import time
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd


inference_frames = 0
duration = 0
device = "tidl"

def preprocess_image(image):
    image = image.copy() / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


def compare_float_3d_arrays(arr1, arr2, error_margin=0.01):
    total_elements = arr1.size
    matching_elements = np.sum(np.abs(arr1 - arr2) <= error_margin)

    similarity = (matching_elements / total_elements) * 100
    print(f"The arrays are {similarity:.2f}% similar.")


def relative_difference_for_arrays(arr1, arr2):
    rel_diff = np.mean(np.abs(arr1 - arr2) / np.abs(arr1)) * 100
    print(f"Relative difference: {rel_diff:.2f}%")


class DeviceExecutor:
    def __init__(self, shape="480x640", frames=1):
        self.onnx_path = "./model.onnx"
        self.artifacts_folder = "./artifacts" 

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list
        self.input_shape = (1, 3, self.w, self.h)

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
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        self._run(inference_tidl_session, executor="tidl")

    def run_cpu(self):
        inference_cpu_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=onnxruntime.SessionOptions(),
        )
        self._run(inference_cpu_session, executor="cpu")


    def _run(self, inference_session: onnxruntime.InferenceSession, executor: str = "cpu"):
        # Warm up the device before measuring time 
        for _ in range(self.warm_up_iterations):
            inference_session.run(None, {self.input_node: self.data[0]})

        start_time =  time.time_ns()
        for inputs in self.data[:self.inference_frames]:
            inference_session.run(None, {self.input_node: inputs})
        end_time = time.time_ns()

        global inference_frames, duration, device
        inference_frames = self.inference_frames
        duration = end_time - start_time
        device = executor

if __name__ == "__main__":
    Fire(DeviceExecutor)
    print(f"Completed {inference_frames} inference iterations on {device} executor in {duration * 1e-6} ms.")
    print("Done")
