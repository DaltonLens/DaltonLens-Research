from cv2 import IMREAD_COLOR
from dlcharts.common.utils import swap_rb
import onnx
import onnxruntime as ort

import numpy as np
import cv2
import cv2.dnn

import torch

from dlcharts.pytorch import color_regression as cr

import time

def denormalize_to_rgbu8(input_bchw):
    input_chw = input_bchw.squeeze(0)
    input_hwc = np.transpose(input_chw, (1, 2, 0))
    input_hwc = 255.999 * np.clip((input_hwc*0.5 + 0.5), 0, 1)
    return input_hwc.astype(np.uint8)

onnx_file = "test.onnx"
sample_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

# Load the ONNX model
model = onnx.load(onnx_file)

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

device = torch.device("cpu")
preprocessor = cr.ImagePreprocessor(device, do_augmentations=False)
image_rgb = swap_rb(cv2.imread("inputs/tests/mpl-generated/img-00000.antialiased.png", cv2.IMREAD_COLOR))
input_tensor = preprocessor.transform (image_rgb, image_rgb)[0]
input_tensor.unsqueeze_ (0) # add the batch dim

ort_session = ort.InferenceSession(onnx_file)

start = time.time()
outputs = ort_session.run(
    None,
    {"input": input_tensor.detach().numpy()},
)
end = time.time()
print('Onnxruntime', outputs[0].shape)
print (f"{end-start:.2f} s")
output_rgb = denormalize_to_rgbu8(outputs[0])
cv2.imwrite("output_ort.png", swap_rb(output_rgb))

opencv_net = cv2.dnn.readNetFromONNX(onnx_file)
print(f"OpenCV model was successfully read. Layers {len(opencv_net.getLayerNames())}: \n")

opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

opencv_net.setInput (input_tensor.detach().numpy())
start = time.time()
output = opencv_net.forward()
end = time.time()
print ('OpenCV', output.shape)
output_rgb = denormalize_to_rgbu8(output)
cv2.imwrite("output_opencv_preprocessor.png", swap_rb(output_rgb))
print (f"{end-start:.2f} s")

input_blob = cv2.dnn.blobFromImage(image_rgb, 1.0 / (0.5*255.0), None, (0.5*255.0, 0.5*255.0, 0.5*255.0), False, False, cv2.CV_32F)
opencv_net.setInput (input_blob)
start = time.time()
output = opencv_net.forward()
end = time.time()
print ('OpenCV', output.shape)
output_rgb = denormalize_to_rgbu8(output)
cv2.imwrite("output_opencv.png", swap_rb(output_rgb))
print (f"{end-start:.2f} s")
