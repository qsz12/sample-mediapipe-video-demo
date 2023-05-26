import cv2
import base64
from fastapi import WebSocket
import asyncio
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# class VideoService:
#     def __init__(self):
#         self.camera = cv2.VideoCapture(0)

#     def read_frame(self):
#         ret, frame = self.camera.read()

#         if not ret:
#             return None

#         _, buffer = cv2.imencode(".jpg", frame)
#         frame_data = base64.b64encode(buffer).decode("utf-8")
#         return frame_data

#     def close(self):
#         self.camera.release()

from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

FrameQueue = asyncio.Queue(maxsize=120)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    annotated_image = image.copy()
    height, width, _ = image.shape
    if detection_result:
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = _normalized_to_pixel_coordinates(
                    keypoint.x, keypoint.y, width, height
                )
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = "" if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + " (" + str(probability) + ")"
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(
                annotated_image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )
        return annotated_image
    else:
        return annotated_image


##################################################################

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a face landmarker instance with the live stream mode:
def print_result(
    result: FaceDetectorResult,
    output_image: mp.Image,
    timestamp_ms: int,
):
    try:
        image_copy = np.copy(output_image.numpy_view())
        annotated_image = visualize(image_copy, result)
        print(annotated_image)
        _, buffer = cv2.imencode(".jpg", annotated_image)
        frame_data = base64.b64encode(buffer).decode("utf-8")

        FrameQueue.put_nowait(frame_data)

        # # Send the frame to the client
        # await websocket.send_text(frame_data)

        # # Introduce a small delay between frames
        # await asyncio.sleep(0.033)  # Approximately 30 frames per second
    except Exception as exc:
        print(exc)


async def video_capture(websocket: WebSocket):
    await websocket.accept()

    # Open the camera
    camera = cv2.VideoCapture(0)

    try:
        while camera.isOpened():
            # Read a frame from the camera
            ret, frame = camera.read()

            if not ret:
                # Unable to read frame
                break

            ts = int(time.time() * 1000000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, ts)
            # Encode the frame as base64 string
            # _, buffer = cv2.imencode(".jpg", frame)
            # frame_data = base64.b64encode(buffer).decode("utf-8")

            # # Send the frame to the client
            if not FrameQueue.empty():
                while not FrameQueue.empty():
                    await websocket.send_text(FrameQueue.get_nowait())

            # # Introduce a small delay between frames
            # await asyncio.sleep(0.033)  # Approximately 30 frames per second
    finally:
        # Release the camera
        camera.release()

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="detector.tflite"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)
detector = FaceDetector.create_from_options(options)

