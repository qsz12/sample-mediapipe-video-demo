import cv2
import base64
from fastapi import WebSocket
from starlette_context import context
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
import math
from typing import Tuple, Union
import math
import cv2
import numpy as np
from loguru import logger

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode


class VideoService:
    def __init__(self, web_socket) -> None:
        self.websocket = web_socket


        async def _normalized_to_pixel_coordinates(
                self,
                normalized_x: float,
                normalized_y: float,
                image_width: int,
                image_height: int,
            ) -> Union[None, Tuple[int, int]]:
                async def is_valid_normalized_value(value: float) -> bool:
                    return (value > 0 or math.isclose(0, value)) and (
                        value < 1 or math.isclose(1, value)
                    )

                if not (
                    await is_valid_normalized_value(normalized_x)
                    and await is_valid_normalized_value(normalized_y)
                ):
                    # TODO: Draw coordinates even if it's outside of the image bounds.
                    return None
                x_px = min(math.floor(normalized_x * image_width), image_width - 1)
                y_px = min(math.floor(normalized_y * image_height), image_height - 1)
                return x_px, y_px

        async def visualize(self, image, detection_result) -> np.ndarray:
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
                            keypoint_px = await self._normalized_to_pixel_coordinates(
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
                        text_location = (
                            MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y,
                        )
                        cv2.putText(
                            annotated_image,
                            result_text,
                            text_location,
                            cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE,
                            TEXT_COLOR,
                            FONT_THICKNESS,
                        )
                    _, buffer = cv2.imencode(".jpg", annotated_image)
                    frame_data = base64.b64encode(buffer).decode("utf-8")
                    return frame_data
                else:
                    _, buffer = cv2.imencode(".jpg", annotated_image)
                    frame_data = base64.b64encode(buffer).decode("utf-8")
                    return frame_data
                
        async def run_async_functions(self,image_copy, result):
                frame_data = await self.visualize(image_copy, result)
                result_2 = await self.websocket.send_text(frame_data)
                return result_2

        def print_result(
                result: FaceDetectorResult,
                output_image: mp.Image,
                timestamp_ms: int,
            ):
                try:
                    self = VideoService(self.websocket)
                    loop1 = asyncio.get_running_loop()
                    asyncio.set_event_loop(loop1)
                    image_copy = np.copy(output_image.numpy_view())
                    # annotated_image = loop1.run_until_complete(
                    #     self.visualize(image_copy, result)
                    # )
                    # print(type(annotated_image))
                    

                    # Send the frame to the client
                    # loop2 = asyncio.new_event_loop()
                    # asyncio.set_event_loop(loop2)
                    result = loop1.run_until_complete(self.run_async_functions(self= VideoService, image_copy=image_copy, result=result))
                    #result = loop1.run_until_complete(self.websocket.send_text(frame_data))

                except Exception as exc:
                    logger.exception(exc)

        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="detector.tflite"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result,
        )

        self.detector = FaceDetector.create_from_options(options)

    # async def _normalized_to_pixel_coordinates(
    #     self,
    #     normalized_x: float,
    #     normalized_y: float,
    #     image_width: int,
    #     image_height: int,
    # ) -> Union[None, Tuple[int, int]]:
    #     async def is_valid_normalized_value(value: float) -> bool:
    #         return (value > 0 or math.isclose(0, value)) and (
    #             value < 1 or math.isclose(1, value)
    #         )

    #     if not (
    #         await is_valid_normalized_value(normalized_x)
    #         and await is_valid_normalized_value(normalized_y)
    #     ):
    #         # TODO: Draw coordinates even if it's outside of the image bounds.
    #         return None
    #     x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    #     y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    #     return x_px, y_px

    # async def visualize(self, image, detection_result) -> np.ndarray:
    #     annotated_image = image.copy()
    #     height, width, _ = image.shape
    #     if detection_result:
    #         for detection in detection_result.detections:
    #             # Draw bounding_box
    #             bbox = detection.bounding_box
    #             start_point = bbox.origin_x, bbox.origin_y
    #             end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    #             cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    #             # Draw keypoints
    #             for keypoint in detection.keypoints:
    #                 keypoint_px = await self._normalized_to_pixel_coordinates(
    #                     keypoint.x, keypoint.y, width, height
    #                 )
    #                 color, thickness, radius = (0, 255, 0), 2, 2
    #                 cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    #             # Draw label and score
    #             category = detection.categories[0]
    #             category_name = category.category_name
    #             category_name = "" if category_name is None else category_name
    #             probability = round(category.score, 2)
    #             result_text = category_name + " (" + str(probability) + ")"
    #             text_location = (
    #                 MARGIN + bbox.origin_x,
    #                 MARGIN + ROW_SIZE + bbox.origin_y,
    #             )
    #             cv2.putText(
    #                 annotated_image,
    #                 result_text,
    #                 text_location,
    #                 cv2.FONT_HERSHEY_PLAIN,
    #                 FONT_SIZE,
    #                 TEXT_COLOR,
    #                 FONT_THICKNESS,
    #             )
    #         _, buffer = cv2.imencode(".jpg", annotated_image)
    #         frame_data = base64.b64encode(buffer).decode("utf-8")
    #         return frame_data
    #     else:
    #         _, buffer = cv2.imencode(".jpg", annotated_image)
    #         frame_data = base64.b64encode(buffer).decode("utf-8")
    #         return frame_data


    async def video_capture(self):
        await self.websocket.accept()

        # pen the camera
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
                self.detector.detect_async(mp_image, ts)
                # Encode the frame as base64 string
                # _, buffer = cv2.imencode(".jpg", frame)
                # frame_data = base64.b64encode(buffer).decode("utf-8")

                # # Send the frame to the client
                # await websocket.send_text(frame_data)

                # # Introduce a small delay between frames
                # await asyncio.sleep(0.033)  # Approximately 30 frames per second
        finally:
            # Release the camera
            camera.release()
