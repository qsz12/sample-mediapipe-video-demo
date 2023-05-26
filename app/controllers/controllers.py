# controller.py
import sys
from starlette_context import context
from loguru import logger
from app.services.video_service import VideoService


class VideoController:
    def __init__(self, web_socket) -> None:
        self.web_socket = web_socket

    async def video_controller(self, queue):
        try:
            logger.info(dict(context.data))
            # sys.exit()
            VIDEO_SERVICE = VideoService(web_socket=self.web_socket)
            await VIDEO_SERVICE.video_capture()
        except Exception as exc:
            logger.exception(exc)
