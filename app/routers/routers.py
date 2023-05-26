# routes.py

from fastapi import WebSocket
from fastapi.responses import HTMLResponse

from app.services.service import VideoService
from app.controllers.controllers import VideoController

# video_service = VideoService()
# video_controller = VideoController(video_service)


# async def video_stream(websocket: WebSocket):
#     await video_controller.render_video_stream(websocket)


# def setup_routes(app):
#     app.websocket_route("/ws")(video_stream)
