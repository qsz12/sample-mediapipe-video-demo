from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from app.controllers.controllers import VideoController
import uvicorn
from app.services.service import video_capture
from app.controllers.controllers import VideoController
from asyncio import Queue

from fastapi.middleware import Middleware
from fastapi.templating import Jinja2Templates
from starlette_context.middleware import RawContextMiddleware
from starlette_context import context

middleware = [Middleware(RawContextMiddleware)]
app = FastAPI(middleware=middleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    # context["websocketObject"] = websocket
    # CONTROLLER = VideoController(web_socket=websocket)
    # socket_queue = Queue
    # await CONTROLLER.video_controller(socket_queue)
    await video_capture(websocket)



@app.get("/")
async def get(request: Request):
    # Serve the index.html file
    # return {"message": "Server is running. Open index.html in a web browser."}
    return templates.TemplateResponse("index.html", {"request": request, "id": id})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
