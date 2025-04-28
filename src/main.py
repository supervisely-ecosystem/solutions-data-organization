# from contextlib import asynccontextmanager

import supervisely as sly
# import supervisely.app.widgets as w
# from fastapi import FastAPI

import src.sly_globals as g
from src.sampling import sampling_router
from src.move_images import move_images_router

# Sly application
app = sly.Application()

# FastAPI application
server = app.get_server()

# Register the router
server.include_router(sampling_router)
server.include_router(move_images_router)


# or
# @server.shutdown()
# def shutdown():
#     g.scheduler_manager.shutdown()


# @asynccontextmanager
# async def lifespan(server: FastAPI):
#     yield
#     g.scheduler_manager.shutdown()
