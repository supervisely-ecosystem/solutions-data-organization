import os

import supervisely as sly
from dotenv import load_dotenv

# from src.tasks_scheduler.scheduler import SchedulerManager

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()
# scheduler_manager = SchedulerManager()


# class SchedulerJobs:
#     pass

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
