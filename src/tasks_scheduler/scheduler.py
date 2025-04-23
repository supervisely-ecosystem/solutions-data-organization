from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler


class SchedulerManager:
    def __init__(self):
        self.scheduler = BackgroundScheduler(jobstores={"default": MemoryJobStore()})
        self.jobs = {}
        self.scheduler.start()

    def add_job(self, job_id, func, sec, replace_existing=True, args=None):
        """Add a new scheduled job"""
        job = self.scheduler.add_job(
            func, "interval", args=args, seconds=sec, id=job_id, replace_existing=replace_existing
        )
        self.jobs[job_id] = job
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        interval = "Every"
        if h > 0:
            interval += f" {h}h"
        if m > 0:
            interval += f" {m}m"
        if s > 0:
            interval += f" {s}s"
        return job

    def remove_job(self, job_id):
        """Remove a scheduled job"""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            return True
        return False

    def modify_interval(self, job_id, interval_seconds):
        """Modify the interval of an existing job"""
        if job_id in self.jobs:
            self.scheduler.reschedule_job(job_id, trigger="interval", seconds=interval_seconds)
            return True
        return False

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
