import os

workers = 4
threads = 8
preload_app = True
bind = "127.0.0.1:8050"

APSC_WORKER = None   # Reference of the posssibly chosen APScheduler worker


def pre_fork(server, worker):
    """
    Before we fork new worker, let's see if we want to choose it as the worker
    that runs APScheduler
    """
    global APSC_WORKER
    global preload_app
    if APSC_WORKER is None and not preload_app:
        APSC_WORKER = worker


def post_fork(server, worker):
    """
    After worker process is forked, check if its the APScheduler worker
    If it is, then set os.environ["APSC"] = "1", otherwise we set it to "0"
    """
    global APSC_WORKER
    os.environ["WORKER_PID"] = str(worker.pid)
    if worker == APSC_WORKER:
        os.environ["APSC"] = "1"
    else:
        os.environ["APSC"] = "0"


def child_exit(server, worker):
    """
    If for any reason (ex: max requests a worker will handle has reached) the worker exited
    then we set APSC_WORKER to None so when the new worker spawns we setup APScheduler again
    """
    global APSC_WORKER
    if worker == APSC_WORKER:
        APSC_WORKER = None
