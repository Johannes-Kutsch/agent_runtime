from datetime import datetime


def now_local() -> datetime:
    return datetime.now().astimezone()
