from .config import ServerConfig


def create_app(config):
    from .app import create_app as _create_app

    return _create_app(config)


def run_server(config):
    from .app import run_server as _run_server

    return _run_server(config)


__all__ = ("ServerConfig", "create_app", "run_server")

