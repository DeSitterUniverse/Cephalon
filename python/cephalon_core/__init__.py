from .config import Settings, settings

__all__ = ["Settings", "settings", "app", "create_app"]


def __getattr__(name: str):
    if name in {"app", "create_app"}:
        from . import app_factory
        return getattr(app_factory, name)
    raise AttributeError(name)
