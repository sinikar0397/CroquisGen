"""Style -> renderer lookup.

The only module that imports from both owners' packages, which is why it is not
render/__init__.py: renderers import render.base, so a registry in the package
__init__ would run on their way in and fail as a circular import.

Renderers are built once and reused. Adding a style means three edits: Style in
core/types.py, the dict below, and style_for_direction() in app/gestures.py to
cover the extra sector.
"""
from __future__ import annotations

from constructive_anatomy.renderer import ConstructiveRenderer
from core.types import Style
from gesture_croquis.renderer import GestureCroquisRenderer

from .base import CroquisRenderer

_RENDERERS: dict[Style, CroquisRenderer] = {
    Style.GESTURE_CROQUIS: GestureCroquisRenderer(),
    Style.CONSTRUCTIVE: ConstructiveRenderer(),
}


def get_renderer(style: Style) -> CroquisRenderer:
    return _RENDERERS[style]


def renderer_name(style: Style) -> str:
    return _RENDERERS[style].name


__all__ = ["CroquisRenderer", "get_renderer", "renderer_name"]
