"""Style -> renderer lookup.

Renderers are built once and reused. Adding a style means adding it to Style in
types.py, registering it here, and widening style_for_direction() in gestures.py
to cover the extra sector.
"""
from __future__ import annotations

from ..types import Style
from .base import CroquisRenderer
from .constructive import ConstructiveRenderer
from .gesture_croquis import GestureCroquisRenderer

_RENDERERS: dict[Style, CroquisRenderer] = {
    Style.GESTURE_CROQUIS: GestureCroquisRenderer(),
    Style.CONSTRUCTIVE: ConstructiveRenderer(),
}


def get_renderer(style: Style) -> CroquisRenderer:
    return _RENDERERS[style]


def renderer_name(style: Style) -> str:
    return _RENDERERS[style].name


__all__ = ["CroquisRenderer", "get_renderer", "renderer_name"]
