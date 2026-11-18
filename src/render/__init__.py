"""The drawing contract and the tools every style shares.

Deliberately empty of imports. Renderers import `render.base` and
`render.outline`, so anything imported here would run on their way in and close
a cycle - which is exactly why the style registry lives in render/registry.py
instead of this file.

    render/base.py      CroquisRenderer protocol, blank paper, scaling, strokes
    render/outline.py   silhouette contour from the segmentation mask
    render/registry.py  Style -> renderer, the one place both owners' packages meet
"""
