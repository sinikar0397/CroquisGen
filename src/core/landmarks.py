"""Landmark layout: which index means which joint.

This is part of the contract, not part of the capture backend. Renderers index
into the arrays directly, so putting the layout next to the backend would make
every renderer depend on the backend and defeat the point of perception/ being
swappable.

Indices follow MediaPipe (Pose 33 points, Hands 21 points). A second backend
producing a different layout converts into these indices rather than adding a
layout of its own.
"""
from __future__ import annotations

# --- pose, 33 points -------------------------------------------------------
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_PINKY, R_PINKY = 17, 18      # hand extremities of the pose model, not the hand model
L_INDEX, R_INDEX = 19, 20
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32

# Torso-and-limbs skeleton. Face/finger detail is left out on purpose - croquis
# cares about the body's flow, not features.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
)

# --- hand, 21 points -------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP = 13
PINKY_MCP, PINKY_TIP = 17, 20

HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
