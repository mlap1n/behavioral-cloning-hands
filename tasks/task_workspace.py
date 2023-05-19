import collections
from dm_control.manipulation.shared import workspaces


ROBOT_OFFSET = [-0.1, -0.9, 0.11]

###########################################################################

MovePropWorkspace = collections.namedtuple('MovePropWorkspace',
                                         ['prop_bbox',
                                          'target_bbox',
                                          'robot_offset'])


move_prop_workspace = MovePropWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.55, -0.4, 0.66),
        upper=(0.55, -0.62, 0.66)),
    target_bbox=workspaces.BoundingBox(
        lower=(0.5, 0.5, 2),
        upper=(1, 1, 2.5+0.1)),
    robot_offset=ROBOT_OFFSET)

###########################################################################

LiftWorkspace = collections.namedtuple('LiftWorkspace',
                                       ['prop_bbox', 'robot_offset'])


_box_size = 0.04
_box_mass = 1.3
lift_box_workspace = LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.55, -0.4, 0.66),
        upper=(0.55, -0.62, 0.66)),
    robot_offset=ROBOT_OFFSET)

###########################################################################

ReachWorkspace = collections.namedtuple('ReachWorkspace',
                                        ['target_bbox', 'robot_offset'])


reach_site_workspace = ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.6, -0.2, 0.66),
        upper=(0.6, -0.7, 1)),
    robot_offset=ROBOT_OFFSET)

###########################################################################
