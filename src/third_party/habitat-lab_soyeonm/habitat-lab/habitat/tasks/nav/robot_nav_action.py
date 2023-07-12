from typing import List, Tuple

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry

# from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle

@registry.register_task_action
class RobotNavAction(BaseVelAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        # config = kwargs["config"]
        # #breakpoint()
        # if config['agent_index'] ==0:
        # 	self.motion_type = "base_velocity"
        # elif config['agent_index'] ==1:
        # 	self.motion_type = "human_joints"

        # #self.motion_type = config.motion_control
        # if self.motion_type == "base_velocity":
        #     BaseVelAction.__init__(self, *args, **kwargs)

        # elif self.motion_type == "human_joints":
        #     raise Exception("Can't happen")

        # else:
        #     raise ValueError("Unrecognized motion type for oracle nav  action")

        # self._task = task
        
        # self._prev_ep_id = None
        # self._targets = {}

        # self.skill_done = False

        
        # print("RobotNavAction is called!")

        # self.poses: List[np.ndarray] = []
        # self.waypoints: List[np.ndarray] = []
        # self.waypoint_pointer: int = 0
        # self.prev_navigable_point: np.ndarray = np.array([])
        # self.prev_pose: Tuple[np.ndarray, np.ndarray]
        pass

    

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "robot_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self._counter = 0
        self.poses = []

    def step(self, *args, is_last_action, **kwargs):
        print("Robot Nav Action step!")
        if self.counter ==0:
            navigable_point = self._sim.pathfinder.get_random_navigable_point()
            _navmesh_vertices = np.stack(
                self._sim.pathfinder.build_navmesh_vertices(), axis=0
            )
            _island_sizes = [
                self._sim.pathfinder.island_radius(p) for p in _navmesh_vertices
            ]
            _max_island_size = max(_island_sizes)
            largest_size_vertex = _navmesh_vertices[
                np.argmax(_island_sizes)
            ]
            _largest_island_idx = self._sim.pathfinder.get_island(
                largest_size_vertex
            )

            start_pos = self._sim.pathfinder.get_random_navigable_point(
                    island_index=_largest_island_idx
                )
            self.cur_articulated_agent.sim_obj.translation = start_pos

        self.skill_done = False
        action_to_take = kwargs[self._action_arg_prefix + "robot_nav_action"]

        curr_path_points = []
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        self.poses.append(robot_pos)

        if self.motion_type == "base_velocity":
            # 0: stop, 1: forward, 2: left, 3: right
            if action_to_take == 1:  # forward
                vel = [self._config.forward_velocity, 0]
            elif action_to_take == 2:  # turn left #Just changed for oGN
                vel = [0, self._config.turn_velocity] #[0, -self._config.turn_velocity]
            elif action_to_take == 3:  # turn right
                vel = [0, -self._config.turn_velocity] #[0, self._config.turn_velocity]
            else:  # stop
                vel = [0, 0]
            # else:
            #     vel = [0, 0]
            self.skill_done = True
            kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
            return BaseVelAction.step(
                self, *args, is_last_action=is_last_action, **kwargs
            )

        else:
            raise ValueError(
                "Unrecognized motion type for oracle nav action"
            )
