from abc import ABC, abstractmethod
from typing import Tuple

import json
import numpy as np
import os
import rospy
import rospkg
import yaml

from gym import spaces

from geometry_msgs.msg import Twist

from rl_agent.utils.observation_collector import (
    ObservationCollector,
)

DEFAULT_ACTION_SPACE = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "default_settings_jackal.yaml",
)
DEFAULT_HYPERPARAMETER = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "hyperparameters",
    "default.json",
)
DEFAULT_NUM_LASER_BEAMS, DEFAULT_LASER_RANGE = 720, 30
GOAL_RADIUS = 0.33


class BaseDRLAgent(ABC):
    def __init__(
        self,
        ns: str = None,
        robot_name: str = None,
        hyperparameter_path: str = DEFAULT_HYPERPARAMETER,
        action_space_path: str = DEFAULT_ACTION_SPACE,
        *args,
        **kwargs,
    ) -> None:
        """[summary]

        Args:
            ns (str, optional):
                Robot specific ROS namespace extension.
                Defaults to None.
            robot_name (str, optional):
                Agent name (directory has to be of the same name).
                Defaults to None.
            hyperparameter_path (str, optional):
                Path to json file containing defined hyperparameters.
                Defaults to DEFAULT_HYPERPARAMETER.
            action_space_path (str, optional):
                Path to yaml file containing action space settings.
                Defaults to DEFAULT_ACTION_SPACE.
        """

        # Setup node namespace
        self._ns = BaseDRLAgent._create_namespace(ns, robot_name)
        self._sim_ns = robot_name

        # Load robot and model specific parameters
        self._hyperparams = BaseDRLAgent._load_hyperparameters(
            path=hyperparameter_path
        )
        (
            self._num_laser_beams,
            self._laser_range,
        ) = BaseDRLAgent._get_robot_settings(self._sim_ns)
        (
            self._discrete_actions,
            self._continuous_actions,
            self._is_holonomic,
        ) = BaseDRLAgent._read_action_space(action_space_path)

        self._observation_collector = ObservationCollector(
            self._ns, self._num_laser_beams, self._laser_range, True
        )

        self._action_pub = rospy.Publisher(
            f"{self._ns}cmd_vel",
            Twist,
            queue_size=1,
        )

    def get_observations(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
            Retrieves the latest synchronized observation.

            Returns:
                Tuple, where first entry depicts the observation data concatenated \
                into one array. Second entry represents the observation dictionary.
        """
        merged_obs, obs_dict = self._observation_collector.get_observations(
            kwargs
        )
        return merged_obs, obs_dict

    def publish_action(self, action: np.ndarray) -> None:
        """
        TODO
        Publishes an action on 'self._action_pub' (ROS topic).

        Args:
            action (np.ndarray):
                For none holonomic robots action is [xVel, angularVel]
                For holonomic robots action is [xVel, yVel, angularVel]
                xVel and yVel in m/s, angularVel in rad/s
        """
        assert len(action) == 3, f"Expected action of size 3"

        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]

        self._action_pub.publish(action_msg)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Infers an action based on the given observation.

        Args:
            obs (np.ndarray): Merged observation array.

        Returns:
            np.ndarray:
                Action in [linear velocity, angular velocity]
        """
        assert self._agent, "Agent model not initialized!"
        action = self._agent.predict(obs, deterministic=True)[0]

        if self._hyperparams["discrete_action_space"]:
            action = self._get_disc_action(action)
        # else:
        #     # clip action
        #     action = np.maximum(
        #         np.minimum(self._action_space.high, action),
        #         self._action_space.low,
        #     )

        return action

    def _get_disc_action(self, action: int) -> np.ndarray:
        """
            Returns defined velocity commands for parsed action index.\
            (Discrete action space)

            Args:
                action (int): Index of the desired action.

            Returns:
                np.ndarray: Velocity commands corresponding to the index.
        """
        return np.array(
            [
                self._discrete_actions[action]["linear"],
                self._discrete_actions[action]["angular"],
            ]
        )

    @abstractmethod
    def _setup_agent(self) -> None:
        """
        Sets up the new agent / loads a pretrained one.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplementedError

    @staticmethod
    def _create_namespace(ns: str, robot_name: str):
        ns = "" if ns is None or ns == "" else ns + "/"

        if robot_name == None:
            return ns
        return ns + robot_name + "/"

    @staticmethod
    def _load_hyperparameters(path: str) -> None:
        """
        Path should point to a file containing the
        specific hyperparameters used for training
        """
        try:
            with open(path, "r") as file:
                return json.load(file)
        except:
            with open(DEFAULT_HYPERPARAMETER, "r") as file:
                return json.load(file)

    @staticmethod
    def _get_robot_settings(ns: str):
        """
        Setup robot specific parameters by reading ros params
        """
        _num_laser_beams = rospy.get_param("laser_beams", None)
        _laser_range = rospy.get_param("laser_range", None)

        if _num_laser_beams is None:
            _num_laser_beams = DEFAULT_NUM_LASER_BEAMS
            print(
                f"{ns}:\t"
                "Unable to read the number of laser beams."
                f"Set to default: {DEFAULT_NUM_LASER_BEAMS}"
            )
        if _laser_range is None:
            _laser_range = DEFAULT_LASER_RANGE
            print(
                f"{ns}:\t"
                "Unable to read the laser range."
                f"Set to default: {DEFAULT_LASER_RANGE}"
            )

        return _num_laser_beams, _laser_range

    @staticmethod
    def _read_action_space(action_space_yaml_path: str) -> None:
        """
        Retrieves the robot action space from respective yaml file.

        Args:
            action_space_yaml_path (str):
                Yaml file containing the action space configuration.
        """
        assert os.path.isfile(
            action_space_yaml_path
        ), f"Action space file cannot by found at {action_space_yaml_path}"

        with open(action_space_yaml_path, "r") as fd:
            setting_data = yaml.safe_load(fd)

            return (
                setting_data["robot"]["discrete_actions"],
                {
                    "linear_range": setting_data["robot"]["continuous_actions"][
                        "linear_range"
                    ],
                    "angular_range": setting_data["robot"][
                        "continuous_actions"
                    ]["angular_range"],
                },
                setting_data["robot"]["holonomic"],
            )
