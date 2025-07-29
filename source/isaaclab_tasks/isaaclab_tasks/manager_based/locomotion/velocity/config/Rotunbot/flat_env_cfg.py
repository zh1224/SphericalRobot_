# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import RotunbotRoughEnvCfg           #一样的，将脚本中的'Rotunbot'替换为自己的机器人名称,其他的不用改


@configclass
class RotunbotFlatEnvCfg(RotunbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # self.rewards.feet_air_time.weight = 1.0
        # self.rewards.feet_air_time.params["threshold"] = 0.6


class RotunbotFlatEnvCfg_PLAY(RotunbotFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
