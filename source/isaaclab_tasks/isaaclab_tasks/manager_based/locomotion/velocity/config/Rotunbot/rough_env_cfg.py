# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.Rotunbot_velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from .Rotunbot import Rotunbot_CFG                                    # 导入前面编写的机器人参数配置脚本
import random  

@configclass
class RotunbotRewards(RewardsCfg):
    """Reward terms for the MDP."""
    
    """机器人死亡时的惩罚"""
    termination_penalty = RewTerm(
        func=mdp.is_terminated, 
        weight=-200.0)
    
    lin_vel_z_l2 = None
    
    """奖励机器人在机器人自身的朝向坐标系下跟踪期望的xy线速度"""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,  
        params={"command_name": "base_velocity", 
                "std": 0.5},  
    )
    
    """奖励机器人在世界坐标系下跟踪期望的z轴角速度"""
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", 
                "std": 0.5}
    )
    
    

    """惩罚膝关节(knee)偏离默认值"""


"""对应:Rough-Rotunbot-train"""
@configclass
class RotunbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RotunbotRewards = RotunbotRewards()

    def __post_init__(self):
        super().__post_init__()
        # Scene
        self.scene.robot = Rotunbot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"  # 将robotis_Rotunbot/base替换为自己的机器人名称/base_link的link名称
    
        self.events.push_robot = None     
        self.events.add_base_mass = None  
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*base_link"] # 将base替换为自己的机器人base_link的link名称
        self.events.reset_base.params = {
            "pose_range": {"x": (0, 0), "y": (-0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*base_link"]

        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 前后速度范围
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)   # 左右速度范围
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # 旋转速度范围

        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*base_link"

        
"""对应:Rough-Rotunbot-Play"""
@configclass
class RotunbotRoughEnvCfg_PLAY(RotunbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


        #　与RotunbotRoughEnvCfg的commands对应:
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
