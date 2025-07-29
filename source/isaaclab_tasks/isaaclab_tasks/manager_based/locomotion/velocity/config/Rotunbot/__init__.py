# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
"""
地形：崎岖地形(rough terrain),有地形生成器和难度课程。
用途：标准训练环境，适合训练机器人在复杂地形上行走。
奖励、终止、观测等：完整，适合正式训练。
"""
gym.register(
    id="Rough-Rotunbot-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RotunbotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RotunbotRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

"""
地形:崎岖地形,但用于“Play”模式。
区别：
环境数量:1,更适合测试和可视化。
地形课程关闭，地形数量减少，内存占用低。
随机扰动/推搡等事件关闭，更稳定。
观测扰动关闭，便于观察真实表现。
用途：用于演示、可视化、调试和模型评估。
"""
gym.register(
    id="Rough-Rotunbot-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RotunbotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RotunbotRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

""" 
地形：平地(flat terrain),无地形生成器。
区别：
无地形难度课程，地形始终为平面。
无高度扫描观测，观测量减少。
奖励参数适配平地。
用途：适合在平地上训练，便于对比和基础能力训练。
"""

gym.register(
    id="Flat-Rotunbot-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RotunbotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RotunbotFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

"""
地形：平地,Play模式。
区别：
环境数量少，适合测试。
无扰动、无观测噪声，便于可视化和调试。
用途：平地上的演示、可视化、调试和模型评估。
"""
gym.register(
    id="Flat-Rotunbot-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RotunbotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RotunbotFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
