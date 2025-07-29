import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

Rotunbot_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/lzh/IsaacLab/Rotunbot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.4),         # 依然略高于地面
        rot = (0.0, 0.0, 0.0, 1),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
        },
        joint_vel={
            "joint1": 0.0,
            "joint2": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "main": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2"],
            effort_limit_sim=1000,
            velocity_limit_sim=1000.0,
            stiffness={
                "joint1": 0.0,
                "joint2": 0.0,
            },
            damping={
                "joint1": 0.5,
                "joint2": 0.5,
            },
        ),
    },
)
