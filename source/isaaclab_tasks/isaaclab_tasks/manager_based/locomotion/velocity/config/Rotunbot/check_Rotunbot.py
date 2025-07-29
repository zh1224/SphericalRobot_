import argparse
import torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

## 改为你的机器人的参数配置脚本：
from Rotunbot import Rotunbot_CFG

def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    cfg = sim_utils.GroundPlaneCfg()  
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origins = torch.tensor([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]).to(device=sim.device)
    hr2 = Articulation(Rotunbot_CFG.replace(prim_path="/World/G1"))
    robots = [hr2]
    return robots, origins

def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        if count % 1000 == 0:
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
            print(">>>>>>>> Reset!")
        for robot in robots:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        for robot in robots:
            robot.update(sim_dt)
            root_pos = robot.data.root_pos_w  # 获取机器人根位置
            print(f"Robot height (z): {root_pos[:, 2].item():.3f}")  # 打印 z 坐标（高度）

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device,gravity=[0.0, 0.0, -9.81], )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])
    robots, origins = design_scene(sim)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, robots, origins)

if __name__ == "__main__":
    main()
    simulation_app.close()
