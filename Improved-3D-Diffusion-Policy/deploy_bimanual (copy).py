import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
import numpy as np
import torch
from termcolor import cprint
#####################################################################################
# aloha thing
#####################################################################################
from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
    InterbotixRobotNode
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

from threading import Thread
from typing import Optional

from interbotix_common_modules.common_robot.exceptions import InterbotixException
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

import numpy as np
np.set_printoptions(suppress=True,precision=4)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Bool

import threading
import time

new_action = None
current_action = None
follower_bot_left = None
follower_bot_right = None
gripper_left_command = None
gripper_right_command = None
last_left_openness = None
last_right_openness = None

node = None
current_idx = 0
save_data_idx = 0
current_traj = []

lock = threading.Lock()

env = load_yaml( "/home/jiahe/data/config/env.yaml" )
world_2_head = np.array( env.get("world_2_head") )
world_2_left_base = np.array( env.get("world_2_left_base") )
world_2_right_base = np.array( env.get("world_2_right_base") )
left_ee_2_left_cam = np.array( env.get("left_ee_2_left_cam") )    
right_ee_2_right_cam = np.array( env.get("right_ee_2_right_cam") )
left_bias = world_2_left_base
left_tip_bias = np.array( env.get("left_ee_2_left_tip") )
right_bias = world_2_right_base
right_tip_bias = np.array( env.get("right_ee_2_right_tip") )
right_tip_bias2 = np.array( env.get("right_ee_2_right_tip2") )
right_tip_bias3 = np.array( env.get("right_ee_2_right_tip3") )

task_name = "insert_marker_into_cup"
data_idx = 21
if(task_name == "open_marker"):
    right_tip_bias = right_tip_bias3
if(task_name == "lift_ball"):
    right_tip_bias = right_tip_bias3
if(task_name == "ziploc"):
    right_tip_bias = right_tip_bias2
if(task_name == "handover_block"):
    right_tip_bias = right_tip_bias3

def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:

    global task_name
    global data_idx
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
    torque_on(follower_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    # print("start_arm_qpos: ", [start_arm_qpos] * 2)
    # start_arm_qpos[4] += 0.4
    data = np.load("/home/jiahe/data/raw_demo/{}/traj/{}.npy".format(task_name, data_idx), allow_pickle=True)

    left_joint = data[0]['left_pos'][0:6] 
    right_joint = data[0]['right_pos'][0:6] 
    start_poses = [ 
        # open_marker
        left_joint,
        right_joint
    ]
    # start_poses = [start_arm_qpos] * 2
    print("start_poses: ", start_poses)
    move_arms(
        [follower_bot_left, follower_bot_right],
        # [start_arm_qpos] * 2,
        start_poses,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [1.62, 1.62],
        moving_time=0.5
    )


class BimanualEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=224,
                 num_points=4096,
                 use_waist=False):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        
        self.use_waist = use_waist

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.node = create_interbotix_global_node('aloha')
    
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            act = action_util.joint25_to_joint32(act)
            
            filtered_act = act.copy()
            filtered_pos = filtered_act[:-12]
            filtered_handpos = filtered_act[-12:]
            if not self.use_waist:
                filtered_pos[0:6] = 0.
            
            # self.upbody_comm.set_pos(filtered_pos)
            # self.hand_comm.send_hand_cmd(filtered_handpos[6:], filtered_handpos[:6])
            
            
            cam_dict = self.camera()
            self.cloud_array.append(cam_dict['point_cloud'])
            self.color_array.append(cam_dict['color'])
            self.depth_array.append(cam_dict['depth'])
            
            try:
                hand_qpos = self.hand_comm.get_qpos()
            except:
                cprint("fail to fetch hand qpos. use default.", "red")
                hand_qpos = np.ones(12)
            env_qpos = np.concatenate([self.upbody_comm.get_pos(), hand_qpos])
            self.env_qpos_array.append(env_qpos)
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
    
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict
    
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
    
    
        # pos init
        qpos_init1 = np.array([-np.pi / 12, 0, 0, -1.6, 0, 0, 0, 
            -np.pi / 12, 0, 0, -1.6, 0, 0, 0])
        qpos_init2 = np.array([-np.pi / 12, 0, 1.5, -1.6, 0, 0, 0, 
                -np.pi / 12, 0, -1.5, -1.6, 0, 0, 0])
        hand_init = np.ones(12)
        # hand_init = np.ones(12) * 0

        if first_init:
            # ======== INIT ==========
            upbody_initpos = np.concatenate([qpos_init2])
            self.upbody_comm.init_set_pos(upbody_initpos)
            self.hand_comm.send_hand_cmd(hand_init[6:], hand_init[:6])

        upbody_initpos = np.concatenate([qpos_init1])
        self.upbody_comm.init_set_pos(upbody_initpos)
        q_14d = upbody_initpos.copy()
            
        body_action = np.zeros(6)
        
        # this is a must for eef pos alignment
        arm_pos, arm_rot_quat = action_util.init_arm_pos, action_util.init_arm_quat
        q_14d = self.arm_solver.ik(q_14d, arm_pos, arm_rot_quat)
        self.upbody_comm.init_set_pos(q_14d)
        time.sleep(2)
        
        print("Robot ready!")
        
        # ======== INIT ==========
        # camera.start()
        cam_dict = self.camera()
        self.color_array.append(cam_dict['color'])
        self.depth_array.append(cam_dict['depth'])
        self.cloud_array.append(cam_dict['point_cloud'])

        try:
            hand_qpos = self.hand_comm.get_qpos()
        except:
            cprint("fail to fetch hand qpos. use default.", "red")
            hand_qpos = np.ones(12)

        env_qpos = np.concatenate([self.upbody_comm.get_pos(), hand_qpos])
        self.env_qpos_array.append(env_qpos)
                        
        self.q_14d = q_14d
        self.body_action = body_action
    

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        obs_cloud = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
            
        return obs_dict


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 224
    num_points = 4096
    use_waist = True
    first_init = True
    record_data = True

    env = BimanualEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points,
                             use_waist=use_waist)

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    # if record_data:
    #     import h5py
    #     root_dir = "/home/gr1p24ap0049/projects/gr1-learning-real/"
    #     save_dir = root_dir + "deploy_dir"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     record_file_name = f"{save_dir}/demo.h5"
    #     color_array = np.array(env.color_array)
    #     depth_array = np.array(env.depth_array)
    #     cloud_array = np.array(env.cloud_array)
    #     qpos_array = np.array(env.qpos_array)
    #     with h5py.File(record_file_name, "w") as f:
    #         f.create_dataset("color", data=np.array(color_array))
    #         f.create_dataset("depth", data=np.array(depth_array))
    #         f.create_dataset("cloud", data=np.array(cloud_array))
    #         f.create_dataset("qpos", data=np.array(qpos_array))
        
    #     choice = input("whether to rename: y/n")
    #     if choice == "y":
    #         renamed = input("file rename:")
    #         os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
    #         new_name = record_file_name.replace("demo.h5", renamed+'.h5')
    #         cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
    #     else:
    #         cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
