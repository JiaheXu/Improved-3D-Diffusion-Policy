import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
# import diffusion_policy_3d.common.gr1_action_util as action_util
# import diffusion_policy_3d.common.rotation_util as rotation_util
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
#zed camera
#####################################################################################
import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1,-1 )

#####################################################################################
# aloha thing
#####################################################################################
from aloha.constants import (
    # DT_DURATION,
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
import yaml
import threading
import time

from utils import *


from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.05 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)

def load_yaml(file_dir):
    # Load the YAML file
    with open(file_dir, "r") as file:
        data = yaml.safe_load(file)

    return data

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

task_name = "straighten_rope"
data_idx = 21




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

        ################################################
        # camera stuff
        ################################################
        ip_address = "10.3.1.4:30000"
        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_parameters.coordinate_units = sl.UNIT.MILLIMETER
        init_parameters.depth_minimum_distance = 300 # 0.3m
        init_parameters.sdk_verbose = 1
        init_parameters.set_from_stream(ip_address.split(':')[0],int(ip_address.split(':')[1]))
        self.cam = sl.Camera()
        status = self.cam.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()
        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.depth_mat = sl.Mat(1920, 1080, sl.MAT_TYPE.U16_C1, sl.MEM.CPU)

        self.head_cam = load_yaml( "/home/jiahe/data/config/head.yaml" )
        self.head_cam_intrinsic_np = np.array( self.head_cam.get("intrinsic") )
        o3d_data = self.head_cam.get("intrinsic_o3d")[0]
        self.head_cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])
        # self.head_cam_extrinsic = self.world_2_head

        self.env = load_yaml( "/home/jiahe/data/config/env.yaml" )
        self.world_2_head = np.array( self.env.get("world_2_head") )
        self.head_bound_box = np.array( self.env.get("head_bounding_box") )
        self.hand_bound_box = np.array( self.env.get("hand_bounding_box") )
        o3d_data = self.env.get("intrinsic_o3d")[0]
        self.cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])
        self.cam_intrinsic_np = np.array( self.env.get("intrinsic") )
        self.resized_image_size = (256,256)

    def get_obs(self):

        err = self.cam.grab(self.runtime) #Check that a new image is successfully acquired
        cam_dict = None
        while( cam_dict is None):
            if err == sl.ERROR_CODE.SUCCESS:
                self.cam.retrieve_image(self.mat, sl.VIEW.LEFT) #Retrieve left image
                head_rgb = self.mat.get_data()
                head_rgb = head_rgb[:,:,:3]
                self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH_U16_MM)
                head_depth = self.depth_mat.get_data().astype(np.uint16)
                
                head_resized_rgb, head_resized_depth = transfer_camera_param(head_rgb, head_depth, self.head_cam_intrinsic_np, self.cam_intrinsic_np, self.resized_image_size )

                colored_cloud = create_colored_point_cloud( head_resized_rgb, head_resized_depth, self.cam_intrinsic_np )
                cam_dict = {}
                cam_dict['color'] = head_resized_rgb
                cam_dict['depth'] = head_resized_depth            
                cam_dict['point_cloud'] = colored_cloud
        return cam_dict

    def step(self, action_list):
        global follower_bot_left
        global follower_bot_right

        # Teleoperation loop
        gripper_left_command = JointSingleCommand(name='gripper')
        gripper_right_command = JointSingleCommand(name='gripper')

        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            # act = action_util.joint25_to_joint32(act)
            left_cmds = act[0:6]
            right_cmds = act[7:14]            
            follower_bot_left.arm.set_joint_positions(left_cmds, blocking=False)
            follower_bot_right.arm.set_joint_positions(right_cmds, blocking=False)

            # sync gripper positions
            left_openness = act[6]
            right_openness = act[13]
            gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
                left_openness
            )
            gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
                right_openness
            )
            follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
            follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
            # sleep DT
            get_interbotix_global_node().get_clock().sleep_for(CONTROL_DT_DURATION)
            
            cam_dict = self.get_obs()
            if self.use_point_cloud:
                self.cloud_array.append(cam_dict['point_cloud'])
            if self.use_image:
                self.color_array.append(cam_dict['color'])
            # self.depth_array.append(cam_dict['depth'])
        
            follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
            follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]

            env_qpos = np.concatenate([follower_left_state_joints, follower_right_state_joints])
            
            self.env_qpos_array.append(env_qpos)
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)    
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict
    
    def reset(self, first_init=True):
        global follower_bot_left
        global follower_bot_right
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        print("Robot ready!")
        
        # ======== INIT ==========
        # camera.start()
        cam_dict = self.get_obs()
        self.color_array.append(cam_dict['color'])
        self.depth_array.append(cam_dict['depth'])
        self.cloud_array.append(cam_dict['point_cloud'])


        follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
        follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]
        env_qpos = np.concatenate([follower_left_state_joints, follower_right_state_joints])
        self.env_qpos_array.append(env_qpos)

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
    global follower_bot_left
    global follower_bot_right
    ################################################
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

    node = create_interbotix_global_node('aloha')

    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    robot_startup(node)

    opening_ceremony(
        follower_bot_left,
        follower_bot_right,
    )




    # node.bgr_sub = Subscriber(node, Image, "/camera_1/left_image")
    # node.depth_sub = Subscriber(node, Image, "/camera_1/depth")


    queue_size = 1000
    max_delay = 0.05 # 50ms

    env = BimanualEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points,
                             use_waist=use_waist)

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    
    while step_count < roll_out_length:
        print("obs_dict: ")
        print("point_cloud: ", obs_dict['point_cloud'].shape)
        print("agent_pos", obs_dict['agent_pos'].shape)
        print("obs_dict: ", obs_dict)
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")


if __name__ == "__main__":
    main()
