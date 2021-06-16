import numpy as np
from PIL import Image

from fetch_block_construction.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, obs_type, render_size
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string):            path to the environments XML file
            n_substeps (int):               number of substeps the simulation runs on every call to step
            gripper_extra_height (float):   additional height above the table when positioning the gripper
            block_gripper (boolean):        whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean):           whether or not the environment has an object
            target_in_the_air (boolean):    whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements):     offset of the target
            obj_range (float):              range of a uniform distribution for sampling initial object positions
            target_range (float):           range of a uniform distribution for sampling a target
            distance_threshold (float):     the threshold after which a goal is considered achieved
            initial_qpos (dict):            a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.grayscale = True
        self.normalize_img = True
        self.obs_type = obs_type
        self.render_size = render_size

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, render_size=render_size)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        '''
        1. Compute the distance (euclidean norm) between the achieved goal and the desired goal. Note that this could include multiple objects. 
        2. Once you have that distance, use logic to compute a spare or dense reward. 0 means success. negative values otherwise. 
        '''
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        # Return -1 for failure and 0 for success (distance below threshold)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)

        # Dense reward is just the -ve distance. The closer the achieved goal to the distance, the smaller the penalty.
        elif self.reward_type == "dense":
            return -d

        else:
            raise NotImplementedError

    def compute_reward_image(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            object_pos = self.sim.data.get_site_xpos('object0')
            achieved_goal = np.squeeze(object_pos.copy())
        d = np.linalg.norm(achieved_goal - self.goal.copy())
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        '''
        For some Fetch/HER environments, robot fingers are closed to enact push. 
        This is done at the end of the step. 
        '''
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def set_goal(self, goal):
        self.goal = goal

    def _set_action(self, action):
        '''
        Set the desired action to the robot's gripper and fingers util.via ctrl_set_action
        As well as mocaps. 
        '''
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        
        # Update mocap welded to robot gripper by adding a delta to position and quaternion
        utils.mocap_set_action(self.sim, action) 

    def _get_obs(self):
        '''
        Observations consist of follow:

        Robot Data
        1a. Extract robot xyz
        1b. Extract finger pos
        1c. Extract robot vel
        1d. Extract finger vel

        Object Data [1...n objs]
        2a. positions
        2b. orientation
        2c. linear vel
        2d. rotational vel
        2e. Relative between obj pos and gripper pos. 

        Achieve Goal
        if it has the object, use robotic gripper. otherwise use the object position. 
        '''
        # Robot Data:positions
        grip_pos    = self.sim.data.get_site_xpos('robot0:grip')
        dt          = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp   = self.sim.data.get_site_xvelp('robot0:grip') * dt

        # Object
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        
        if self.has_object:            
            # positions
            object_pos = self.sim.data.get_site_xpos('object0')

            # orientation
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))

            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            object_velp -= grip_velp

            # Relative Position wrt to gripper
            object_rel_pos = object_pos - grip_pos
            
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # Fingers
        gripper_state   = robot_qpos[-2:]
        gripper_vel     = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # Achieved Goal: if it has the object, use robotic gripper. otherwise use the object position. 
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        # Create Robot State info
        robot_state = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        # For images: normalize and append to observations. 
        if self.obs_type == 'dictimage':
            image_obs = self.render(mode='rgb_array', size=self.render_size)

            if self.grayscale:
                image_obs = Image.fromarray(image_obs).convert('L')
                image_obs = np.array(image_obs).flatten()

            if self.normalize_img:
                image_obs = image_obs / 255.0

            obs = np.concatenate([
                robot_state.copy(),
                image_obs.copy()])

            return {
                'observation':       obs.copy(),
                'achieved_goal':     achieved_goal.copy(),
                'desired_goal':      self.goal.copy(),
                'robot_state':       robot_state.copy(),
                'image_observation': image_obs.copy()
            }

        # If no image
        elif self.obs_type == 'dictstate':
            obs = np.concatenate([
                robot_state.copy(),
                object_pos.copy(),
                object_rot.copy(),
                object_velp.copy(),
                object_velr.copy(),
                object_rel_pos.copy(),
            ])
            return {
                'observation':      obs.copy(),
                'achieved_goal':    achieved_goal.copy(),
                'desired_goal':     self.goal.copy(),
                'robot_state':      robot_state.copy(),
            }

        # np?
        elif self.obs_type == 'np':
            image_obs = self.render(mode='rgb_array', size=self.render_size)

            if self.grayscale:
                image_obs = Image.fromarray(image_obs).convert('L')
                image_obs = np.array(image_obs).flatten()

            if self.normalize_img:
                image_obs = image_obs / 255.0

            obs = np.concatenate([
                robot_state.copy(),
                image_obs.copy()])

            return obs.copy()
        else:
            raise("Obs_type not recognized")

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        '''
        Strategy: follows HER strategy. 

        A. If robot has object then, set position of object.
        1. Set object pos to self.initial_gripper_xpos (xyz), computed under _env_setup()
            - If displacement between obj and gripper is less than 10, re-compute a position for object
            - Set position of object in simulation
        2. Step forward in simulation. 
        '''
        self.sim.set_state(self.initial_state)

        # 1. Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]

            # If gripper is within 10cm of the object, then change the xy pos of the object, by sampling from +- obj_range (passed into the environment)
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)

            # Overwrite the object_qpos with fresh posiiton            
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # 2. Update simulator. 
        self.sim.forward()

        return True

    def sample_goals(self, batch_size):
        '''
        Sampe batch size goals
        return as dictionary
        '''
        goals = []
        for i in range(batch_size):
            goals.append(self._sample_goal())

        return dict(desired_goal=np.asarray(goals))

    def _sample_goal(self):
        '''
        According to HER position, the robot will start with the object in hand (self.has_object) 50% of the time. 
        If has.object: randomly assigns x,y positions. And 50% of time set z=0, else 50% in the air
        else: place the goal near the initial gripper position within 15cm. 
        '''
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset

            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        '''
        Checks if the distance between the achieved goal and the desired goal are smaller than the treshold. 
        Returns bool. 
        '''
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        '''
        1. Set object positions
        2. Reset mocaps
        3. Set new mocap pos/quat for target
        4. Robot: set the initial_gripper_xpos
            - Under HER formulation, if the robot has the object, set height_offset to object 0 z-pos. 
        '''
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim) # sets mocap welds to [0,0...]
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Set the Robot's initial gripper position
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()

        # Set the height offset
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
