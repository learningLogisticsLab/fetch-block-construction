import os
import tempfile
import numpy as np

# Gym
from gym import utils as gym_utils

# Fetch
from fetch_block_construction.envs.robotics import utils
from fetch_block_construction.envs.robotics import fetch_env, rotations

# Mujoco
import mujoco_py
from .xml import generate_xml

# Key class to construct the Fetch environment

class FetchBlockConstructionEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    '''
    First assign parameters related to the block construction environment. 
    After that call the fetch_env.FetchEnv class to generate it (includes strategies related to HER)
    '''

    def __init__(self, initial_qpos,
                 num_blocks=1,
                 reward_type='incremental',
                 obs_type='np',
                 render_size=42,
                 stack_only=False,
                 case="Singletower"):

        self.num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.stack_only = stack_only
        self.case = case

        # Ensure we get the path separator correct on windows
        # MODEL_XML_PATH = os.path.join('fetch', F'stack{self.num_blocks}.xml')

        with tempfile.NamedTemporaryFile(mode='wt', dir=F"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/assets/fetch/", delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.num_blocks)) # Generates an xml file based on the number of objects. 
            MODEL_XML_PATH = fp.name

        # Call parent Fetch Class with fundamental parameters to create the environment. 
        fetch_env.FetchEnv.__init__(
                                    self, 
                                    MODEL_XML_PATH, 
                                    has_object          = True, # how does this option affect the construction of the environment. Is it: has object all the time, or has object in the hand 50% of the time?
                                    block_gripper       = False, 
                                    n_substeps          = 20,
                                    gripper_extra_height= 0.2, 
                                    target_in_the_air   = True, 
                                    target_offset       = 0.0,
                                    obj_range           = 0.15, 
                                    target_range        = 0.15, 
                                    distance_threshold  = 0.05,
                                    initial_qpos        = initial_qpos, 
                                    reward_type         = reward_type, 
                                    obs_type            = obs_type, 
                                    render_size         = render_size)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, reward_type, obs_type, render_size, stack_only)
        self.render_image_obs = False

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[..., -3:] # Get the grip position only

        block_goals = goal[..., :-3] # Get all the goals EXCEPT the zero'd out grip position

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i + 1) * 3:(i + 2) * 3].shape
        return [
            np.linalg.norm(goal_a[..., i * 3:(i + 1) * 3] - goal_b[..., i * 3:(i + 1) * 3], axis=-1) for i in
            range(self.num_blocks)
        ]

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes reward, perhaps in an off-policy way during training. Doesn't make sense to use any of the simulator state besides that provided in achieved_goal, goal.
        :param achieved_goal:
        :param goal:
        :param info:
        :return:
        """
        subgoal_distances = self.subgoal_distances(achieved_goal, goal)
        if self.reward_type == 'incremental':
            # Using incremental reward for each block in correct position
            reward = -np.sum([(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0) # If your distance is greather than treshhold, reward += -1. Range is [-num_blocks:0]
            reward = np.asarray(reward)

            # Normal success comes with reward of 0. But, if blocks are successfully aligned with goals (i.e. reward == 0) and the gripper is far away from the goals, change reward to 1 with this mask
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal)) # putmask(a, mask, values): sets a to value if mask is true for index
            return reward
        
        elif self.reward_type == 'block1only':
            return -(subgoal_distances[0] > self.distance_threshold).astype(np.float32)
        
        elif self.reward_type == "sparse":
            reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            reward = np.asarray(reward)
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        
        elif self.reward_type == "dense":
            # Dense incremental
            stacked_reward = -np.sum([(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            stacked_reward = np.asarray(stacked_reward)

            reward = stacked_reward.copy()
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))

            if stacked_reward != 0:
                next_block_id = int(self.num_blocks - np.abs(stacked_reward))
                assert 0 <= next_block_id < self.num_blocks
                gripper_pos = achieved_goal[..., -3:]
                block_goals = goal[..., :-3]
                reward -= .01 * np.linalg.norm(gripper_pos - block_goals[next_block_id*3: (next_block_id+1)*3])
            return reward
        
        else:
            raise ("Reward not defined!")

    def _get_obs(self): 
        '''
        1. Compute observations [grip_xyz, f1x, f2x, grip_vxyz, f1v, f2v, obj1_pos, rel_pos_wrt_gripp_obj, obj1_theta, obj1_vxyz, obj1_dtheta obj2... ]
        2. Achieved goal: [obj1_pos obj2_pos ... objn_pos grip_xyz]
        3. Desired goal: goal obtained in ._sample_goal()
        '''

        # Gripper: pos and vel 
        grip_pos    = self.sim.data.get_site_xpos('robot0:grip')
        dt          = self.sim.nsubsteps * self.sim.model.opt.timestep  # dt is equivalent to the number of substeps in controller times duration of substep  
        grip_velp   = self.sim.data.get_site_xvelp('robot0:grip') * dt  # xvelp is the step vel (??) but needs to be scaled by dt such that we get the instantaneous velocity?
        
        # Finger: Extract robot joints and velocities to get finger data
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)          # retrun joint angles and joint velocities for each of the robot's joints including fingers.
        gripper_state = robot_qpos[-2:]       # Extract gripper pos
        gripper_vel   = robot_qvel[-2:] * dt  # Compute gripper vel

        # Obs: basic info is [ee_pos, grip_pos, ee_vel, grip_vel]
        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])
        
        #-------------------------------------------------------------------------- 
        # Achieved Goal + other strucs for obs
        #-------------------------------------------------------------------------- 
        achieved_goal = [] 

        # For n objects
        for i in range(self.num_blocks): 

            # Get xyz pos ob ith obj
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i]) 

            # Orientation: RPY
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            
            # Linear and Angular Velocity
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt # Will get overwritten by robot's velocity
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            
            # Relative position difference with robot gripper (no relative orientation difference)
            object_i_rel_pos = object_i_pos - grip_pos

            # Overwrite with robots linear velocity
            object_i_velp -= grip_velp

            # <<Augmente observations>>
            obs = np.concatenate([
                obs,                                # Robot
                object_i_pos.ravel(),               # Stack objs: pos
                object_i_rel_pos.ravel(),           # delta x, delta y, delta z (wrt gripper)
                object_i_rot.ravel(),               # orientation: rpy
                object_i_velp.ravel(),              # vx, vy, vz
                object_i_velr.ravel()               # dr dp dy
            ])

            # 01 Achieved goal takes as many object positions as there are objects
            achieved_goal = np.concatenate([
                achieved_goal, object_i_pos.copy()
            ])

        # Finally, append the robot's grip xyz 
        # Achieved goal: why not differentiate between object in hand or not like original fetch?
        achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])
        achieved_goal = np.squeeze(achieved_goal)

        # Returns obs, ag, and also desired goals
        return_dict = {
            'observation':   obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal':  self.goal.copy(),
        }

        ## Images
        # if self.obs_type == 'dictimage':
        if hasattr(self, "render_image_obs") and self.render_image_obs:
            return_dict['image_observation'] = self.render(mode='rgb_array')

        return return_dict

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        for i in range(self.num_blocks):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.goal[i * 3:(i + 1) * 3] - sites_offset[i]

        self.sim.forward()

    def _reset_sim(self):
        '''
        1. extract the fixed initial state of the robot 
        2. Objects: computed a des random pos with uniform noise in [-01.5,0.15] and add to ee pos 
        3. Get actual object pos via get_joint_qpos and then update xy with des pos 
        4. set a fixed hieght 
        5. Set in simulation 
        6. step sim forward
        '''
        assert self.num_blocks <= 17 # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize start position of objects.
        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2) # Rojas: why set to initial gripper pos? does robot gripper go there?

            while not ((np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1) and np.all([np.linalg.norm(object_xypos - other_xpos) >= 0.06 for other_xpos in prev_obj_xpos])): # goal pos must be at least 10cm away from gripper pos and must be at least 6cm away from previous pos
                object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)


            prev_obj_xpos.append(object_xypos)

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")     # Object's pose|twist
            assert object_qpos.shape == (7,)

            object_qpos[:2] = object_xypos          # xy
            object_qpos[2] = self.height_offset     # z

            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()                      # Take simulation step and perform all calculations

        return True

    def _sample_goal(self): # Adapts according to case. SingleTower
        cases = ["Singletower", "Pyramid", "Multitower"]
        if self.case == "All":
            case_id = np.random.randint(0, len(cases))
            case = cases[case_id]
        elif self.case in cases:
            case = self.case
        else:
            raise NotImplementedError

        goals = []

        if case == "Singletower": 
            # Add random noise to xyz of init_grip_xpos. Then for z, 50% add rand noize from [0,0.45]
          
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3) 
            goal_object0 += self.target_offset
            goal_object0[2] = self.height_offset                        # Fix goal height unless place object in air below
            
            # HER Strategy: sample object in air half the time
            # If we're only stacking, do not allow the block0 to be in the air
            if self.target_in_the_air and self.np_random.uniform() < 0.5 and not self.stack_only:                 
                goal_object0[2] += self.np_random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)

            # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
            previous_xys = [goal_object0[:2]]
            current_tower_heights = [goal_object0[2]]

            num_configured_blocks = self.num_blocks - 1
            # Set goal for other objects: objecti_xy
            for i in range(num_configured_blocks):
                if hasattr(self, "stack_only") and self.stack_only:
                    # If stack only, use the object0 position as a base
                    goal_objecti = goal_object0[:2]
                    objecti_xy = goal_objecti
                else:
                    objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range, size=2)
                    # Keep rolling if the other block xys are too close to the green block xy
                    # This is because the green block is sometimes lifted into the air
                    while np.linalg.norm(objecti_xy - goal_object0[0:2]) < .071:
                        objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
                    goal_objecti = objecti_xy

                # Check if any of current block xy matches any previous xy's
                for _ in range(len(previous_xys)):
                    previous_xy = previous_xys[_]
                    if np.linalg.norm(previous_xy - objecti_xy) < .071:
                        goal_objecti = previous_xy

                        new_height_offset = current_tower_heights[_] + .05
                        current_tower_heights[_] = new_height_offset
                        goal_objecti = np.append(goal_objecti, new_height_offset)

                # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
                if len(goal_objecti) == 2:
                    goal_objecti = np.append(goal_objecti, self.height_offset)
                    previous_xys.append(objecti_xy)
                    current_tower_heights.append(self.height_offset)

                goals.append(goal_objecti)
        elif case == "Pyramid":
            def skew(x):
                return np.array([[0, -x[2], x[1]],
                                 [x[2], 0, -x[0]],
                                 [-x[1], x[0], 0]])

            def rot_matrix(A, B):
                """
                Rotate A onto B
                :param A:
                :param B:
                :return:
                """
                A = A/np.linalg.norm(A)
                B = B/np.linalg.norm(B)
                v = np.cross(A, B)
                s = np.linalg.norm(v)
                c = np.dot(A, B)

                R = np.identity(3) + skew(v) + np.dot(skew(v), skew(v)) * ((1 - c) / s ** 2)
                return R

            def get_xs_zs(start_point):
                start_point[2] = self.height_offset
                xs = [0, 1, .5]
                zs = [0, 0, 1]
                diagonal_start = 2
                x_bonus = 0
                z = 0
                while len(xs) <= self.num_blocks:
                    next_x = diagonal_start + x_bonus
                    xs.append(next_x)
                    zs.append(z)
                    x_bonus -= .5
                    z += 1
                    if x_bonus < -.5 * diagonal_start:
                        diagonal_start += 1
                        x_bonus = 0
                        z = 0
                return xs, zs
            x_scaling = .06
            start_point = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                              size=3)
            xs, zs = get_xs_zs(start_point) # Just temporary
            # xs is actually ys, because we rotate

            attempt_count = 0
            while start_point[1] + max(xs)*x_scaling > self.initial_gripper_xpos[1] + self.target_range:
                start_point = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,
                                                                                         self.target_range,
                                                                                         size=3)
                if attempt_count > 10:
                    start_point[1] = self.initial_gripper_xpos[1] - self.target_range

                xs, zs = get_xs_zs(start_point)  # Just temporary
                attempt_count += 1

            for i in range(self.num_blocks):
                new_goal = start_point.copy()
                new_goal[0] += xs[i]*x_scaling
                new_goal[2] += zs[i]*.05

                if i > 0:
                    target_dir_vec = np.zeros(3)
                    target_dir_vec[:2] = self.initial_gripper_xpos[:2]-goals[0][:2]

                    target_dir_vec = np.array([0, 1, 0])

                    new_goal_vec = np.zeros(3)
                    new_goal_vec[:2] = new_goal[:2] - goals[0][:2]

                    new_goal = rot_matrix(new_goal_vec, target_dir_vec)@(new_goal - goals[0]) + goals[0]

                goals.append(new_goal)
        elif case == "Multitower":
            if self.num_blocks < 3:
                num_towers = 1
            elif 3 <= self.num_blocks <= 5:
                num_towers = 2
            else:
                num_towers = 3
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                while not np.all(
                    [np.linalg.norm(base_xy - other_xpos) >= 0.06 for other_xpos in tower_bases]):
                    base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range, size=2)

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_%num_towers][:2]
                goal_objecti[2] = tower_heights[_%num_towers] + .05
                tower_heights[_%num_towers] = tower_heights[_%num_towers] + .05
                goals.append(goal_objecti.copy())
        elif case == "Togethertower":
            num_towers = 2
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                if i == 0:
                    base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                else:
                    base_xy = tower_bases[-1].copy()
                    base_xy[1] += .05

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_%num_towers][:2]
                goal_objecti[2] = tower_heights[_%num_towers] + .05
                tower_heights[_%num_towers] = tower_heights[_%num_towers] + .05
                goals.append(goal_objecti.copy())
        elif case == "PickAndPlace":
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,
                                                                                  self.target_range,
                                                                                  size=3)
            goal_object0[2] = self.height_offset

            if np.random.uniform() < 0.5:
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += self.np_random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)
            for i in range(self.num_blocks - 1):
                objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                while not np.all([np.linalg.norm(objecti_xy - goal[:2]) > .071 for goal in goals]):
                    objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range,
                                                                                        size=2)
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = objecti_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti)
        else:
            raise NotImplementedError

        goals.append([0.0, 0.0, 0.0]) # Orientation in Euler angles fixed to 0. Concat to goals before returning. TODO: for multiple blocks should this be added to each block? right now only added at very end....
        return np.concatenate(goals, axis=0).copy()

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        if np.sum([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances]) == 0:
            return True
        else:
            return False

    def _set_action(self, action):
        assert action.shape == (4,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl]) # extend command for both fingers
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action) # Updates gripper via sim.data.ctrl
        utils.mocap_set_action(self.sim, action)

    def step(self, action): 
        '''
        1) Take clipped actions, 
        2) Set action values in appropriate data strucs for gripper and ee, 
        3) Step simulation forward, 
        4) Retrieve observations, done, rewards
        '''
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action) # Pass action to robot (gripper and ee)
        try:
            self.sim.step() # Advance simulation
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(F"action {action}")
        self._step_callback() # Used when gripper set to be blocked/closed. 
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            reward = self.compute_reward_image()
            if reward < .05:
                info = {
                    'is_success': True,
                }
            else:
                info = {
                    'is_success': False,
                }
        elif "state" in self.obs_type:
            info = {
                'is_success': self._is_success(obs['achieved_goal'], self.goal),
            }
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            raise ("Obs_type not recognized")
        return obs, reward, done, info