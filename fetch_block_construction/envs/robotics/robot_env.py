import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, render_size):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        # print("FULL path")
        # print(fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)

        # Turn off mocap
        for body_idx1, val in enumerate(self.sim.model.body_mocapid):
            if val != -1:
                for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
                    if body_idx1 == body_idx2:
                        # Store transparency for later to show it.
                        self.sim.extras[
                            geom_idx] = self.sim.model.geom_rgba[geom_idx, 3]
                        self.sim.model.geom_rgba[geom_idx, 3] = 0

        self.viewer = None

        # TODO: Is this for BC?
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        # Set the Initial State
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        # Get Goals for 1 or multiple objects
        self.goal = self._sample_goal()

        # Get observations for one or multiple objects
        obs = self._get_obs()

        # Determine the action space
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        
        
        # Determine the observation space
        if "dict" in self.obs_type:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        elif self.obs_type == "np":
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32'),
        else:
            raise("Obs_type not recognized")
        self.render_size = render_size

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps # dt for the RL loop is the total amount of time taken across substeps in the control loop

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        1. Clips actions within the min/max limits
        2. Set them to the robot and fingers
        3. advance the simulation and do forward computations

        5. Get new observations s', r, done, info
            - If image: TODO study these computations
        '''
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            reward = self.compute_reward_image() # 
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
            raise("Obs_type not recognized")
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()   # reset simulation.
        self.goal = self._sample_goal().copy()  # add noise to init pos. varies according to strategy. TODO: currently appending orientation at the end once regardless # of objs
        obs = self._get_obs()                   # [observations, achieved goal, and desired goal]
        return obs

    def close(self):
        if self.viewer is not None:
            #self.viewer.finish()
            self.viewer = None

    def render(self, mode='human', size=None):
        self._render_callback() # Updates target sites and calls sim.forward()
        if mode == 'rgb_array':
            size=800
            # window size used for old mujoco-py:
            if size:
                data = self.sim.render(size, size, camera_name="external_camera_0")
            else:
                data = self.sim.render(self.render_size, self.render_size, camera_name="external_camera_0")
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup() # Sets the viewer to point to the robot location from some elevation/azimuth/angle
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
