import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

'''
__init__.py will be called when gym.make(id) is called. This top-level function accesses the registration.py module
which defines a class by the given the entry_point that we register here. 
Then the class is insantiated by passing kwargs into it. 
'''

for num_blocks in range(1, 25):
    for reward_type in ['sparse', 'dense', 'incremental', 'block1only']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for render_size in [42, 84]:
                for stack_only in [True, False]:
                    for case in ["Singletower", "Pyramid", "Multitower", "All"]:

                        # Set the position of the robot base
                        initial_qpos = {
                            'robot0:slide0': 0.405, # slide joints for the base of the robot (xyz position wrt world)
                            'robot0:slide1': 0.48,
                            'robot0:slide2': 0.0,
                            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.], # base pos,orientation of obj wrt world
                        }

                        # Set the position of objects, where each new object is shifted up by 6cm
                        for i in range(num_blocks):
                            initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.] # pos for additional objects. height changes by 6cm. each block is 5cm 

                        kwargs = {
                            'reward_type':  reward_type,
                            'initial_qpos': initial_qpos,
                            'num_blocks':   num_blocks,
                            'obs_type':     obs_type,
                            'render_size':  render_size,
                            'stack_only':   stack_only,
                            'case':         case
                        }

                        register(
                            id='FetchBlockConstruction_{}Blocks_{}Reward_{}Obs_{}Rendersize_{}Stackonly_{}Case-v1'.format(*[kwarg.title() 
                                if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, reward_type, obs_type, render_size, stack_only, case]]),
                            entry_point='fetch_block_construction.envs.robotics:FetchBlockConstructionEnv',
                            kwargs=kwargs,
                            max_episode_steps=50 * num_blocks,
                        )