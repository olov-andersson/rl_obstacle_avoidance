# -*- coding: utf-8 -*-
"""
Nth degree integrator test environment, optionally with avoidance of randomly moving obstacles.
@author: Olov Andersson
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import math
import time
import sys
import gym
import numpy as np
from scipy.special import expit 

from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

## USAGE EXAMPLE WITH OPENAI GYM: ##
#
#    # Generate custom paramerized env_id
#    if args.env == 'NthOrderIntegratorObstaclesParam':  # Parameteric custom env
#        env_id = my_envs.nth_order_integrator.register_parametric(order=args.env_order, dim=args.env_dim, constraint_a=args.env_constr_a, dynamics_std=args.env_dynamics_std, max_steps=args.env_steps, #num_obst=args.env_obst, scenario=args.env_scenario, agent_start=args.env_agent_start, render=args.env_render, verbose=args.env_verbose)
#
#    set_global_seeds(seed)
#    def make_env(rank=0):
#        env = gym.make(env_id) 
#        env.seed(seed+rank)  # Want to seed the environment too...
#        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))  # NOTE: Stores raw episode rewards and lengths
#        return env
#
#    if args.nparallel > 0:
#        print('Using sub-process vector env.')
#        env = SubprocVecEnv([make_env(i) for i in range(args.npar)])
#    else:
#         print('Using dummy vector env.')
#         env = DummyVecEnv([make_env])    
#
# Then use your favorite RL algo on env. Results in our paper were from PPO2, slightly modified to use the Mujoco defaults from the PPO paper, including EnvNormalize.

# Optionally, the environment can be visualized with Rviz on the /gym_deepca topic if ROS is installed.

# Parameterized environment factory method, circumvents OpenAI Gym's hardcoded environment constructors to facillitate testing variations
def register_parametric(order=1, dim=2, num_obst=3, constraint_a=None, dynamics_std=0., max_steps=128, scenario="default", agent_start="default", render=False, verbose=False):
    stem = 'NthOrderIntegratorObstaclesParam_o{}d{}n{}'.format(order, dim, num_obst)
    entry_point = lambda : NthOrderIntegratorObstacles(order=order, dim=dim, dynamics_std=dynamics_std, num_obst=num_obst, scenario=scenario, agent_start=agent_start, constraint_a=constraint_a, render=render, verbose=verbose)

    # Use regular gym API code path for compatibility
    id = '{}-v0'.format(stem)
    print('Registering custom env_id:', id)
    register(id=id, entry_point=entry_point, max_episode_steps=max_steps)
    return id


# NOTE: Environment needs explicit calls to seed() and reset() before step, where the latter can be implicitly handled by certain gym env wrappers 
# The environment will draw a need seed for each episde, based on the initial one. This way both the sequence is replicable, as well as individual 
# episodes if quirks are detected
class NthOrderIntegrator(gym.Env):

    def __init__(self, order=1, dim=1, dynamics_std=0., agent_start="default", render=False, verbose=False):
        self.order = order
        self.dim = dim
        print('Order', self.order, 'Dim', self.dim, 'integrator.')
        if self.order > 0:
            self.nstate = self.order 
        else:
            self.nstate = 1  # For API reasons we always need a state, just set it equal to action
        self.dt = 1/10.0
        self.dynamics_std = dynamics_std
        
        # Custom agent scenarios
        if agent_start == "start_left":
            self.start_pos = np.zeros((self.nstate, self.dim))
            self.start_pos[0,:] = 1.5  # Fixed start pos
            self.random_mag = 0.1  # Small variation
        elif agent_start == "start_top":
            self.start_pos = np.zeros((self.nstate, self.dim))
            self.start_pos[0,1] = 3.  # Fixed start pos
            self.random_mag = 0.  # No variation
        else:  # Default
            agent_start = "default"
            self.start_pos = np.zeros((self.nstate, self.dim))  # Zero-centered start state
            self.random_mag = 0.1*np.ones((self.nstate, self.dim))  # Small variation on dynamic state
            self.random_mag[0,:] = 2.  # Large variation on start pos 
        print("Agent scenario: ", agent_start)
        
        state_max = np.inf*np.ones(self.nstate*self.dim)  
        state_min = -np.inf*np.ones(self.nstate*self.dim)  
        action_max = np.inf*np.ones(self.dim)  
        action_min = -np.inf*np.ones(self.dim)  

        self.action_space = spaces.Box(action_min, action_max)
        self.observation_space = spaces.Box(state_min, state_max)
        print('Obs space', self.observation_space, 'Action space', self.action_space)

        self.state_packed = np.zeros((self.nstate, self.dim))
        self.state = self.state_packed.reshape(-1, order='F')

        self.verbose = verbose
        # Set up optional ROS import for visualization. NOTE: This requires ROS to be installed.
        if render:
            global rospy, Point, Pose, PoseStamped, PoseArray, Quaternion, Marker, ColorRGBA
            import rospy
            from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion
            from visualization_msgs.msg import Marker
            from std_msgs.msg import ColorRGBA 
            
            # Publish visualization messages to external Rviz viewer
            rospy.init_node('gym_deepca', disable_signals=True)
            self.rviz_pub = rospy.Publisher("/gym_deepca", Marker, queue_size=100)

        self.step_counter = 0
        self.last_action = 0.  # Used for visualization

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._curr_seed = seed
        return [seed]

    def _step(self, action):
        assert(action.shape == self.action_space.shape)
        self.last_action = action
        action = action.astype('float64')
        self.step_counter += 1

        # Nth order integrator, using simple Euler
        step_size = self.dt
        if self.order > 1:
            self.state_packed[0:-1,:] +=  step_size*self.state_packed[1:,:]
        
        if self.order > 0:
            self.state_packed[-1,:] += step_size*action
        else:
            self.state_packed[-1,:] = action  # 0th order is handled by setting state equal to action (gym API requires a state)

        if self.dynamics_std > 0.:  # Non-deterministic dynamics
            self.state_packed[-1,:] += self.np_random.normal(0, self.dynamics_std, size=self.state_packed[-1,:].size)  

        reward = -np.linalg.norm(self.state_packed[0,:])**2  # Big penalty on positions
        reward += -0.01*np.linalg.norm(action)**2  # Small square penalty on actions
        
        reward /= self.dim  # Normalize rewards 
        self.state = self.state_packed.reshape(-1, order='F')  
        
        return self.state.copy(), reward.copy(), False, {}  # False implies a continuing task, True would end the episode prematurely
        
    def _reset(self):
        print("Final state:", self.state, "Final dist:", np.linalg.norm(self.state_packed[0,:]), flush=True)                  
        if self.step_counter > 0:  # Not first run, generate new seed so we can repeat episode
            new_seed = self.np_random.randint(10**9)
            self._seed(new_seed)
        
        init_mag = self.random_mag * np.ones((self.nstate, self.dim))
        self.state_packed = self.start_pos + self.np_random.uniform(low=-init_mag, high=init_mag)  # Randomize start pos
        self.state = self.state_packed.reshape(-1, order='F')  # TODO: Not needed due to being view?
        print("########## Env reset! ############# seed=", self._curr_seed, 'init state=', self.state, flush=True)
        return self.state.copy()

    # Optionally render, using ROS and rviz.
    def _render(self, mode='human', close=False):

        # Publish the agent and goal
        draw_pos_vel(self.rviz_pub, self.state_packed[0,:], self.last_action, self.radius, agent=True, idx=0)
        draw_goal(self.rviz_pub, np.zeros((3)), self.radius)
        return None

# Composition of simple integrator agent and randomly moving obstacles (spheres)
class NthOrderIntegratorObstacles(NthOrderIntegrator):

    def __init__(self, order=1, dim=2, constraint_a=2.0, dynamics_std=0., num_obst=3, obst_vel_state=True, scenario="default", agent_start="default", scale=3., render=False, verbose=False):

        start_pos = None
        # Custom scenario parameters
        if scenario == "static_local_minima":  # Scenario: Static local minima test with three obstacles in triangle. Uses agent_start="start_top"
            assert(dim == 2)
            assert(num_obst == 3)
            start_pos = np.array([[0., 1.], [-0.9, 1.9], [0.9, 1.9]])
            obst_speed = 0.
            static_init = True
            obst_vel_state = False
            agent_start = "start_top"
        elif scenario == "static_1d_origo":  # Scenario: 1D simple deterministic vs. stochastic convergence speed test. Uses agent_start="start_left"
            assert(dim == 1)
            assert(num_obst == 1)
            start_pos = 0.  # Static obstacle(s) in origo
            obst_speed = 0.
            static_init = True
            obst_vel_state = False
            agent_start = "start_left"
        elif scenario == "static_ep_random":
            obst_speed = 0.  
            static_init = False
        elif scenario == "static_fixed":        
            obst_speed = 0.
            static_init = True
        elif scenario == "default":  # The default scenario with randomly moving and initialized ostacles
            obst_speed = 1.0
            static_init = False
        elif scenario == "small":
            scale = 2.
            obst_speed = 1.0
            static_init = False
        else:
            raise Exception("UNSUPPORTED ENV SCENARIO")
        print("Obstacle scenario: ", scenario)
        self.scenario = scenario

        super().__init__(order, dim, dynamics_std=dynamics_std, agent_start=agent_start, render=render)

        self.radius = 0.5/2
        obst_radius = 1.0/2

        bounds_max = np.zeros((self.dim,))+scale #+3. #+2.  # For obst pos
        bounds_min = np.zeros((self.dim,))-scale #-3. #-2.  

        print('Populating obstacles, static_init:', static_init, 'speed:', obst_speed)            
        self.obstacles = MovingObstacles(num_obst, obst_radius, dim, bounds_min, bounds_max, start_pos=start_pos, max_speed=obst_speed, static_init=static_init, vel_state=obst_vel_state)

        # Extend state space for obstacles
        state_min = np.concatenate((self.observation_space.low, np.tile(bounds_min, self.obstacles.num*(1+(obst_vel_state>0)))))  # For obst pos and vel
        state_max = np.concatenate((self.observation_space.high, np.tile(bounds_max, self.obstacles.num*(1+(obst_vel_state>0)))))
        self.observation_space = spaces.Box(state_min, state_max) 
        print(num_obst, 'obstacles, augmented state max:', self.observation_space.high)

        self.verbose = verbose

        self.last_collision = np.zeros(self.obstacles.num)
        self.num_collisions = 0
        self.ep_rewards = 0
        self.batch_rewards = 0
        self.log_rewards = 0 
        self.constraint_a = constraint_a 

    def _seed(self, s=None):  # Called from base class
        seed = super()._seed(s)
        self.obstacles.set_rng(self.np_random)  # RNG set by parent seed
        return seed
       
    def _step(self, action):
        if self.constraint_a is not None:
            action = self.constraint_a*(-1+2*expit(action))
        (o,r,e,info) = super()._step(action)
        self.obstacles.update(self.dt)
        aug_r = r
        penetration = self.obstacles.obstacle_penetration(self.state_packed[0,:], self.radius)
        for i in range(self.obstacles.num):
            if penetration[i] > 0:
                if self.scenario == "static_1d_origo":
                    aug_r -= 100.  # 1D toy objective  
                else:
                    aug_r -= 500*(np.asscalar(penetration[i]/self.obstacles.radius[i]))  # WS paper objective
#                    aug_r -= 50+500*(np.asscalar(penetration[i]/self.obstacles.radius[i]))  # This should discourage slight touches, but only made convergence slower...
                 
                print('obst', i, 'penetration: ', penetration[i], 'aug_r', aug_r, flush=True)
                self.last_collision[i] = time.time()
                self.num_collisions += 1
        aug_o = np.concatenate((o, self.obstacles.get_state()))
        self.ep_rewards += aug_r
        self.log_rewards += aug_r

        if self.verbose:    
             print("step=", self.step_counter, "a=", action, " -> s=", aug_o, "r=", aug_r, flush=True)                       
        
        return (aug_o.copy(),aug_r,e,info)

    def _reset(self):
        self.batch_rewards += self.ep_rewards
        if (self.step_counter % 2048) == 0:
            print_batchrew = round(self.batch_rewards/16,3)
            self.batch_rewards = 0
        else:
            print_batchrew = '-'

        if (self.step_counter % (2048*40) == 0):
            print('MOVING AVERAGE REWARDS', round(self.log_rewards/(2048.*40)*128.0,3), 'AT STEP', self.step_counter)
            self.log_rewards = 0
         
        print('PROGRESS Batch mean', print_batchrew,'Episode sum rewards', round(self.ep_rewards,3),'num_collisions:', self.num_collisions, 'at step', self.step_counter, 'seed', self._curr_seed)
        self.num_collisions = 0
        self.ep_rewards = 0
        agent_state = super()._reset()
        self.obstacles.reset(reject_sphere=(self.state_packed[0,:], self.radius+0.5))  # Safety margin on spawn, 0.3 is a bit tight
        aug_o = np.concatenate((agent_state, self.obstacles.get_state()))
        print('RESET aug state=', aug_o, flush=True)        
        return aug_o

    def _render(self, *args, **kwargs):
        if self.state is None:
            return

        # Publish obstacles
        for i in range(self.obstacles.num):
            collided = self.last_collision[i] > time.time()-1.  # agent has touched this obstacle within last second
            draw_pos_vel(self.rviz_pub, self.obstacles.pos[i,:], self.obstacles.vel[i,:], self.obstacles.radius[i], agent=False, collided=collided, idx=1+i)
        
        return super()._render(*args, **kwargs)
       
        
class MovingObstacles:
    def __init__(self, num, radius, dim, bounds_min, bounds_max, start_pos=None, start_vel=None, max_speed=1.5, max_acc=1.0, noise=0.0, start_goals=None, static_init=False, vel_state=True):
        # pos and vel size = (num,3) (x,y,z)
        self.num = num
        self.dim = dim
        self.radius = radius*np.ones((self.num, 1)  )  # Fix sizes by broadcasting
        self.bounds_min=bounds_min*np.ones((self.num, self.dim))
        self.bounds_max=bounds_max*np.ones((self.num, self.dim))
        self.start_pos = None
        if start_pos is not None:
            self.start_pos = start_pos*np.ones((self.num, self.dim))
            assert(np.all(np.less_equal(self.start_pos, self.bounds_max)))
            assert(np.all(np.greater_equal(self.start_pos, self.bounds_min)))
        self.start_vel = None
        if start_vel is not None:
            self.start_vel = start_vel*np.ones((self.num, self.dim))
            assert(np.all(np.less_equal(self.start_vel, self.max_speed)))
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.start_goals = None
        if start_goals is not None:
            self.start_goals = start_goals*np.ones((self.num, self.dim))
        self.static_init = static_init
        self.vel_state = vel_state
        
    def get_state(self):
        if self.vel_state == 0:
            return np.reshape(self.pos,-1)
        elif self.vel_state == 1:
            return np.concatenate((np.reshape(self.pos,-1), np.reshape(self.vel,-1)))
        elif self.vel_state == 2:
            return np.concatenate((np.reshape(self.pos,-1), 0.*np.reshape(self.vel,-1)))  # Zeroed vel state

    def set_rng(self, rng):
        self.np_random = rng

    def reset(self, reject_sphere=None):
        # Randomized or fixed positions at episode start
        if self.start_pos is None:
            rejected = True
            while rejected: 
                self.pos = self.np_random.uniform(low=self.bounds_min, high=self.bounds_max)  
                rejected = reject_sphere is not None and any(self.obstacle_penetration(reject_sphere[0], reject_sphere[1]) > 0) 
            if self.static_init:  # Reuse pos for every episode
                self.start_pos = self.pos
        else:
            self.pos = self.start_pos

        if self.start_vel is None:
            self.vel = np.zeros((self.num, self.dim))    
        else:
            self.vel = self.start_vel

        if self.start_goals is None:
            self.goals = self.np_random.uniform(low=self.bounds_min, high=self.bounds_max)  
        else:
            self.goals = self.start_goals

    def update(self, dt):
        # Vectorized motion update for obstacles
        self.pos = self.pos + self.vel*dt
        # P-controller on vel, clipped by max_acc
        dg = self.goals - self.pos
        goal_dist = np.linalg.norm(dg, axis=1, keepdims=True)
        target_vel = self.max_speed*dg / goal_dist+0.000001
        dvel = target_vel - self.vel
        vel_max_change = dt*self.max_acc
        max_dvel = np.abs(vel_max_change*dvel / (np.linalg.norm(dvel, axis=1, keepdims=True) +0.000001))
        self.vel += np.clip(dvel, -max_dvel, max_dvel)
        # Update goal
        if self.max_speed > 0.0:  # Moving obstacle, update goal
            for i in range(self.num):
                if goal_dist[i] < 0.5:
                    self.goals[i,:] =  self.np_random.uniform(low=self.bounds_min[i,:], high=self.bounds_max[i,:]) 
                    print('New goal', self.goals[i,:])
               
        
    def obstacle_penetration(self, agent_pos, agent_radius):
        agent_dists = np.linalg.norm(self.pos-agent_pos, axis=1, keepdims=True) 
        return (self.radius+agent_radius) - agent_dists


##########################################################################
############### ROS HELPER FUNCTIONS FOR VISUALIZATION ###################
##########################################################################

def draw_pos_vel(rviz_pub, pos, vel, radius, agent=False, collided=False, idx=0):
    m = Marker()
    m.header.frame_id = "/world"
    m.header.stamp = rospy.Time.now()
    m.ns = "obstacle";
    m.type = m.SPHERE;
    m.action = m.ADD;
    if len(pos) == 3:
        m.pose.position = Point(pos[0], pos[1], pos[2])
    elif len(pos) == 2:
        m.pose.position = Point(pos[0], pos[1], 0.)
    else:
        m.pose.position = Point(pos[0], 0., 0.)
        
    # Color coding
    if agent:
        m.color = ColorRGBA(0.0, 1.0, 0, 1)        
    elif collided:
        m.color = ColorRGBA(1.0, 1.0, 0, 1)
    else:
        m.color = ColorRGBA(1.0, 0, 0, 1)
    m.id = 100+idx*100

    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = 0.0;
    m.pose.orientation.w = 1.0;
    m.scale.x = radius*2.0;
    m.scale.y = radius*2.0;
    m.scale.z = radius*2.0;
    rviz_pub.publish(m)
    
    vel_norm = vel.copy()  # / np.linalg.norm(vel+0.001)
    vel_norm.resize((3,))
    m2 = Marker()
    m2.header.frame_id = "/world"    
    m2.header.stamp = rospy.Time.now()
    m2.ns = "obstacle"
    m.action = m.ADD
    m2.type = m.ARROW
    m2.id = m.id + 1
    if (agent):
        m2.color = ColorRGBA(0.0, 0.5, 0.0, 0.6)  # TODO: TEMP
    else:
        m2.color = ColorRGBA(0.5, 0.0, 0.0, 0.6)
    m2.scale = Point(0.15, 0.3, 0.3)
    p = m.pose.position
    p.z = p.z + 0.5  # Arrows hover above object
    m2.points.append(p)
    m2.points.append(Point(p.x+vel_norm[0], p.y+vel_norm[1], p.z+vel_norm[2]))
    m2.pose.position = Point(0., 0., 0.)
    rviz_pub.publish(m2)
            
    return                 

def draw_goal(rviz_pub, pos, radius):
    m = Marker()
    m.header.frame_id = "/world"
    m.header.stamp = rospy.Time.now()
    m.ns = "obstacle";
    m.type = m.CYLINDER;
    m.action = m.ADD;
    if len(pos) == 3:
        m.pose.position = Point(pos[0], pos[1], pos[2])
    elif len(pos) == 2:
        m.pose.position = Point(pos[0], pos[1], 0.)
    else:
        m.pose.position = Point(pos[0], 0., 0.)
        
    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = 0.0;
    m.pose.orientation.w = 1.0;
    m.scale.x = radius*2.0*1.1;
    m.scale.y = radius*2.0*1.1;
    m.scale.z = 0.1;

    m.color = ColorRGBA(0.0, .3, 0.0, 1)        
    m.id = 1
    rviz_pub.publish(m)  # Outer circle
    m.color = ColorRGBA(1.0, 1.0, 1.0, 1)        
    m.id = 2
    m.scale.x = radius*2.0*0.8;
    m.scale.y = radius*2.0*0.8;
    m.scale.z = 0.15;        
    rviz_pub.publish(m)  # Inner

       


