# rl_obstacle_avoidance
Simple obstacle avoidance domain for testing RL algorithm convergence on control problems with stochastics and multiple minima, which are common for real-world autonomous robots.  

From our paper:
> Andersson, Olov and Patrick Doherty. "Deep RL for Autonomous Robots: Limitations and Safety Challenges." Procceedings of the European Symposium on Neural Networks (ESANN). 2019.

An earlier version was also presented at the ICML'18 Workshop on Reproducible Machine Learning.

## Usage Example with OpenAI Gym: ##

    # Prereqs:
    # Import from nth_order_integrator.py (due to parameterization we dynamically load the Gym environment)
    # Assumes relevant env parameters are in the arg parser
    
    # Then create the paramerized environment and register an env_id for Gym
    if args.env == 'NthOrderIntegratorObstaclesParam':  # Obstacle avoidance env w/ parameters from arg parser
        env_id = nth_order_integrator.register_parametric(order=args.env_order, dim=args.env_dim, 
        constraint_a=args.env_constr_a, dynamics_std=args.env_dynamics_std, max_steps=args.env_steps, 
        num_obst=args.env_obst, scenario=args.env_scenario, agent_start=args.env_agent_start, 
        render=args.env_render, verbose=args.env_verbose)

    set_global_seeds(seed)
    def make_env():
        env = gym.make(env_id) 
        env.seed(seed)  # Seed the environment also
        env = bench.Monitor(env, logger.get_dir())  # NOTE: Stores raw episode rewards and lengths
        return env

    if args.nparallel > 0:
        print('Using sub-process vector env.')
        env = SubprocVecEnv([make_env(i) for i in range(args.npar)])
    else:
         print('Using dummy vector env.')
         env = DummyVecEnv([make_env])    

Then use your favorite RL algo on env. The results in our paper were from PPO2, slightly modified to use the Mujoco defaults from the PPO paper, including EnvNormalize.
 
The environment can also be conveniently visualized with Rviz on the /gym_deepca topic if ROS (Robot Operating System, www.ros.org) is installed.

See the code for reasonable default parameters. There are also special scenarios (c.f. paper), but the default is just randomly moving obstacles in 2D. The code is pretty self-explanatory, but don't hesitate to contact me if you have any questions.
