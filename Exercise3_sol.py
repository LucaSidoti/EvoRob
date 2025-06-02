from src.EA.CMAES_sol import CMAES_sol, CMAES_opts
from src.EA.NSGA_sol import NSGAII_sol, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise3 body-brain: This is your first full body+brain evolution by adapting a custom Ant-v5 gym environment. 
    We adjust both leg lengths and controller weights for a locomotion task.   
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'
STRATEGY = 'NSGAII'  # 'CMAES' for multi-objective evolution
radius_cum_sum = 0


class AntWorld(World):
    def __init__(self, ):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 30  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.n_repeats = 3
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params

        self.n_params = self.n_weights # + 8
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           ]

    def geno2pheno(self, genotype):
        control_weights = genotype # [-self.n_weights:]
        # body_params = (genotype[:-self.n_weights] + 1.5) / 5 * 0.5 + 0.1
        # assert len(body_params) == 8
        # assert len(control_weights) == self.n_weights
        # assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        # front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # # Define the 3D coordinates of the relative tree structure
        # front_left_hip_xyz = np.array([0.2, 0.2, 0])
        # front_left_knee_xyz = np.array(
        #     [np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        # front_left_toe_xyz = np.array(
        #     [np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        # front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        # front_right_knee_xyz = np.array(
        #     [-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        # front_right_toe_xyz = np.array(
        #     [-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        # back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        # back_left_knee_xyz = np.array(
        #     [-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        # back_left_toe_xyz = np.array(
        #     [-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        # back_right_hip_xyz = np.array([0.2, -0.2, 0])
        # back_right_knee_xyz = np.array(
        #     [np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        # back_right_toe_xyz = np.array(
        #     [np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        # points = np.vstack([front_left_hip_xyz,
        #                     front_left_knee_xyz,
        #                     front_left_toe_xyz,
        #                     front_right_hip_xyz,
        #                     front_right_knee_xyz,
        #                     front_right_toe_xyz,
        #                     back_left_hip_xyz,
        #                     back_left_knee_xyz,
        #                     back_left_toe_xyz,
        #                     back_right_hip_xyz,
        #                     back_right_knee_xyz,
        #                     back_right_toe_xyz,
        #                     ])

        # # define the type of connections [FIXED ARCHITECTURE]
        # connectivity_mat = np.array(
        #     [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        # )

        points, connectivity_mat = self.get_fixed_morphology()

        return points, connectivity_mat
    
    def get_fixed_morphology(self):
        # Morphologie fixe "standard"
        points = np.array([
            [0.2, 0.2, 0], [0.35, 0.35, 0], [0.45, 0.45, 0],
            [-0.2, 0.2, 0], [-0.35, 0.35, 0], [-0.45, 0.45, 0],
            [-0.2, -0.2, 0], [-0.35, -0.35, 0], [-0.45, -0.45, 0],
            [0.2, -0.2, 0], [0.35, -0.35, 0], [0.45, -0.45, 0]
        ])
        connectivity_mat = np.array([
            [150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        return points, connectivity_mat

    def evaluate_individual(self, genotype):
        global radius_cum_sum
        points, connectivity_mat = self.geno2pheno(genotype)

        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()

        # % Defining the Robot environment in MuJoCo
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')

        with open(self.world_file, "w") as f:
            f.write(world_xml)

        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))
        multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))  # TODO

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)
    
        # initial_positions = np.array([[info_["x_position"], info_["y_position"]] for info_ in info])
        initial_positions = np.stack((info["x_position"], info["y_position"]), axis=-1)
        previous_positions = initial_positions.copy()

        # print("initial_positions:", initial_positions)

        previous_angles = np.arctan2(initial_positions[:, 1], initial_positions[:, 0])

        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, _, dones, truncated, infos = envs.step(actions)

            # observations = np.concatenate(observation, )

            x_pos = infos["x_position"]
            y_pos = infos["y_position"]

            current_positions = np.stack([x_pos, y_pos], axis=1)

            # Angle wrt the center
            current_angles = np.arctan2(y_pos, x_pos)
            delta_angles = current_angles - previous_angles

            # Correction for -pi/+pi transition
            delta_angles = (delta_angles + np.pi) % (2 * np.pi) - np.pi

            # Indirect rotations
            angular_reward = np.where(delta_angles > 0, delta_angles, 0)

            # Tangential velocity
            displacement = np.linalg.norm(current_positions - previous_positions, axis=1)
            tangential_reward = displacement

            # Penalization for frontier proximity (rayon max ~1.2 par exemple)
            radius = np.linalg.norm(current_positions, axis=1)
            upper_radius_too_far = 2
            lower_radius_too_far = 1.0
            penalty_too_far = np.where((radius > upper_radius_too_far), -(radius - upper_radius_too_far)**2, 0.0)
            penalty_too_close = np.where((radius < lower_radius_too_far), -(radius - lower_radius_too_far)**2, 0.0)
            combined_offtrack_penalty = penalty_too_far + penalty_too_close

            # Introduce cumulative sum of radius errors
            goal_radius = 1.5
            radius_cum_sum += (radius - goal_radius)**2

            
            # position_step = 0.25
            exp_traj_vec=np.array([-radius*np.cos(current_angles),radius*np.sin(current_angles)])
            # goal_point = current_positions + exp_traj_vec.T*position_step

            current_spd_vec = np.stack([infos["x_velocity"], infos["y_velocity"]], axis=0)
            

            # MONO-OBJECTIF GLOBAL
            # combined_reward = -20*(radius - goal_radius)**2 + 20*(np.linalg.norm(current_positions - goal_point, axis=1))**2 + combined_offtrack_penalty

            # print("-20*(radius - goal_radius)**2:", -20*(radius - goal_radius)**2)
            # print("0*np.linalg.norm(current_spd_vec, axis=0):", 0*np.linalg.norm(current_spd_vec, axis=0))
            # print("20*(np.sum(current_spd_vec * exp_traj_vec, axis=0)):", 20*(np.sum(current_spd_vec * exp_traj_vec, axis=0)))
            # print("- 5e-7*(radius_cum_sum)**4:", - 5e-7*(radius_cum_sum)**4)
            
            combined_reward = - 5e-8*(radius_cum_sum)**4 #+ 0*np.linalg.norm(current_spd_vec, axis=0) + 200*(np.sum(current_spd_vec * exp_traj_vec, axis=0)) + combined_offtrack_penalty
            # print(combined_reward)

            rewards_full[step, ~done_mask] = combined_reward[~done_mask]


            circle_deviation = (radius - 1.0) ** 2  # Distance quadratique au rayon 1   ####### TODO

            # MULTI-OBJECTIFS POUR NSGA-II
            # For visualisation
            # multi_obj_reward = np.array([angular_reward, tangential_reward]).T        ####### INITAL 
            multi_obj_reward = np.array([tangential_reward, -5.0 * circle_deviation]).T
            multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

            # Update
            previous_positions = current_positions
            previous_angles = current_angles
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        radius_cum_sum = 0
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


def run_EA_single(ea_single, world):
    for gen in range(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen)


def run_EA_multi(ea_multi, world):
    for gen in range(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen)
        


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def visualise_individual(genotype):
    world = AntWorld()
    points, connectivity_mat = world.geno2pheno(genotype)
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(1000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    print(np.sum(rewards_list))


def load_best_individual(result_folder: str, generation: int = 99):
    filepath = os.path.join(result_folder, str(generation), "x_best.npy")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Best individual not found at: {filepath}")
    return np.load(filepath)


def main():
    # %% Understanding the world
    state_size = 30
    action_size = 8
    genotype_size = (state_size * state_size) + (state_size * action_size)
    genotype = np.random.uniform(-1, 1, genotype_size)  # 8 body parameters, 945 NN weights
    visualise_individual(genotype)

    # %% Optimise single-objective
    if STRATEGY == 'CMAES':
        world = AntWorld()
        n_parameters = world.n_params

        population_size = 250
        CMAES_opts["min"] = -1
        CMAES_opts["max"] = 1
        CMAES_opts["num_parents"] = population_size
        CMAES_opts["num_generations"] = 100
        CMAES_opts["mutation_sigma"] = 0.33

        results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
        ea_single = CMAES_sol(population_size, n_parameters, CMAES_opts, results_dir)

        run_EA_single(ea_single, world)

    # %% Optimise multi-objective
    # TODO implement the NSGAII
    if STRATEGY == 'NSGAII':
        world = AntWorld()
        n_parameters = world.n_params

        population_size = 150 #250
        NSGA_opts["min"] = -1
        NSGA_opts["max"] = 1
        NSGA_opts["num_parents"] = population_size
        NSGA_opts["num_generations"] = 100
        NSGA_opts["mutation_prob"] = 0.3
        NSGA_opts["crossover_prob"] = 0.65

        results_dir = os.path.join(ROOT_DIR, "results", "Ant_custom", "multi")

        # best_genotype = load_best_individual(results_dir, generation=generation)
        
        ea_multi_obj = NSGAII_sol(population_size, n_parameters, NSGA_opts, results_dir)

        # print("Best genotype from generation", generation, ":")
        # visualise_individual(best_genotype)

        print("Starting multi-objective evolution...")
        run_EA_multi(ea_multi_obj, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.

    # if ea_multi_obj.current_gen % 5 == 0:
    #     gen_dir = os.path.join(results_dir, str(ea_multi_obj.current_gen))
    #     x_best_path = os.path.join(gen_dir, "x_best.npy")
    #     if os.path.exists(x_best_path):
    #         best_individual = np.load(x_best_path)
    #         video_path = os.path.join(gen_dir, "best_individual.mp4")
    #         world.evaluate_individual(best_individual, make_video=True, video_name=video_path)
    
    best_individual = np.load(os.path.join(results_dir, ea_multi_obj.n_gen, "x_best.npy"))

    points, connectivity_mat = world.geno2pheno(best_individual)
    
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    
    robot.xml = robot.define_robot()

    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    print("Expected to generate a video of the best individual.")
    generate_best_individual_video(world)


if __name__ == "__main__":
    main()

