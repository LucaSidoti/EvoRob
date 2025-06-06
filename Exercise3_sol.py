import os
import gymnasium as gym
import numpy as np
from src.EA.CMAES_sol import CMAES_sol, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World
import imageio
import xml.etree.ElementTree as xml

ENV_NAME = 'Ant_custom'
ROOT_DIR = get_project_root()


class AntMLPWorld(World):
    def __init__(self):
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        # Write a static AntRobot.xml include to AntEnv.xml
        world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world_xml.getroot()
        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_str = xml.tostring(robot_env, encoding='unicode')
        with open(self.world_file, "w") as f:
            f.write(world_str)

        self.env = gym.make(ENV_NAME, robot_path=self.world_file)
        action_space = self.env.action_space.shape[0]
        state_space = self.env.observation_space.shape[0]
        self.controller = MLP.NNController(state_space, action_space)
        self.dt = self.env.get_wrapper_attr('dt')
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        trial_time = 50  # seconds in simulation
        n_sim_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)

        rewards_list = []
        observations, info = self.env.reset()
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
        return np.sum(rewards_list), None


def run_EA_single(ea_single, world):
    for gen in range(ea_single.n_gen):
        print(f"Generation {gen}")
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name='AntMLP_best.mp4'):
    env = gym.make(ENV_NAME, robot_path=world.world_file, render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for _ in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, reward, terminated, truncated, info = env.step(action)
        rewards_list.append(reward)
        if terminated:
            break
    imageio.mimsave(video_name, frames, fps=30)
    env.close()
    print(f"Total reward: {np.sum(rewards_list)}")


def main():
    world = AntMLPWorld()
    n_parameters = world.n_params

    # EA settings
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 15
    CMAES_opts["num_generations"] = 50
    CMAES_opts["mutation_sigma"] = 0.2

    population_size = 30
    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'MLP_only')

    ea = CMAES_sol(population_size, n_parameters, CMAES_opts, results_dir)
    run_EA_single(ea, world)

    # Load and visualize best individual
    best_individual_path = os.path.join(results_dir, "99", "x_best.npy")
    if os.path.exists(best_individual_path):
        best_individual = np.load(best_individual_path)
        world.controller.geno2pheno(best_individual)
        generate_best_individual_video(world)
    else:
        print(f"Best individual not found at: {best_individual_path}")


if __name__ == '__main__':
    main()
