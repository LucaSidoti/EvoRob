import numpy as np
import os
import gymnasium as gym
from src.utils.Filesys import get_project_root
from Exercise3_sol import AntWorld, generate_best_individual_video

def load_best_individual(result_folder: str, generation: int = 99):
    filepath = os.path.join(result_folder, str(generation), "x_best.npy")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Best individual not found at: {filepath}")
    return np.load(filepath)

def visualise_individual(world, genotype):
    world.controller.geno2pheno(genotype)

    env = gym.make("Ant_custom", robot_path=world.world_file, render_mode="human")
    observations, info = env.reset()
    for _ in range(1000):
        action = world.controller.get_action(observations)
        observations, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    env.close()

if __name__ == "__main__":
    ENV_NAME = "Ant_custom"
    ROOT_DIR = get_project_root()
    results_dir = os.path.join(ROOT_DIR, "results", ENV_NAME, "multi")
    generation = 2

    best_genotype = load_best_individual(results_dir, generation=generation)

    # Create the Ant world (fixed morphology)
    world = AntWorld()

    # Option 1: visualize in real time
    visualise_individual(world, best_genotype)

    # Option 2: save as video
    # generate_best_individual_video(world, video_name=f"gen_{generation}_best.mp4")