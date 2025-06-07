import numpy as np
import os
import matplotlib.pyplot as plt

# Assure-toi que cette variable pointe vers la base de tes résultats
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Ou définis le chemin manuellement
RESULTS_BASE = os.path.join(ROOT_DIR, 'results', 'Ant_custom')


def plot_pareto_front(result_dir, generation):
    """
    Affiche le front de Pareto pour une génération donnée.
    Utilise le fichier 'f.npy' qui contient les fitness des parents d'élite.
    """
    plt.figure(figsize=(8, 6))
    
    # Chemin vers le fichier de fitness de la génération spécifiée
    fitness_file = os.path.join(result_dir, str(generation), "f.npy")
    
    if not os.path.exists(fitness_file):
        print(f"Fichier manquant pour la génération {generation}: {fitness_file}")
        plt.close()
        return

    # Charger les fitness (shape: [n_parents, 2])
    fitnesses = np.load(fitness_file)
    
    # Le premier objectif est sur l'axe X, le second sur l'axe Y
    obj1 = fitnesses[:, 0]
    obj2 = fitnesses[:, 1]
    
    plt.scatter(obj1, obj2, c='blue', alpha=0.7, label=f"Front (Gen {generation})")
    
    # Améliorer le graphique
    plt.title(f"Pareto front at generation {generation}")
    plt.xlabel("Obj1 : Circular path fitting")
    plt.ylabel("Obj2 : Circular direction")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder la figure
    save_path = os.path.join(result_dir, f"pareto_front_gen_{generation}.pdf")
    plt.savefig(save_path)
    print(f"saved : {save_path}")
    plt.close()


def plot_objectives_over_generations(result_dir):
    """
    Affiche l'évolution moyenne de chaque objectif au fil des générations.
    Utilise le fichier 'full_f.npy'.
    """
    plt.figure(figsize=(10, 5))
    
    full_f_path = os.path.join(result_dir, 'full_f.npy')
    if not os.path.exists(full_f_path):
        print(f"missing file : {full_f_path}")
        plt.close()
        return

    # Charger toutes les fitness (shape: [n_gen, n_pop, 2])
    all_fitnesses = np.load(full_f_path)
    n_gen = all_fitnesses.shape[0]
    gens = np.arange(n_gen)

    # --- Objectif 1 ---
    obj1_mean = np.mean(all_fitnesses[:, :, 0], axis=1)
    obj1_std = np.std(all_fitnesses[:, :, 0], axis=1)
    plt.plot(gens, obj1_mean, label='Obj1 mean', color='dodgerblue')
    plt.fill_between(gens, obj1_mean - obj1_std, obj1_mean + obj1_std, alpha=0.2, color='dodgerblue')

    # --- Objectif 2 ---
    obj2_mean = np.mean(all_fitnesses[:, :, 1], axis=1)
    obj2_std = np.std(all_fitnesses[:, :, 1], axis=1)
    plt.plot(gens, obj2_mean, label='Obj2 mean', color='orangered')
    plt.fill_between(gens, obj2_mean - obj2_std, obj2_mean + obj2_std, alpha=0.2, color='orangered')

    # Améliorer le graphique
    plt.title("Objectives evolution over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder la figure
    save_path = os.path.join(result_dir, "objectives_evolution.pdf")
    plt.savefig(save_path)
    print(f"saved : {save_path}")
    plt.close()



if __name__ == "__main__":
    # Définit le chemin vers le dossier de résultats de ton run NSGA-II
    results_dir_multi = os.path.join(RESULTS_BASE, 'multi')

    print(f"Analysis in : {results_dir_multi}")

    # 1. Affiche l'évolution des deux objectifs au fil du temps
    plot_objectives_over_generations(results_dir_multi)

    # 2. Affiche les fronts de Pareto à différentes étapes de l'entraînement
    # (Début, milieu et fin)
    plot_pareto_front(results_dir_multi, generation=0)
    plot_pareto_front(results_dir_multi, generation=20) # Adapte ce numéro si besoin
    plot_pareto_front(results_dir_multi, generation=40) # La dernière génération