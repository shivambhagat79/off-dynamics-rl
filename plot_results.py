import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_scalar(env_name, scalar_name, x_label, y_label, title):
    """
    Plots a scalar for a given environment from tensorboard logs.
    """
    plt.figure(figsize=(12, 8))

    results_dir = 'results'
    algos = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    for algo in algos:
        env_path = os.path.join(results_dir, algo, env_name)
        if not os.path.exists(env_path):
            continue

        all_seeds_data = []
        min_len = float('inf')

        seed_dirs = sorted([d for d in os.listdir(env_path) if d.startswith('r') and os.path.isdir(os.path.join(env_path, d))])

        for seed_dir in seed_dirs:
            seed_path = os.path.join(env_path, seed_dir, 'tb')
            if not os.path.exists(seed_path):
                continue

            try:
                ea = event_accumulator.EventAccumulator(seed_path,
                    size_guidance={event_accumulator.SCALARS: 0})
                ea.Reload()

                if scalar_name in ea.Tags()['scalars']:
                    scalar_data = ea.Scalars(scalar_name)
                    steps = [s.step for s in scalar_data]
                    values = [s.value for s in scalar_data]

                    # Use steps as x-axis for total_novel_states, otherwise use episode number
                    if 'total_novel_states' in scalar_name:
                        all_seeds_data.append((steps, values))
                        min_len = min(min_len, len(steps))
                    else:
                        all_seeds_data.append(values)
                        min_len = min(min_len, len(values))

            except Exception as e:
                print(f"Could not process {seed_path} for scalar {scalar_name}: {e}")

        if not all_seeds_data:
            continue

        # Trim all data to the minimum length
        if 'total_novel_states' in scalar_name:
            processed_seeds_data = []
            for steps, values in all_seeds_data:
                processed_seeds_data.append((steps[:min_len], values[:min_len]))
            all_seeds_data = processed_seeds_data
            steps = all_seeds_data[0][0]
            all_values = np.array([d[1] for d in all_seeds_data])
        else:
            processed_seeds_data = [d[:min_len] for d in all_seeds_data]
            all_values = np.array(processed_seeds_data)
            steps = np.arange(min_len)


        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        # Smooth the curves
        mean_values = smooth(mean_values, 0.6)

        plt.plot(steps, mean_values, label=algo)
        plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join('plots', env_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_filename = os.path.join(plot_dir, f'{scalar_name.replace("/", "_")}.png')
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='Name of the environment, e.g., "halfcheetah-crippled-thigh-0.0"')
    args = parser.parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot 1: novelty/episode_novel_states
    plot_scalar(args.env, 'novelty/episode_novel_states', 'Episode Number', 'Novel States in Episode', f'Episode Novel States on {args.env}')

    # Plot 2: novelty/total_novel_states
    plot_scalar(args.env, 'novelty/total_novel_states', 'Training Timestep', 'Total Novel States', f'Total Novel States on {args.env}')

    # Plot 3: test/target_return
    plot_scalar(args.env, 'test/target_return', 'Test Number', 'Target Test Score', f'Target Test Score on {args.env}')
