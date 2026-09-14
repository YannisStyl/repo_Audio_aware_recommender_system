import pickle
from generate_reward_models import reflect_to_8_squares, NormalizedPreferenceDensity # type:ignore
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__=="__main__":

    split = "train"
    data_path = f"Data_dir/reward_models/reward_models_augmented_{split}.pkl"
    os.makedirs(f"src/Data_setup/RM_Visual_{split}", exist_ok = True)

    with open(data_path, 'rb') as f:
        reward_models = pickle.load(f)

    # Load prompts from the keys of the reward models dictionary
    prompts = list(reward_models.keys())
    for i, prompt in enumerate(prompts):
        
        prompt_density = reward_models[prompt]["positive_reward"]
        fig, ax = plt.subplots()

        # plot density with a [-6,6]x[-6,6] meshgrid
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        XY = np.array([X.flatten(), Y.flatten()])
        Z = prompt_density(XY)

        im = ax.imshow(Z.reshape(X.shape), 
                    extent=[-6, 6, -6, 6], 
                    origin='lower', cmap='viridis', 
                    aspect='auto')        
        
        fig.colorbar(im, ax=ax, label='Preference Density')

        ax.set_title(f"Reward Density for:\n'{prompt}'", wrap=True)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

        #save image
        plt.savefig(f"src/Data_setup/RM_Visual_{split}/Fig_{i+1}.png")
        plt.close(fig)