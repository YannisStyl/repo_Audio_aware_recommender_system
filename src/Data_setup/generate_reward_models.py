# =============================================================================
# REWARD MODEL GENERATION SCRIPT
# =============================================================================
#
# Author: Ioannis Stylianou/Gemini 2.5 Pro
# Date: September 2025
#
# Description:
# This script serves as the complete data preparation pipeline for creating a
# reward model database used in RL-tuning of language models (e.g.,
# for GRPO or DPO training).
#
# The process involves two main data sources:
#   1. Real User Data: Sourced from CSV files containing prompts, model
#      predictions, and human-rated scores.
#   2. Augmented Data: High-fidelity, semantically equivalent synonyms for the
#      real user prompts are used to expand the dataset, improving model
#      robustness and generalization.
#
# The core of this script is the `NormalizedPreferenceDensity` class, which
# uses Kernel Density Estimation (KDE) to transform sparse data points with
# scores into a continuous reward function over a 2D space.
#
# UPDATED DATA SPLIT STRATEGY:
#   All real-data prompts (previously partitioned into a training pool plus
#   separately curated IID-validation and OOD-test concept clusters) are now
#   pooled together into a single set of concepts. For each concept, every
#   synonym sentence except one goes into the training set; the one remaining
#   synonym sentence is held out as the IID validation set. This is a
#   paraphrase-robustness split, not a concept-generalization split
#
# The final output is two separate pickle files: train and IID validation.
#
# =============================================================================

import pickle
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import os
import json

# =============================================================================
# 1. AUGMENTED DATA DEFINITION
# =============================================================================
# This dictionary defines the synonym sentences with which we will augment the data.
# Each key is the original prompt with its value being a list of paraphrases.

augmented_prompts = {
    "Enhance the high-end details in the audio.": [
        "Increase the detail in the high frequencies.",
        "Bring out more of the top-end details in the sound.",
        "I'd like to enhance the audio's high-end definition.",
        "Sharpen the high-frequency details in the sound.",
        "Add more definition to the upper register of the audio.",
        "Bring more clarity to the audio's high-end frequencies.",
    ],
    "I want a full, rich sound.": [
        "Give me a sound that is rich and full.",
        "The audio should have a full and rich character.",
        "I'm looking for a sound profile that is full and rich.",
        "Create a lush and full-bodied sound.",
        "I'm after a sound that feels thick in its richness.",
        "The audio should be full-bodied and sonically rich.",
    ],
    "Reduce the sharpness of the treble.": [
        "Lower the sharpness in the treble frequencies.",
        "Make the treble less sharp.",
        "I need a reduction in the sharpness of the treble.",
        "Tame the harsh edges of the treble.",
        "Soften the high frequencies to make them less piercing.",
        "I'd like the treble to have a smoother, less edgy character.",
    ],
    "Enhance the warmth of the music.": [
        "Increase the warmth in the music.",
        "The music should have more warmth.",
        "I want you to enhance the music's warmth.",
        "Add a warmer character to the music.",
        "The music needs a warmer tone.",
        "Turn up the warmth in this track.",
    ],
    "Lower the levels of harsh or aggressive audio elements that may cause stress.": [
        "Reduce harsh and aggressive elements in the audio.",
        "Decrease the level of any stressful, harsh sounds.",
        "Please lower the aggressive and harsh parts of the audio.",
        "Take the edge off any harsh or grating sounds.",
        "Soften the more aggressive and tense parts of the audio.",
        "Please tame the sharp or jarring sounds that make the audio stressful.",
    ],
    "Enhance the definition of audio for a deeper appreciation of the music.": [
        "Increase the audio's definition to better appreciate the music.",
        "For a deeper appreciation, enhance the definition of the audio.",
        "The definition of the audio should be enhanced for music appreciation.",
        "Improve the audio's resolution so the music's nuances are more audible.",
        "For a deeper listening experience, sharpen the audio's definition.",
        "I want the audio to be defined enough to reveal the subtleties of the music.",
    ],
    "Increase the music's presence to create a sensation of solitude.": [
        "Boost the presence of the music to feel more solitary.",
        "To create a feeling of solitude, increase the music's presence.",
        "I want a solitary sensation, so enhance the presence of the music.",
        "Heighten the music's sense of space to evoke a feeling of being alone.",
        "Strengthen the music's presence so it feels immersive and solitary.",
        "I want the music to feel more present and enveloping, like listening alone.",
    ],
    "Enhance the depth and power of the bass.": [
        "Increase the power and depth of the bass frequencies.",
        "I want the bass to have more depth and power.",
        "The depth and power of the bass should be enhanced.",
        "Boost the low end to give the bass more weight and impact.",
        "I want the bass frequencies to feel deeper and more powerful.",
        "Add more punch and fullness to the bass in the audio.",
    ],
    "Calibrate the sound to prevent harshness, especially when playing piano tracks.": [
        "Adjust the sound on piano tracks to avoid harshness.",
        "Prevent harshness on piano music by calibrating the sound.",
        "For piano tracks, the sound needs to be calibrated against harshness.",
        "Fine-tune the sound so piano music doesn't sound harsh or brittle.",
        "Smooth out any harshness in the audio, particularly on piano recordings.",
        "The piano tracks need a sound adjustment to remove any harsh or shrill qualities.",
    ],
    "Adjust the sound balance for a comforting effect.": [
        "Balance the audio to create a comforting effect.",
        "I want the sound balance adjusted for comfort.",
        "For a comforting effect, please adjust the sound's balance.",
        "Tune the audio to feel comforting and reassuring.",
        "I'd like the sound balance tweaked so it feels more comforting and gentle.",
        "Make the overall sound balance soothing and easy to listen to.",
    ],
    "Adjust the audio to highlight previously unnoticed instruments.": [
        "Change the audio to bring out instruments I hadn't noticed before.",
        "Highlight the previously hidden instruments by adjusting the audio.",
        "I want the audio adjusted to reveal unnoticed instruments.",
        "Rebalance the audio so subtler instruments can be heard more clearly.",
        "I'd like to hear the instruments that were buried in the mix before.",
        "Make the less prominent instruments stand out by adjusting the audio.",
    ],
    "Increase the separation between instruments and voices to add airiness.": [
        "Add airiness by increasing the separation of instruments and voices.",
        "Create more separation between voices and instruments for an airy feel.",
        "For more airiness, increase the instrumental and vocal separation.",
        "Open up the sound by creating more space between the instruments and vocals.",
        "Give the audio a more spacious and airy quality by widening the mix.",
        "I want the instruments and vocals to feel more distinct and spread out.",
    ],
    "Make the solo more prominent.": [
        "The solo needs to be more prominent in the mix.",
        "Increase the prominence of the solo.",
        "I want the solo to be featured more prominently.",
        "Bring the solo forward in the mix so it stands out.",
        "I'd like the solo to cut through more clearly.",
        "Turn up the solo so it takes centre stage.",
    ],
    "Adjust the audio settings to create a feel-good atmosphere.": [
        "Change the audio to create a feel-good atmosphere.",
        "Create a feel-good atmosphere by adjusting the audio settings.",
        "I want the audio settings adjusted for a feel-good vibe.",
        "Tune the audio to lift the mood and create a positive, feel-good vibe.",
        "I want the sound settings optimized for a cheerful and uplifting listening experience.",
        "Set the audio up to create an atmosphere that feels upbeat and enjoyable.",
    ],
    "Ensure the audio has a rich and engaging sound profile.": [
        "Make sure the audio's sound profile is engaging and rich.",
        "I want an engaging and rich sound profile from the audio.",
        "The audio needs a sound profile that is both rich and engaging.",
        "I need the audio to sound rich and captivating.",
        "Make sure the sound profile is both engaging and richly textured.",
        "The audio should have a sound that is layered, engaging, and full.",
    ],
    "Transition the sound ambiance to reflect the vibrancy of summer.": [
        "Change the sound's ambiance to be vibrant like summer.",
        "The ambiance of the sound should reflect summer's vibrancy.",
        "Make the sound ambiance transition to a vibrant, summery feel.",
        "Make the sound feel bright and alive, like a summer day.",
        "Shift the audio's character to something vibrant and full of energy, like summer.",
        "I want the sound ambiance to have the same lively and warm quality as summertime.",
    ],
    "Ensure the audio has a smooth and soothing character that aids concentration.": [
        "Make sure the audio is smooth and soothing to help with concentration.",
        "For concentration, the audio should have a soothing and smooth character.",
        "The character of the audio needs to be smooth and soothing to aid focus.",
        "The audio should be calm and unobtrusive enough to let me focus.",
        "Tune the audio to a smooth, even-keeled character that supports deep concentration.",
        "I need the sound to be tranquil and soothing so it doesn't break my concentration.",
    ],
    "Ensure the drums are distinctly audible.": [
        "Make sure the drums can be heard distinctly.",
        "The drums need to be distinctly audible in the mix.",
        "I want to ensure I can clearly hear the drums.",
        "I want the drum hits to be clear and well-defined in the mix.",
        "Make sure the percussion stands out and is easy to follow.",
        "Bring the drums forward enough so they are crisp and clearly audible.",
    ],
    "Enhance the clarity of sound to ensure nostalgia is amplified by crisp audio quality.": [
        "Improve sound clarity for a nostalgic audio experience.",
        "To amplify nostalgia, enhance the sound's clarity for crispness.",
        "The clarity of the sound should be enhanced to make the audio crisp and nostalgic.",
        "Make the audio crisp and clear to bring out its nostalgic qualities.",
        "A clean, high-clarity sound would really amplify the nostalgic feeling of this audio.",
        "Sharpen the audio so it feels crisp and evokes a sense of nostalgia.",
    ],
    "Ensure the sharpness of the instrument sounds is enhanced.": [
        "Make sure the instruments sound sharper.",
        "The sharpness of the instrumental sounds needs to be enhanced.",
        "I want to ensure the instruments have an enhanced sharpness.",
        "I want the instruments to have a sharper, more defined sound.",
        "Give each instrument a sharper, more present character in the audio.",
        "Enhance the transient definition of the instruments so they sound sharper.",
    ],
    "Balance the audio to ensure the backing vocals are not overshadowed by the lead vocals.": [
        "Balance the lead and backing vocals so the backing isn't overshadowed.",
        "Make sure the backing vocals are not buried by the lead vocals.",
        "Adjust the audio balance so the backing vocals are audible behind the lead.",
        "Blend the vocals so the backing harmonies are as audible as the lead.",
        "Pull the backing vocals up so they are not lost behind the lead voice.",
        "Make sure the lead vocals don't dominate to the point where the backing vocals disappear.",
    ],
}

# =============================================================================
# 1.5. REAL-DATA PROMPT POOL (formerly split into train / IID / OOD groups)
# =============================================================================
# These prompts previously defined separately curated IID-validation and
# OOD-test concept clusters. They are kept here (unmodified) purely as
# additional real-data concepts with their own synonym lists; they are now
# pooled together with `augmented_prompts` in Pass 3 rather than being
# treated as held-out concept clusters.

iid_validation_prompts = {
    "Provide clear and distinct audio playback for music.": [
        "The music playback should be clear and distinct.",
        "For music, provide audio playback that is clear and distinct.",
        "I need clear and distinct audio for my music.",
        "I want the music to play back with maximum clarity and definition.",
        "Make sure the audio reproduction is clean, detailed, and distinct.",
        "The music needs to be reproduced with clear and well-defined audio quality.",
    ],
    "Optimize vocal balance for acapella performances.": [
        "Balance the vocals for this acapella performance.",
        "I need an optimal vocal balance for acapella music.",
        "Create the best vocal balance for an acapella track.",
        "For an acapella arrangement, make sure all voices are balanced and audible.",
        "Adjust the vocal levels so each voice is equally present in the acapella.",
        "Tune the balance so all vocals in the acapella can be heard distinctly.",
    ],
    "Adjust the sound to emphasize the intricacies of vocal harmonization.": [
        "Change the sound to highlight the complex vocal harmonies.",
        "Emphasize the intricate vocal harmonizations with an audio adjustment.",
        "I want the sound adjusted to feature the vocal harmony details.",
        "Tune the audio so the subtle vocal harmonies are easier to appreciate.",
        "I want the sound adjusted so the layers of vocal harmony are more pronounced.",
        "Fine-tune the audio to draw out the fine details of the vocal harmonies.",
    ],
}

ood_test_prompts = {
    "Enhance vocal clarity in the audio.": [
        "Improve the clarity of the vocals in the audio.",
        "The vocals in the audio need enhanced clarity.",
        "Increase the vocal clarity within the audio.",
        "I want the vocal parts of the audio to sound cleaner and more intelligible.",
        "Make the voice clearer and more defined in the audio.",
        "Sharpen the vocal presence so the voice comes through with greater clarity.",
    ],
    "Optimize vocal presence for an immersive storytelling experience.": [
        "For immersive storytelling, optimize the presence of the vocals.",
        "Achieve an optimal vocal presence for an immersive story.",
        "The vocal presence should be optimized to make the storytelling immersive.",
        "Tune the vocal presence so the narrator draws the listener in fully.",
        "I want the voice to feel close and present, creating an immersive story experience.",
        "Adjust the vocal character so it commands attention and feels deeply engaging.",
    ],
    "Adjust the sound so I can hear the instructions better.": [
        "Change the audio so the instructions are easier to hear.",
        "I need the sound adjusted to better hear the instructions.",
        "To hear the instructions clearly, please adjust the sound.",
        "Please tune the audio so spoken instructions are clear and easy to follow.",
        "Make the voice giving instructions more audible and intelligible.",
        "I need the audio adjusted so every word in the instructions can be clearly understood.",
    ],
    "Enhance the liveliness of the narrator's voice delivery.": [
        "Increase the liveliness in the narrator's voice.",
        "The narrator's delivery should be enhanced for liveliness.",
        "Make the narrator's voice delivery more lively.",
        "Make the narrator's delivery sound more animated and engaging.",
        "I want the narrator's voice to feel more dynamic and expressive.",
        "Give the narration a more lively quality.",
    ],
    "Enhance the clarity of the vocal track.": [
        "Increase the clarity on the vocal track.",
        "The vocal track's clarity needs to be enhanced.",
        "Make the vocals on this track clearer.",
        "Improve the definition of the vocal track.",
        "I want the vocals on this track to sound more articulate and distinct.",
        "Clean up the vocal track so the voice is crisp and clearly defined.",
    ],
    "Ensure the narrator's voice is clear.": [
        "Make sure the narrator's voice is clear and understandable.",
        "The voice of the narrator needs to be clear.",
        "I need to ensure the clarity of the narrator's voice.",
        "Make certain the narrator can be heard and understood without any muddiness.",
        "The narrator's voice should come through cleanly and without ambiguity.",
        "Tune the audio so the narrator's delivery is crisp and easy to understand.",
    ],
}

# =============================================================================
# 1b. SIMULATED ANCHOR PROMPTS
# =============================================================================
# These prompts map to known geometric anchor points in the [-6, 6] x [-6, 6]
# EQ space. They are used exclusively to build synthetic reward models for
# diagnostic validation (see Pass 4 and SimulatedEvalCallback in t2b_fx_GRPO.py).
# They are NOT added to any training split.
#
# Axis semantics (confirmed from listening experiment):
#   x-axis:  Relaxed (−6) <--> Energetic (+6)
#   y-axis:  Warm / Bass-heavy (−6) <--> Bright / Treble-heavy (+6)
#
# Anchors are placed at ±5 rather than ±6 to keep the Gaussian density peak
# away from the reflection boundary, which prevents KDE edge artifacts.
# The "anchor" field is stored inside each reward_dict in the pickle so that
# SimulatedEvalCallback can compute L2 distances without re-importing this dict.

SIMULATED_ANCHOR_PROMPTS = {
    "energetic": {
        "anchor": (5.0, 0.0),
        "prompts": [
            "Make the audio sound more energetic.",
            "Give the music a punchy, dynamic character.",
            "I want an energetic and lively sound.",
            "The audio should feel more vibrant and driven.",
            "Add more energy and momentum to the sound.",
            "Give the track a more exciting and powerful quality.",
        ],
    },
    "relaxed": {
        "anchor": (-5.0, 0.0),
        "prompts": [
            "Make the audio sound more relaxed.",
            "Give the music a calm, laid-back character.",
            "I want a more subdued and gentle sound.",
            "The audio should feel softer and less intense.",
            "Create a more mellow and easygoing sound.",
            "Reduce the intensity of the audio so it feels more peaceful.",
        ],
    },
    "bright": {
        "anchor": (0.0, 5.0),
        "prompts": [
            "Make the audio sound brighter.",
            "Increase the brightness of the sound.",
            "I want a crisp, airy, and bright tone.",
            "The audio should have more high-frequency presence.",
            "Give the sound a more open, bright quality.",
            "Boost the clarity and airiness of the upper frequencies.",
        ],
    },
    "warm": {
        "anchor": (0.0, -5.0),
        "prompts": [
            "Make the audio sound warmer.",
            "Add warmth to the overall sound.",
            "I want a mellow, smooth, and warm tone.",
            "The audio should feel richer and less bright.",
            "Create a warmer, more enveloping sound character.",
            "Reduce the brightness and add more warmth to the audio.",
        ],
    },
    "high_treble": {
        "anchor": (5.0, 5.0),
        "prompts": [
            "Boost the treble and make the sound more energetic.",
            "I want a bright and punchy audio character.",
            "Increase the high frequencies and add energy.",
            "Give the sound a lively, high-frequency-forward character.",
            "I'm looking for a sound that is both bright and full of energy.",
            "Make the audio feel bright, crisp, and dynamically charged.",
        ],
    },
    "reduce_low_freq": {
        "anchor": (-5.0, 5.0),
        "prompts": [
            "Reduce the low frequencies in the audio.",
            "I want less bass and a thinner, lighter sound.",
            "Cut the low end to make the sound feel lighter.",
            "Make the audio feel airy by removing some of the bass.",
            "Lighten the sound by pulling back the low-frequency content.",
            "I want a thinner, brighter sound with less bass weight.",
        ],
    },
    "boost_low_freq": {
        "anchor": (5.0, -5.0),
        "prompts": [
            "Boost the low frequencies for a powerful sound.",
            "I want more bass impact and a fuller low end.",
            "Give the audio a bass-heavy, powerful character.",
            "Increase the low-frequency content for more impact.",
            "Make the bass feel fuller and more impactful in the mix.",
            "Add weight and punch to the low end of the audio.",
        ],
    },
    "low_treble": {
        "anchor": (-5.0, -5.0),
        "prompts": [
            "Reduce the treble and make the sound more relaxed.",
            "I want a dark, mellow sound with less brightness.",
            "Cut the high frequencies for a softer, warmer character.",
            "The audio should feel darker and less sharp.",
            "Make the sound more laid-back by rolling off the top end.",
            "I'd like the audio to have a darker, more subdued character with less treble.",
        ],
    },
    "neutralize": {
        "anchor": (0.0, 0.0),
        "prompts": [
            "Remove all audio filtering and return to a neutral sound.",
            "Neutralize the audio to a flat, unprocessed state.",
            "Reset the sound to its neutral, unfiltered baseline.",
            "Return the audio to a balanced, centre state with no adjustments.",
            "I want a completely neutral sound with no filters applied.",
            "Please remove any processing and restore the audio to its natural state.",
        ],
    },
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def reflect_to_8_squares(points):
    """
    Reflects a set of points from a central square into the 8 surrounding squares.
    This technique is used to handle boundary conditions in Kernel Density Estimation.
    By creating "ghost" points across the boundaries, we ensure the density
    estimate does not artificially drop to zero at the edges of our domain.
    Source: https://link.springer.com/article/10.1007/BF00147776

    Args:
        points (np.ndarray): An array of shape (N, 2) of points within the
                             [-6, 6] x [-6, 6] square.

    Returns:
        np.ndarray: An array of shape (N * 9, 2) containing the original points
                    plus all 8 sets of reflected "ghost" points.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input 'points' must be a NumPy array of shape (N, 2).")

    px, py = points[:, 0], points[:, 1]

    # Reflections across the four walls and four corners
    reflections = [
        np.column_stack((px, 12 - py)),       # Top
        np.column_stack((px, -12 - py)),      # Bottom
        np.column_stack((12 - px, py)),       # Right
        np.column_stack((-12 - px, py)),      # Left
        np.column_stack((12 - px, 12 - py)),  # Top-Right
        np.column_stack((-12 - px, 12 - py)), # Top-Left
        np.column_stack((12 - px, -12 - py)), # Bottom-Right
        np.column_stack((-12 - px, -12 - py)) # Bottom-Left
    ]

    return np.vstack([points] + reflections)

def parse_stringified_array(array_of_strings):
    """
    Converts a 2D array of strings (e.g., from a CSV) where each string
    represents a coordinate pair like '[1.23 4.56]' into a 3D NumPy array of floats.
    """
    rows, cols = array_of_strings.shape
    result = np.zeros((rows, cols, 2), dtype=float)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.fromstring(array_of_strings[i, j].strip('[]'), sep=' ')
    return result

# =============================================================================
# 3. CORE DENSITY ESTIMATION CLASS
# =============================================================================

class NormalizedPreferenceDensity:
    """
    Computes a normalized preference density function from scattered data points.

    This class takes a set of 2D samples and associated preference scores
    (weights) and uses Kernel Density Estimation (KDE) to create a smooth,
    continuous function representing the density of preferred points.

    Crucially, it normalizes the output to a consistent [-1, 1] range, across all
    data. This requires a first pass for the computation of the global maximum and
    minimum, and a second pass for the computation of the density functions.
    The class is designed to be pickle-safe by storing raw data and recreating
    KDEs on-the-fly, rather than pickling the KDE objects themselves.
    """
    def __init__(self, samples, preference_weights, use_reflection=True,
                 bw_method="scott", epsilon=1e-9, normalize=False,
                 global_min=None, global_max=None):
        """
        Initializes the density estimator.

        Args:
            samples (np.ndarray): Data points of shape (n_samples, 2).
            preference_weights (np.ndarray): Weights for each sample.
            use_reflection (bool): If True, use boundary reflection for KDE.
            bw_method (str): Bandwidth estimation method for KDE.
            epsilon (float): Small constant for numerical stability.
            normalize (bool): If True, scale the output to [-1, 1].
            global_min (float, optional): A global minimum value to use for normalization.
            global_max (float, optional): A global maximum value to use for normalization.
        """
        if samples.shape[0] != len(preference_weights):
            raise ValueError("The number of samples must match the number of weights.")
        if samples.shape[0] < 2:
            raise ValueError("KDE requires at least 2 data points.")

        self.epsilon = epsilon
        self.normalize = normalize
        self.global_min = global_min
        self.global_max = global_max

        # Store the data required to reconstruct KDEs. This is pickle-safe.
        # If reflection is used, we pre-reflect the points and weights.
        if use_reflection:
            self.dataset = reflect_to_8_squares(samples)
            self.weights = np.tile(preference_weights, 9)
        else:
            self.dataset = samples
            self.weights = preference_weights

        # We need a bandwidth factor for KDE. We calculate it once here.
        kde_naive = gaussian_kde(self.dataset.T, bw_method=bw_method)
        self.bw_factor = kde_naive.factor
        if use_reflection:
            self.bw_factor *= 0.25 # Scaling factor for reflection, empirically verified

        # Calculate local normalization parameters (min/max of this specific function)
        self._calculate_local_normalization_params()

    def _calculate_local_normalization_params(self):
        """
        Calculates the min and max raw density values for this specific function.
        This is used for normalization if global min/max are not provided.
        """
        kde_all = gaussian_kde(self.dataset.T, bw_method=self.bw_factor)

        # If there are no positive weights, density is flat.
        if not np.any(self.weights > 0):
            self.max_density, self.min_density = 0.0, 0.0
            return

        kde_spec = gaussian_kde(self.dataset.T, bw_method=self.bw_factor, weights=self.weights)

        # Evaluate over a grid to find the density range
        x_coords = np.linspace(-6, 6, 100)
        y_coords = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([X.ravel(), Y.ravel()])

        density_all_vals = kde_all(grid_points)
        density_spec_vals = kde_spec(grid_points)
        raw_grid_densities = density_spec_vals / (density_all_vals + self.epsilon)

        self.min_density = np.min(raw_grid_densities)
        self.max_density = np.max(raw_grid_densities)

    def __call__(self, points):
        """
        Evaluates the preference density at the given points.
        This method is the callable interface of the object.
        """
        # Recreate KDEs on-the-fly from stored data for pickle compatibility
        kde_all = gaussian_kde(self.dataset.T, bw_method=self.bw_factor)

        if not np.any(self.weights > 0):
            return np.zeros(points.shape[1] if points.ndim > 1 else 1)

        kde_spec = gaussian_kde(self.dataset.T, bw_method=self.bw_factor, weights=self.weights)

        raw_density = kde_spec(points) / (kde_all(points) + self.epsilon)

        if self.normalize:
            # Prioritize global normalization if values were provided during init.
            # Otherwise, fall back to the locally calculated min/max.
            min_val = self.global_min if self.global_min is not None else self.min_density
            max_val = self.global_max if self.global_max is not None else self.max_density

            denominator = max_val - min_val
            # Avoid division by zero if the density is completely flat
            if denominator < self.epsilon:
                return np.zeros_like(raw_density)

            # Scale the raw density to the [-1, 1] range for the reward signal
            normalized_density = 2 * (raw_density - min_val) / denominator - 1
            return normalized_density
        else:
            return raw_density

def make_simulated_reward_model(anchor_xy, n_samples=80, sigma=1.0, seed=42):
    """
    Constructs a NormalizedPreferenceDensity from synthetic Gaussian samples
    centred around a known semantic anchor in the EQ space.

    Synthetic "votes" are drawn from a 2D isotropic Gaussian and assigned a
    uniform preference weight of 1.0, producing a smooth density peak at the
    anchor coordinate. These reward models are used exclusively for diagnostic
    validation; they are NOT included in any training split.

    Args:
        anchor_xy (tuple): (x, y) target coordinate in [-6, 6]^2.
        n_samples (int):   Number of synthetic vote points. Controls peak
                           sharpness: higher = tighter, more concentrated peak.
        sigma (float):     Gaussian spread in EQ-space units. sigma=1.0
                           produces a reward region spanning roughly ±2 units.
        seed (int):        RNG seed for reproducibility across runs.

    Returns:
        NormalizedPreferenceDensity: Callable reward function peaking at
        anchor_xy, normalized to [-1, 1] using local min/max.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=anchor_xy, scale=sigma, size=(n_samples, 2))
    samples = np.clip(samples, -6.0, 6.0)  # Keep all votes inside the EQ domain
    weights = np.ones(n_samples)            # Uniform maximum preference
    return NormalizedPreferenceDensity(samples, weights, normalize=True)


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    REAL_DATA_RESULTS_PATH = "Listening_Exp/results_filtered.csv"
    REAL_DATA_PROMPTS_PATH = "Listening_Exp/prompt_sequence.csv"
    OUTPUT_DIR = "Data_dir/reward_models"
    
    # --- Setup ---
    print("Loading real user data...")
    results_df = pd.read_csv(REAL_DATA_RESULTS_PATH)
    prompts_df = pd.read_csv(REAL_DATA_PROMPTS_PATH, index_col=0)
    base_reward_models = {}

    point_cols = ["LoRA_pred", "RAG_pred", "Random_pred", "T2B_pred"]
    score_cols = ["LoRA_score", "RAG_score", "Random_score", "T2B_score"]
    
    # =========================================================================
    # MULTI-PASS WORKFLOW FOR GLOBAL NORMALIZATION AND DATA SPLITTING
    # =========================================================================
    
    # --- PASS 1 (Done beforehand): Calculate Global Min/Max from ALL Real Data ---
    # To ensure all reward functions from real data share a consistent scale,
    # we first loop through all available data to find the global min and max
    # raw density values before any normalization is applied. 
    # =========================================================================

    """
    print("\n--- Pass 1: Calculating global normalization parameters from all real data ---")
    Maxes = []
    Mins = []

    for i in range(len(prompts_df)):
        results_per_prompt = results_df[results_df["promptId"] == i]
        if results_per_prompt.empty:
            continue

        prompt = prompts_df[prompts_df['promptId'] == i]["name"].iloc[0]

        prompt_points_flat = parse_stringified_array(results_per_prompt[point_cols].values).reshape(-1, 2)
        prompt_scores_flat = results_per_prompt[score_cols].values.flatten()

        if prompt_points_flat.shape[0] < 2:
            print(f"Skipping prompt '{prompt}' in Pass 1 due to insufficient data.")
            continue

        # Create a temporary, UN-NORMALIZED function just to find its density range
        temp_func = NormalizedPreferenceDensity(
            prompt_points_flat, np.maximum(0, prompt_scores_flat), normalize=False
        )

        Maxes.append(temp_func.max_density)
        Mins.append(temp_func.min_density)

    overall_max = max(Maxes)
    overall_min = min(Mins)
    max_avg = np.mean(Maxes)
    min_avg = np.mean(Mins)

    print(f"Global Min Raw Density: {overall_min:.4f}, Global Max Raw Density: {overall_max:.4f}")
    print(f"Average Min Raw Density: {min_avg:.4f}, Average Max Raw Density: {max_avg:.4f}")
    """
    
    # Results:
    #Global Min Raw Density: 0.0001, Global Max Raw Density: 2.3905
    #Average Min Raw Density: 0.1321, Average Max Raw Density: 1.8947
    overall_max=2.3905
    overall_min=0.0001

    # =========================================================================
    # PASS 2: Generate Base Reward Models with Global Normalization
    # Now that we have the global min/max, we create the final, correctly
    # normalized reward function for each original prompt.
    # =========================================================================
    print("\n--- Pass 2: Creating reward models for each original prompt ---")
    for i in range(len(prompts_df)):
        results_per_prompt = results_df[results_df["promptId"] == i]
        if results_per_prompt.empty:
            continue

        prompt = prompts_df[prompts_df['promptId'] == i]["name"].iloc[0]

        prompt_points_flat = parse_stringified_array(results_per_prompt[point_cols].values).reshape(-1, 2)
        prompt_scores_flat = results_per_prompt[score_cols].values.flatten()

        if prompt_points_flat.shape[0] < 2:
            continue

        # Create the FINAL "positive" reward model, passing the global values
        NORMALIZE = False
        positive_reward_func = NormalizedPreferenceDensity(
            prompt_points_flat,
            np.maximum(0, prompt_scores_flat),
            normalize=True
        )
        base_reward_models[prompt] = {"positive_reward": positive_reward_func}
        print(f"Generated reward function for: '{prompt}'")

    # =========================================================================
    # PASS 3: Pool, Augment, and Save the Final Datasets
    # All real-data prompts are now pooled together (no held-out concept
    # clusters). For each prompt, all synonyms except the last go to train;
    # the last synonym sentence is held out as the IID validation point.
    # =========================================================================
    print("\n--- Pass 3: Pooling, augmenting, and saving final datasets ---")

    # Pool every real-data prompt's synonym list together. Formerly these
    # three dicts defined separate train / IID / OOD concept groups -- now
    # they're just one combined lookup of prompt -> synonym list.
    ALL_PROMPT_SYNONYMS = {**augmented_prompts, **iid_validation_prompts, **ood_test_prompts}

    # Seeded RNG so the held-out synonym per prompt is reproducible across runs.
    SPLIT_SEED = 42
    split_rng = np.random.default_rng(SPLIT_SEED)

    final_reward_models_train = {}
    final_reward_models_val = {}

    for prompt, reward_dict in base_reward_models.items():
        if prompt not in ALL_PROMPT_SYNONYMS:
            print(f"Warning: no synonym list found for '{prompt}', skipping augmentation for this prompt.")
            continue

        synonyms = ALL_PROMPT_SYNONYMS[prompt]

        # Randomly pick one synonym per prompt to hold out for validation.
        val_idx = split_rng.integers(len(synonyms))
        val_synonym = synonyms[val_idx]
        train_synonyms = synonyms[:val_idx] + synonyms[val_idx + 1:]

        # Original prompt + all synonyms except the held-out one -> training set
        final_reward_models_train[prompt] = reward_dict
        for synonym in train_synonyms:
            final_reward_models_train[synonym] = reward_dict

        # Held-out synonym -> IID validation set
        final_reward_models_val[val_synonym] = reward_dict

    print("\n--- Dataset Split Summary ---")
    print(f"Final training set size (with augmentations): {len(final_reward_models_train)} prompts")
    print(f"Final IID validation set size: {len(final_reward_models_val)} prompts")

    # --- Final Step: Save the two databases to separate files ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Define output paths
    TRAIN_DB_PATH = os.path.join(OUTPUT_DIR, "reward_models_augmented_train.pkl")
    VAL_DB_PATH = os.path.join(OUTPUT_DIR, "reward_models_augmented_val.pkl")

    with open(TRAIN_DB_PATH, "wb") as f:
        pickle.dump(final_reward_models_train, f)
    print(f"\nSuccessfully saved training reward models to '{TRAIN_DB_PATH}'")

    with open(VAL_DB_PATH, "wb") as f:
        pickle.dump(final_reward_models_val, f)
    print(f"Successfully saved validation reward models to '{VAL_DB_PATH}'")

    # =========================================================================
    # PASS 4: Generate Simulated Anchor Reward Models (Validation / Sanity Check)
    # Each concept in SIMULATED_ANCHOR_PROMPTS is mapped to a synthetic
    # NormalizedPreferenceDensity peaking at its anchor coordinate (sigma=1.0).
    # The anchor tuple is stored alongside the reward function in the reward_dict
    # so that SimulatedEvalCallback in t2b_fx_GRPO.py can compute L2 distances
    # without needing to re-import SIMULATED_ANCHOR_PROMPTS at training time.
    # =========================================================================
    print("\n--- Pass 4: Generating simulated anchor reward models ---")
    simulated_reward_models = {}
    for concept_name, entry in SIMULATED_ANCHOR_PROMPTS.items():
        reward_func = make_simulated_reward_model(entry["anchor"])
        reward_dict = {
            "positive_reward": reward_func,
            "anchor": entry["anchor"],  # Stored for L2 evaluation in callbacks
        }
        for prompt_text in entry["prompts"]:
            simulated_reward_models[prompt_text] = reward_dict
        print(f"  Concept '{concept_name}' @ anchor {entry['anchor']}: {len(entry['prompts'])} prompts")

    SIMULATED_DB_PATH = os.path.join(OUTPUT_DIR, "reward_models_simulated_val.pkl")
    with open(SIMULATED_DB_PATH, "wb") as f:
        pickle.dump(simulated_reward_models, f)
    print(f"\nSuccessfully saved {len(simulated_reward_models)} simulated reward models to '{SIMULATED_DB_PATH}'")
    
    """
    # Dictionary for retrieving the original prompts (ran once)
    REVERSE_AUG_PROMPTS = {}
    for original, aug_list in augmented_prompts.items():
        for aug in aug_list:
            REVERSE_AUG_PROMPTS[aug] = original
            
    for original, aug_list in iid_validation_prompts.items():
        for aug in aug_list:
            REVERSE_AUG_PROMPTS[aug] = original 
    
    for original, aug_list in ood_test_prompts.items():
        for aug in aug_list:
            REVERSE_AUG_PROMPTS[aug] = original
            
    
    
    with open('Audio_directory/reverse_augmented_prompts.json', 'w') as f:
        json.dump(REVERSE_AUG_PROMPTS, f, indent=4)
        
    """