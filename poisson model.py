import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Parameters for the simulation
# [For the exact A1 model parameters see Han et al. (2007)]

# Set the frequency range
low = 1
high = 20
duration = 800  # Set the resolution of the simulation
x = np.linspace(low, high, duration)

# Set the number of neurons
num_neurons = 100

# Specify the category boundaries
class_boundary = [5, 10, 15]

# Take only a subset of neurons
neurons_per_class = int(num_neurons / 4)
if neurons_per_class % 2 != 0:
    neurons_per_class += 1
ones = np.ones(neurons_per_class)
sigma = 0.3  # the spread of the neurons around the class boundaries/center
labels = np.concatenate((ones * 0, ones * 1, ones * 2, ones * 0))

# Set the best frequency parameters - Uniform
best_frequency = np.linspace(low, high, num_neurons)

# Choose neuronal arrangement
cb_clusters = False
cm_clusters = not cb_clusters

# According to the choice above - arrange the neurons in the following way:
# If cb_clusters: Clustered around the category boundaries
if cb_clusters:
    heading = "Category Boundaries"

    # Set the frequency tuning curve parameters - non-Uniform
    for i in range(len(class_boundary)):

        # Find the boundaries of the class
        cb = np.where((best_frequency > class_boundary[i]) & (best_frequency < class_boundary[i] + 1))[0][0]

        # Set the frequency tuning curve around the boundaries of the class
        best_frequency[cb - int(neurons_per_class / 2):cb + int(neurons_per_class / 2)] = ones * class_boundary[
            i] + sigma * np.random.randn(neurons_per_class)

# If cm_clusters: Clustered around the category centers
elif cm_clusters:
    heading = "Category Center"

    for i in range(len(class_boundary) - 1):

        # Find the center of the class
        class_center = (class_boundary[i + 1] + class_boundary[i]) / 2
        cm = np.where((best_frequency > class_center) & (best_frequency < class_center + 1))[0][0]

        # Set the frequency tuning curve around the center of the class
        best_frequency[
        cm - int(neurons_per_class / 2):cm + int(neurons_per_class / 2)] = ones * class_center + sigma * np.random.randn(
            neurons_per_class)
else:
    heading = "Uniform"

# Generate the frequency tuning curves for each neuron
dx = x[1] - x[0]

tuning_curves = np.zeros((len(best_frequency), duration))
std_tuning_curves = np.zeros(len(best_frequency))
FI = np.zeros((len(best_frequency), duration))

for i in range(len(best_frequency)):
    mu = best_frequency[i]
    variance = np.random.lognormal(-0.75, .47)
    sigma = np.sqrt(variance)
    tuning_curves[i] = norm.pdf(x, mu, sigma)
    std_tuning_curves[i] = sigma
    FI[i] = np.power(np.gradient(tuning_curves[i], dx), 2) / (tuning_curves[i] + np.finfo(float).eps)

# Define the Poisson firing rate for each neuron
firing_rate = np.random.poisson(tuning_curves)

# Generate the spike trains for each neuron
spike_trains = np.random.poisson(firing_rate)


# Plot the tuning & FI
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, tuning_curves.T)
ax2.plot(x, FI.sum(axis=0))
ax2.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Firing Rate')
ax2.set_ylabel('FI')
fig.suptitle(heading)

for i in range(len(class_boundary)):
    ax1.axvline(class_boundary[i])
    ax2.axvline(class_boundary[i])

plt.show()

