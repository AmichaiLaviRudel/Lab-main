import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# For the exact A1 model parameters see Han et. al. (2007)
low = 1
high = 20
# Set the number of neurons
num_neurons = 800
class_boundary = [5, 10, 15]

# Set the duration of the simulation
duration = 800

x = np.linspace(low, high, duration)

# Choose neuronal arrangement
cb_clusters = False
cm_clusters = True

# On the CB
neurons_per_class = int(num_neurons/5)
ones = np.ones(neurons_per_class)
sigma = 0.3

best_frequency = np.linspace(low, high, num_neurons)
heading = "uniform"


if cb_clusters:
    # Set the frequency tuning curve parameters - non-Uniform
    for i in range(len(class_boundary)):
        cb = np.where((best_frequency > class_boundary[i]) & (best_frequency < class_boundary[i] + 1))[0][0]
        best_frequency[cb - int(neurons_per_class / 2):cb + int(neurons_per_class / 2)] = ones * class_boundary[
            i] + sigma * np.random.randn(neurons_per_class)
    heading = "Category Boundaries"

elif cm_clusters:
    # Inbetween the CB
    for i in range(len(class_boundary)):

        cb = np.where((best_frequency > class_boundary[i]) & (best_frequency < class_boundary[i] + 1))[0][0]
        try:
            class_center = (class_boundary[i + 1] - class_boundary[i]) / 2
        except:
            continue

        best_frequency[cb - int(neurons_per_class / 2):cb + int(neurons_per_class / 2)] = \
            ones * class_boundary[i] + sigma * np.random.randn(neurons_per_class) \
            + class_center
    heading = "Category Center"


# Generate the frequency tuning curves for each neuron
dx = x[1] - x[0]


tuning_curves = np.zeros((len(best_frequency), duration))
FI = np.zeros((len(best_frequency), duration))

for i in range(len(best_frequency)):
    mu = best_frequency[i]
    variance = np.random.lognormal(-0.75, .47)
    sigma = np.sqrt(variance)
    tuning_curves[i] = norm.pdf(x, mu, sigma)
    FI[i] = np.power(np.gradient(tuning_curves[i], dx), 2)/(tuning_curves[i]+np.finfo(float).eps)

# Define the Poisson firing rate for each neuron
firing_rate = np.random.poisson(tuning_curves)

# Plot the tuning & FI
fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True)
# ax.imshow(firing_rate, cmap='binary', aspect="auto")
ax1.plot(x, tuning_curves.T)
ax2.plot(x, FI.sum(axis=0))
ax2.set_xlabel('Frequency (Hz)')
ax.set_ylabel('# Neuron')
ax1.set_ylabel('Firing Rate')
ax2.set_ylabel('FI')
fig.suptitle(heading)

for i in range(len(class_boundary)):
    ax1.axvline(class_boundary[i])
    ax2.axvline(class_boundary[i])
plt.show()

