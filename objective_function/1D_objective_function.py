import numpy as np
import matplotlib.pyplot as plt


def simple_objective_function(energy_loss, expected_mean=24):
    return np.mean(np.power(energy_loss - expected_mean, 2))


def generate_data_energy_loss(data_len, noise_amplitude=None, main_amplitude=None):
    starting_point = np.random.randint(0, 100)
    if not noise_amplitude:
        noise_amplitude = np.random.randint(2, 5)
    if not main_amplitude:
        main_amplitude = np.random.randint(2, 5)
    energy_loss = np.sin([val / 10 + np.random.rand() * noise_amplitude for val in
                          range(starting_point, starting_point + data_len)]) * main_amplitude + 24 + np.random.rand()
    return energy_loss


def plot_result(energy_loss, expected_mean=24):
    loss = simple_objective_function(energy_loss, expected_mean)
    plt.plot(energy_loss)
    mean = np.mean(energy_loss)
    plt.plot([mean for _ in range(len(energy_loss))])
    plt.plot([expected_mean for _ in range(len(energy_loss))])
    plt.title("Objective function loss: " + str(loss) + "\nMean: " + str(mean))
    plt.show()
