""" Plot Functions """
import matplotlib.pyplot as plt

def plot_matches(genuine_match, disguise_match, simulated_match, threshold):
    """ Plot Match Point percentages """

    plt.figure(figsize = (8, 8))
    match = [genuine_match, disguise_match, simulated_match]
    plt.hist(match, label=['Genuine', 'Disguise', 'Simulated'])
    plt.legend()
    plt.title('Matched Points')
    plt.savefig('../Plots/PMatchpoints' + str(threshold) + '.jpg')
    plt.show()

def plot_EER(theta_range, far, frr, threshold):
    """ Plot EER vs Theta """

    plt.figure(figsize = (8, 8))
    plt.plot(theta_range, far, color = 'red')
    plt.plot(theta_range, frr, color = 'blue')
    plt.title('EER vs Theta')
    plt.legend(['FAR', 'FRR'])
    plt.savefig('../Plots/Error' + str(threshold) + '.jpg')
    plt.show()
