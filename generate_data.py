import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) #same data is generated each time you run the code (answer to life universe and everything)

GENRES = ['Classical', 'HipHop', 'Metal', 'Pop']
N_SAMPLES = 200        #samples per genre
N_ELECTRODES = 14      #EEG headsets typically have 14 channels

#each genre has characteristic alpha and beta wave patterns
#these are based on real neuroscience research findings

def generate_eeg_for_genre(genre, n_samples):
    """
    Simulates EEG alpha/beta wave readings for a given genre.
    Alpha waves (8-12 Hz) = relaxed focus
    Beta waves (13-30 Hz) = alertness and engagement
    """
    if genre == 'Classical':
        #high alpha, low beta
        alpha = np.random.normal(loc=0.75, scale=0.1, size=(n_samples, N_ELECTRODES)) #loc centers the distribution
        beta  = np.random.normal(loc=0.30, scale=0.1, size=(n_samples, N_ELECTRODES))

    elif genre == 'HipHop':
        #midrange alpha, high beta 
        alpha = np.random.normal(loc=0.50, scale=0.1, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.65, scale=0.1, size=(n_samples, N_ELECTRODES))

    elif genre == 'Metal':
        #low alpha, very high beta 
        alpha = np.random.normal(loc=0.30, scale=0.1, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.80, scale=0.1, size=(n_samples, N_ELECTRODES))

    elif genre == 'Pop':
        #midrange alpha and beta (why it's hardest to classify)
        alpha = np.random.normal(loc=0.55, scale=0.12, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.60, scale=0.12, size=(n_samples, N_ELECTRODES))

    #combine alpha and beta into one feature vector per sample
    return np.hstack([alpha, beta])  #shape: (n_samples, 28), stacks later, first 14 alpha last 14 beta

#building the full dataset
X = []  #features (EEG readings)
y = []  #labels (genre index)

for i, genre in enumerate(GENRES):
    data = generate_eeg_for_genre(genre, N_SAMPLES) #loops through each genre and generates the corresponding EEG data
    X.append(data) #adding into X
    y.extend([i] * N_SAMPLES) #adds corresponding index and genre to y (0 for classical, etc)

X = np.vstack(X) #stacking all 4 genres together
y = np.array(y)    


np.save('X.npy', X) #saving data
np.save('y.npy', y)
print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features each") #print to make sure it works 
print(f"Genre breakdown: {N_SAMPLES} samples per genre")

#plotting stuff
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = ['blue', 'orange', 'red', 'green']

for i, genre in enumerate(GENRES):
    genre_data = X[y == i]
    #average alpha power = first 14 features
    avg_alpha = genre_data[:, :N_ELECTRODES].mean(axis=1)
    #average beta power = last 14 features
    avg_beta = genre_data[:, N_ELECTRODES:].mean(axis=1)

    axes[0].scatter(avg_alpha, avg_beta, alpha=0.3, label=genre, color=colors[i])

axes[0].set_xlabel('Average Alpha Power')
axes[0].set_ylabel('Average Beta Power')
axes[0].set_title('EEG Alpha vs Beta by Genre')
axes[0].legend()

alpha_means = [X[y == i, :N_ELECTRODES].mean() for i in range(len(GENRES))] #only shows mean in graph, network is trained with all 14 individual values for each
beta_means  = [X[y == i, N_ELECTRODES:].mean() for i in range(len(GENRES))]

x = np.arange(len(GENRES))
width = 0.35
axes[1].bar(x - width/2, alpha_means, width, label='Alpha', color='steelblue')
axes[1].bar(x + width/2, beta_means,  width, label='Beta',  color='coral')
axes[1].set_xticks(x)
axes[1].set_xticklabels(GENRES)
axes[1].set_title('Mean Alpha & Beta Power by Genre')
axes[1].set_ylabel('Power')
axes[1].legend()

plt.tight_layout()
plt.savefig('eeg_data_visualization.png')
plt.show()
print("Visualization saved as eeg_data_visualization.png")

#Run using
#py -3.9 generate_data.py