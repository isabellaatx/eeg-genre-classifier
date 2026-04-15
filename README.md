# EEG Genre Classifier

Music engages complex neural networks across the brain, influencing alpha waves (8–12 Hz, relaxation) and beta waves (13–30 Hz, alertness) in ways that vary by genre. Musical training further shapes these responses—musicians develop stronger, more consistent neural reactions than non-musicians. My project takes a computational approach to ask: can a model predict what genre of music someone is listening to from brain wave activity alone, and does that accuracy differ between musicians and non-musicians? Understanding this has practical applications for therapeutic uses of music and could help define what practices are most effective per person.

# Dataset Construction

Synthetic EEG data was generated using NumPy with parameters based on real neuroscience research. 1,600 total samples were created across two groups (800 musicians, 800 non-musicians), with 200 samples per genre across the genres Classical, Hip-Hop, Metal, and Pop. Each participant is represented by 28 values: 14 alpha and 14 beta readings, matching the 14 electrode channels on a real EEG headset.

Important design decisions:
+ Normal distribution: Readings were generated on a bell curve around genre-specific means, reflecting how real neural data clusters with natural variation
+ Genre separation: Classical (high alpha, low beta) and metal (low alpha, high beta) sit at opposite ends; pop and hip-hop intentionally overlap to reflect their similar neural patterns
+ Musician vs. non-musician: Musician data used lower standard deviation (0.07 vs. 0.13) and more extreme mean values, reflecting research showing musical training produces stronger, more consistent neural responses

# Neural Network Methodology

+ For the model, I built a feedforward neural network using TensorFlow/Keras
+ Input: 28 normalized EEG readings per participant
+ Architecture: two hidden layers (64 and 32 neurons), ReLU activation, 30% dropout after each layer, softmax output layer (4 neurons)
+ Training: 50 epochs, Adam optimizer, sparse categorical crossentropy loss, 80/20 train/test split
+ Two separate models were trained—one on musician data, one on non-musician data—using the same architecture to ensure their comparability
+ Cross-group test conducted: the musician model was evaluated on non-musician data to measure how distinct the two groups' neural signatures really are

# Results

The following 3 conditions were tested, and are listed with their corresponding accuracies:

+ Musician model tested on musician data: 96.2%
+ Non-musician model tested on non-musician data: 73.1%
+ Musician model tested on non-musician data: 75.9%

The musician model returned a significantly higher classification accuracy (96.2%) compared to the non-musician model (73.1%), indicating that having musical training produces more distinct neural signatures across genres. The confusion matrix analysis revealed that misclassifications were most frequent between pop and hip-hop in both groups, consistent with the overlapping alpha and beta wave values assigned to those genres based on their similar tempo and other characteristics.
The cross-group accuracy of 75.9%, which was much lower than the musician model's 96.2% accuracy within its own group, suggests that the neural signatures of musicians and non-musicians are distinct enough that a model trained on one group does not fully apply to the other.

# Conclusion

The results support the hypothesis that musical training produces brain wave patterns that are more separable and consistently classifiable by a machine learning model. The 23.1% accuracy gap between groups reflects a crucial neurological difference between the two groups, not only in the strength of their responses but also in their consistency across the participants. The genres with the most distinct profiles were classified most accurately in both groups. Classical and metal sit at opposite ends of the alpha/beta spectrum (classical producing high alpha and low beta and metal producing the inverse) making them the easiest for the network to distinguish. Pop and hip-hop produced the most inaccuracies due to their overlapping neural profiles, reflecting real research showing these genres share similar tempo and rhythmic structure. The system views these two genres as virtually the same, making it a lot harder to classify. The cross-group’s musician network accuracy drop from 96.2% to 75.9% suggests the two groups' neural signatures are distinct enough that models cannot easily find the same results using different sets of data, which is important to consider for a practical application of this experiment. In settings where music is used as therapy or even as a sort of medicine, it is important to tailor the  genre used based on the patient’s musical background. 

# Future Applications

In terms of this research:
+ Apply this methodology to real EEG datasets
+ Explore how accuracy changes based on differing years of musical experience, instrument training, or genre preference
+ Investigate how genre-specific brain waves affect other parts of the body

Personally:
+ Complete Andrej Kaparthy's Neural Networks: Zero to Hero course to understand the math and code that goes into libraries like TensorFlow/Keras

# Image Key

eeg_data_visualization.png:
+ Models showing the deviation of the EEG data constructed in Python (in the base data generation file, not differentiating from musician/non-musician).

training_histroy.png:
+ Training data of the original neural network. The model accuracy increases over sets while the model loss decreases. The convergence of the training and validation curves indicates the model’s success and avoidance of overfitting.

confusion_matrix.png:
+ A confusion matrix showing the results of what genre was predicted vs what genre was actually given in the base model trained on the non-differentiated data.

musician_comparison.png:
+ Left: a bar graph higlighting the different accuracies between each test.
+ Center: the validation accuracy graph showing both the musician and non-musician model.
+ Right: Musician model on Musician data: a confusion matrix showing the results of what genre was predicted vs what genre was actually given.

nonmusician_confusion.png:
+ Non musician model on non-musician data: a confusion matrix showing the results of what genre was predicted vs what genre was actually given.
