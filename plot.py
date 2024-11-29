import numpy as np
import pickle

# Load the existing results dictionary
with open('MLP_MNIST_results.pkl', 'rb') as file:
    results_dict = pickle.load(file)

train_losses_ADAM = results_dict['train_losses_ADAM']
test_losses_ADAM = results_dict['test_losses_ADAM']
train_accuracies_ADAM = results_dict['train_accuracies_ADAM']
test_accuracies_ADAM = results_dict['test_accuracies_ADAM']
times_ADAM = results_dict['times_ADAM']
train_losses_AMSGrad = results_dict['train_losses_AMSGrad']
test_losses_AMSGrad = results_dict['test_losses_AMSGrad']
train_accuracies_AMSGrad = results_dict['train_accuracies_AMSGrad']
test_accuracies_AMSGrad = results_dict['test_accuracies_AMSGrad']
times_AMSGrad = results_dict['times_AMSGrad']
train_losses_ADAMX = results_dict['train_losses_ADAMX']
test_losses_ADAMX = results_dict['test_losses_ADAMX']
train_accuracies_ADAMX = results_dict['train_accuracies_ADAMX']
test_accuracies_ADAMX = results_dict['test_accuracies_ADAMX']
times_ADAMX = results_dict['times_ADAMX']
train_losses_ADAMBS = results_dict['train_losses_ADAMBS']
test_losses_ADAMBS = results_dict['test_losses_ADAMBS']
train_accuracies_ADAMBS = results_dict['train_accuracies_ADAMBS']
test_accuracies_ADAMBS = results_dict['test_accuracies_ADAMBS']
times_ADAMBS = results_dict['times_ADAMBS']
train_losses_ADAMBS_corrected = results_dict['train_losses_ADAMBS_corrected']
test_losses_ADAMBS_corrected = results_dict['test_losses_ADAMBS_corrected']
train_accuracies_ADAMBS_corrected = results_dict['train_accuracies_ADAMBS_corrected']
test_accuracies_ADAMBS_corrected = results_dict['test_accuracies_ADAMBS_corrected']
times_ADAMBS_corrected = results_dict['times_ADAMBS_corrected']
train_losses_ADAMCB = results_dict['train_losses_ADAMCB']
test_losses_ADAMCB = results_dict['test_losses_ADAMCB']
train_accuracies_ADAMCB = results_dict['train_accuracies_ADAMCB']
test_accuracies_ADAMCB = results_dict['test_accuracies_ADAMCB']
times_ADAMCB = results_dict['times_ADAMCB']
seeds = results_dict['seeds']

train_losses_ADAM_avg = np.mean(train_losses_ADAM, axis=0)
train_losses_ADAM_std = np.std(train_losses_ADAM, axis=0)
test_losses_ADAM_avg = np.mean(test_losses_ADAM, axis=0)
test_losses_ADAM_std = np.std(test_losses_ADAM, axis=0)
train_accuracies_ADAM_avg = np.mean(train_accuracies_ADAM, axis=0)
train_accuracies_ADAM_std = np.std(train_accuracies_ADAM, axis=0)
test_accuracies_ADAM_avg = np.mean(test_accuracies_ADAM, axis=0)
test_accuracies_ADAM_std = np.std(test_accuracies_ADAM, axis=0)
times_ADAM_avg = np.mean(times_ADAM, axis=0)

train_losses_AMSGrad_avg = np.mean(train_losses_AMSGrad, axis=0)
train_losses_AMSGrad_std = np.std(train_losses_AMSGrad, axis=0)
test_losses_AMSGrad_avg = np.mean(test_losses_AMSGrad, axis=0)
test_losses_AMSGrad_std = np.std(test_losses_AMSGrad, axis=0)
train_accuracies_AMSGrad_avg = np.mean(train_accuracies_AMSGrad, axis=0)
train_accuracies_AMSGrad_std = np.std(train_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_avg = np.mean(test_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_std = np.std(test_accuracies_AMSGrad, axis=0)
times_AMSGrad_avg = np.mean(times_AMSGrad, axis=0)

train_losses_ADAMX_avg = np.mean(train_losses_ADAMX, axis=0)
train_losses_ADAMX_std = np.std(train_losses_ADAMX, axis=0)
test_losses_ADAMX_avg = np.mean(test_losses_ADAMX, axis=0)
test_losses_ADAMX_std = np.std(test_losses_ADAMX, axis=0)
train_accuracies_ADAMX_avg = np.mean(train_accuracies_ADAMX, axis=0)
train_accuracies_ADAMX_std = np.std(train_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_avg = np.mean(test_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_std = np.std(test_accuracies_ADAMX, axis=0)
times_ADAMX_avg = np.mean(times_ADAMX, axis=0)

train_losses_ADAMBS_avg = np.mean(train_losses_ADAMBS, axis=0)
train_losses_ADAMBS_std = np.std(train_losses_ADAMBS, axis=0)
test_losses_ADAMBS_avg = np.mean(test_losses_ADAMBS, axis=0)
test_losses_ADAMBS_std = np.std(test_losses_ADAMBS, axis=0)
train_accuracies_ADAMBS_avg = np.mean(train_accuracies_ADAMBS, axis=0)
train_accuracies_ADAMBS_std = np.std(train_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_avg = np.mean(test_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_std = np.std(test_accuracies_ADAMBS, axis=0)
times_ADAMBS_avg = np.mean(times_ADAMBS, axis=0)

train_losses_ADAMBS_corrected_avg = np.mean(train_losses_ADAMBS_corrected, axis=0)
train_losses_ADAMBS_corrected_std = np.std(train_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_avg = np.mean(test_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_std = np.std(test_losses_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_avg = np.mean(train_accuracies_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_std = np.std(train_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_avg = np.mean(test_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_std = np.std(test_accuracies_ADAMBS_corrected, axis=0)
times_ADAMBS_corrected_avg = np.mean(times_ADAMBS_corrected, axis=0)

train_losses_ADAMCB_avg = np.mean(train_losses_ADAMCB, axis=0)
train_losses_ADAMCB_std = np.std(train_losses_ADAMCB, axis=0)
test_losses_ADAMCB_avg = np.mean(test_losses_ADAMCB, axis=0)
test_losses_ADAMCB_std = np.std(test_losses_ADAMCB, axis=0)
train_accuracies_ADAMCB_avg = np.mean(train_accuracies_ADAMCB, axis=0)
train_accuracies_ADAMCB_std = np.std(train_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_avg = np.mean(test_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_std = np.std(test_accuracies_ADAMCB, axis=0)
times_ADAMCB_avg = np.mean(times_ADAMCB, axis=0)


import numpy as np
import pickle

# Load the existing results dictionary
with open('MLP_FashionMNIST_results.pkl', 'rb') as file:
    results_dict = pickle.load(file)
    
train_losses_ADAM = results_dict['train_losses_ADAM']
test_losses_ADAM = results_dict['test_losses_ADAM']
train_accuracies_ADAM = results_dict['train_accuracies_ADAM']
test_accuracies_ADAM = results_dict['test_accuracies_ADAM']
times_ADAM = results_dict['times_ADAM']
train_losses_AMSGrad = results_dict['train_losses_AMSGrad']
test_losses_AMSGrad = results_dict['test_losses_AMSGrad']
train_accuracies_AMSGrad = results_dict['train_accuracies_AMSGrad']
test_accuracies_AMSGrad = results_dict['test_accuracies_AMSGrad']
times_AMSGrad = results_dict['times_AMSGrad']
train_losses_ADAMX = results_dict['train_losses_ADAMX']
test_losses_ADAMX = results_dict['test_losses_ADAMX']
train_accuracies_ADAMX = results_dict['train_accuracies_ADAMX']
test_accuracies_ADAMX = results_dict['test_accuracies_ADAMX']
times_ADAMX = results_dict['times_ADAMX']
train_losses_ADAMBS = results_dict['train_losses_ADAMBS']
test_losses_ADAMBS = results_dict['test_losses_ADAMBS']
train_accuracies_ADAMBS = results_dict['train_accuracies_ADAMBS']
test_accuracies_ADAMBS = results_dict['test_accuracies_ADAMBS']
times_ADAMBS = results_dict['times_ADAMBS']
train_losses_ADAMBS_corrected = results_dict['train_losses_ADAMBS_corrected']
test_losses_ADAMBS_corrected = results_dict['test_losses_ADAMBS_corrected']
train_accuracies_ADAMBS_corrected = results_dict['train_accuracies_ADAMBS_corrected']
test_accuracies_ADAMBS_corrected = results_dict['test_accuracies_ADAMBS_corrected']
times_ADAMBS_corrected = results_dict['times_ADAMBS_corrected']
train_losses_ADAMCB = results_dict['train_losses_ADAMCB']
test_losses_ADAMCB = results_dict['test_losses_ADAMCB']
train_accuracies_ADAMCB = results_dict['train_accuracies_ADAMCB']
test_accuracies_ADAMCB = results_dict['test_accuracies_ADAMCB']
times_ADAMCB = results_dict['times_ADAMCB']
seeds = results_dict['seeds']


train_losses_ADAM_avg_1 = np.mean(train_losses_ADAM, axis=0)
train_losses_ADAM_std_1 = np.std(train_losses_ADAM, axis=0)
test_losses_ADAM_avg_1 = np.mean(test_losses_ADAM, axis=0)
test_losses_ADAM_std_1 = np.std(test_losses_ADAM, axis=0)
train_accuracies_ADAM_avg_1 = np.mean(train_accuracies_ADAM, axis=0)
train_accuracies_ADAM_std_1 = np.std(train_accuracies_ADAM, axis=0)
test_accuracies_ADAM_avg_1 = np.mean(test_accuracies_ADAM, axis=0)
test_accuracies_ADAM_std_1 = np.std(test_accuracies_ADAM, axis=0)
times_ADAM_avg_1 = np.mean(times_ADAM, axis=0)

train_losses_AMSGrad_avg_1 = np.mean(train_losses_AMSGrad, axis=0)
train_losses_AMSGrad_std_1 = np.std(train_losses_AMSGrad, axis=0)
test_losses_AMSGrad_avg_1 = np.mean(test_losses_AMSGrad, axis=0)
test_losses_AMSGrad_std_1 = np.std(test_losses_AMSGrad, axis=0)
train_accuracies_AMSGrad_avg_1 = np.mean(train_accuracies_AMSGrad, axis=0)
train_accuracies_AMSGrad_std_1 = np.std(train_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_avg_1 = np.mean(test_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_std_1 = np.std(test_accuracies_AMSGrad, axis=0)
times_AMSGrad_avg_1 = np.mean(times_AMSGrad, axis=0)

train_losses_ADAMX_avg_1 = np.mean(train_losses_ADAMX, axis=0)
train_losses_ADAMX_std_1 = np.std(train_losses_ADAMX, axis=0)
test_losses_ADAMX_avg_1 = np.mean(test_losses_ADAMX, axis=0)
test_losses_ADAMX_std_1 = np.std(test_losses_ADAMX, axis=0)
train_accuracies_ADAMX_avg_1 = np.mean(train_accuracies_ADAMX, axis=0)
train_accuracies_ADAMX_std_1 = np.std(train_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_avg_1 = np.mean(test_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_std_1 = np.std(test_accuracies_ADAMX, axis=0)
times_ADAMX_avg_1 = np.mean(times_ADAMX, axis=0)

train_losses_ADAMBS_avg_1 = np.mean(train_losses_ADAMBS, axis=0)
train_losses_ADAMBS_std_1 = np.std(train_losses_ADAMBS, axis=0)
test_losses_ADAMBS_avg_1 = np.mean(test_losses_ADAMBS, axis=0)
test_losses_ADAMBS_std_1 = np.std(test_losses_ADAMBS, axis=0)
train_accuracies_ADAMBS_avg_1 = np.mean(train_accuracies_ADAMBS, axis=0)
train_accuracies_ADAMBS_std_1 = np.std(train_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_avg_1 = np.mean(test_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_std_1 = np.std(test_accuracies_ADAMBS, axis=0)
times_ADAMBS_avg_1 = np.mean(times_ADAMBS, axis=0)

train_losses_ADAMBS_corrected_avg_1 = np.mean(train_losses_ADAMBS_corrected, axis=0)
train_losses_ADAMBS_corrected_std_1 = np.std(train_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_avg_1 = np.mean(test_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_std_1 = np.std(test_losses_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_avg_1 = np.mean(train_accuracies_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_std_1 = np.std(train_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_avg_1 = np.mean(test_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_std_1 = np.std(test_accuracies_ADAMBS_corrected, axis=0)
times_ADAMBS_corrected_avg_1 = np.mean(times_ADAMBS_corrected, axis=0)

train_losses_ADAMCB_avg_1 = np.mean(train_losses_ADAMCB, axis=0)
train_losses_ADAMCB_std_1 = np.std(train_losses_ADAMCB, axis=0)
test_losses_ADAMCB_avg_1 = np.mean(test_losses_ADAMCB, axis=0)
test_losses_ADAMCB_std_1 = np.std(test_losses_ADAMCB, axis=0)
train_accuracies_ADAMCB_avg_1 = np.mean(train_accuracies_ADAMCB, axis=0)
train_accuracies_ADAMCB_std_1 = np.std(train_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_avg_1 = np.mean(test_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_std_1 = np.std(test_accuracies_ADAMCB, axis=0)
times_ADAMCB_avg_1 = np.mean(times_ADAMCB, axis=0)


import numpy as np
import pickle

# Load the existing results dictionary
with open('MLP_CIFAR10_results.pkl', 'rb') as file:
    results_dict = pickle.load(file)

train_losses_ADAM = results_dict['train_losses_ADAM']
test_losses_ADAM = results_dict['test_losses_ADAM']
train_accuracies_ADAM = results_dict['train_accuracies_ADAM']
test_accuracies_ADAM = results_dict['test_accuracies_ADAM']
times_ADAM = results_dict['times_ADAM']
train_losses_AMSGrad = results_dict['train_losses_AMSGrad']
test_losses_AMSGrad = results_dict['test_losses_AMSGrad']
train_accuracies_AMSGrad = results_dict['train_accuracies_AMSGrad']
test_accuracies_AMSGrad = results_dict['test_accuracies_AMSGrad']
times_AMSGrad = results_dict['times_AMSGrad']
train_losses_ADAMX = results_dict['train_losses_ADAMX']
test_losses_ADAMX = results_dict['test_losses_ADAMX']
train_accuracies_ADAMX = results_dict['train_accuracies_ADAMX']
test_accuracies_ADAMX = results_dict['test_accuracies_ADAMX']
times_ADAMX = results_dict['times_ADAMX']
train_losses_ADAMBS = results_dict['train_losses_ADAMBS']
test_losses_ADAMBS = results_dict['test_losses_ADAMBS']
train_accuracies_ADAMBS = results_dict['train_accuracies_ADAMBS']
test_accuracies_ADAMBS = results_dict['test_accuracies_ADAMBS']
times_ADAMBS = results_dict['times_ADAMBS']
train_losses_ADAMBS_corrected = results_dict['train_losses_ADAMBS_corrected']
test_losses_ADAMBS_corrected = results_dict['test_losses_ADAMBS_corrected']
train_accuracies_ADAMBS_corrected = results_dict['train_accuracies_ADAMBS_corrected']
test_accuracies_ADAMBS_corrected = results_dict['test_accuracies_ADAMBS_corrected']
times_ADAMBS_corrected = results_dict['times_ADAMBS_corrected']
train_losses_ADAMCB = results_dict['train_losses_ADAMCB']
test_losses_ADAMCB = results_dict['test_losses_ADAMCB']
train_accuracies_ADAMCB = results_dict['train_accuracies_ADAMCB']
test_accuracies_ADAMCB = results_dict['test_accuracies_ADAMCB']
times_ADAMCB = results_dict['times_ADAMCB']
seeds = results_dict['seeds']


train_losses_ADAM_avg_2 = np.mean(train_losses_ADAM, axis=0)
train_losses_ADAM_std_2 = np.std(train_losses_ADAM, axis=0)
test_losses_ADAM_avg_2 = np.mean(test_losses_ADAM, axis=0)
test_losses_ADAM_std_2 = np.std(test_losses_ADAM, axis=0)
train_accuracies_ADAM_avg_2 = np.mean(train_accuracies_ADAM, axis=0)
train_accuracies_ADAM_std_2 = np.std(train_accuracies_ADAM, axis=0)
test_accuracies_ADAM_avg_2 = np.mean(test_accuracies_ADAM, axis=0)
test_accuracies_ADAM_std_2 = np.std(test_accuracies_ADAM, axis=0)
times_ADAM_avg_2 = np.mean(times_ADAM, axis=0)

train_losses_AMSGrad_avg_2 = np.mean(train_losses_AMSGrad, axis=0)
train_losses_AMSGrad_std_2 = np.std(train_losses_AMSGrad, axis=0)
test_losses_AMSGrad_avg_2 = np.mean(test_losses_AMSGrad, axis=0)
test_losses_AMSGrad_std_2 = np.std(test_losses_AMSGrad, axis=0)
train_accuracies_AMSGrad_avg_2 = np.mean(train_accuracies_AMSGrad, axis=0)
train_accuracies_AMSGrad_std_2 = np.std(train_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_avg_2 = np.mean(test_accuracies_AMSGrad, axis=0)
test_accuracies_AMSGrad_std_2 = np.std(test_accuracies_AMSGrad, axis=0)
times_AMSGrad_avg_2 = np.mean(times_AMSGrad, axis=0)

train_losses_ADAMX_avg_2 = np.mean(train_losses_ADAMX, axis=0)
train_losses_ADAMX_std_2 = np.std(train_losses_ADAMX, axis=0)
test_losses_ADAMX_avg_2 = np.mean(test_losses_ADAMX, axis=0)
test_losses_ADAMX_std_2 = np.std(test_losses_ADAMX, axis=0)
train_accuracies_ADAMX_avg_2 = np.mean(train_accuracies_ADAMX, axis=0)
train_accuracies_ADAMX_std_2 = np.std(train_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_avg_2 = np.mean(test_accuracies_ADAMX, axis=0)
test_accuracies_ADAMX_std_2 = np.std(test_accuracies_ADAMX, axis=0)
times_ADAMX_avg_2 = np.mean(times_ADAMX, axis=0)

train_losses_ADAMBS_avg_2 = np.mean(train_losses_ADAMBS, axis=0)
train_losses_ADAMBS_std_2 = np.std(train_losses_ADAMBS, axis=0)
test_losses_ADAMBS_avg_2 = np.mean(test_losses_ADAMBS, axis=0)
test_losses_ADAMBS_std_2 = np.std(test_losses_ADAMBS, axis=0)
train_accuracies_ADAMBS_avg_2 = np.mean(train_accuracies_ADAMBS, axis=0)
train_accuracies_ADAMBS_std_2 = np.std(train_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_avg_2 = np.mean(test_accuracies_ADAMBS, axis=0)
test_accuracies_ADAMBS_std_2 = np.std(test_accuracies_ADAMBS, axis=0)
times_ADAMBS_avg_2 = np.mean(times_ADAMBS, axis=0)

train_losses_ADAMBS_corrected_avg_2 = np.mean(train_losses_ADAMBS_corrected, axis=0)
train_losses_ADAMBS_corrected_std_2 = np.std(train_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_avg_2 = np.mean(test_losses_ADAMBS_corrected, axis=0)
test_losses_ADAMBS_corrected_std_2 = np.std(test_losses_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_avg_2 = np.mean(train_accuracies_ADAMBS_corrected, axis=0)
train_accuracies_ADAMBS_corrected_std_2 = np.std(train_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_avg_2 = np.mean(test_accuracies_ADAMBS_corrected, axis=0)
test_accuracies_ADAMBS_corrected_std_2 = np.std(test_accuracies_ADAMBS_corrected, axis=0)
times_ADAMBS_corrected_avg_2 = np.mean(times_ADAMBS_corrected, axis=0)

train_losses_ADAMCB_avg_2 = np.mean(train_losses_ADAMCB, axis=0)
train_losses_ADAMCB_std_2 = np.std(train_losses_ADAMCB, axis=0)
test_losses_ADAMCB_avg_2 = np.mean(test_losses_ADAMCB, axis=0)
test_losses_ADAMCB_std_2 = np.std(test_losses_ADAMCB, axis=0)
train_accuracies_ADAMCB_avg_2 = np.mean(train_accuracies_ADAMCB, axis=0)
train_accuracies_ADAMCB_std_2 = np.std(train_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_avg_2 = np.mean(test_accuracies_ADAMCB, axis=0)
test_accuracies_ADAMCB_std_2 = np.std(test_accuracies_ADAMCB, axis=0)
times_ADAMCB_avg_2 = np.mean(times_ADAMCB, axis=0)


import matplotlib.pyplot as plt

# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 8))

# Plot for MNIST dataset
line1, = axs[0, 0].plot(train_losses_ADAM_avg, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_ADAM_avg)), train_losses_ADAM_avg - train_losses_ADAM_std, train_losses_ADAM_avg + train_losses_ADAM_std, color='#2ca02c', alpha=0.2)
line2, = axs[0, 0].plot(train_losses_ADAMBS_avg, "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_ADAMBS_avg)), train_losses_ADAMBS_avg - train_losses_ADAMBS_std, train_losses_ADAMBS_avg + train_losses_ADAMBS_std, color='#1f77b4', alpha=0.2)
line3, = axs[0, 0].plot(train_losses_ADAMCB_avg, "^-",label='AdamCB (ours)', color='#d62728', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_ADAMCB_avg)), train_losses_ADAMCB_avg - train_losses_ADAMCB_std, train_losses_ADAMCB_avg + train_losses_ADAMCB_std, color='#d62728', alpha=0.2)
line4, = axs[0, 0].plot(train_losses_AMSGrad_avg, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_AMSGrad_avg)), train_losses_AMSGrad_avg - train_losses_AMSGrad_std, train_losses_AMSGrad_avg + train_losses_AMSGrad_std, color='#ff7f0e', alpha=0.2)
line5, = axs[0, 0].plot(train_losses_ADAMX_avg, "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_ADAMX_avg)), train_losses_ADAMX_avg - train_losses_ADAMX_std, train_losses_ADAMX_avg + train_losses_ADAMX_std, color='#9467bd', alpha=0.2)
line6, = axs[0, 0].plot(train_losses_ADAMBS_corrected_avg, "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[0, 0].fill_between(range(len(train_losses_ADAMBS_corrected_avg)), train_losses_ADAMBS_corrected_avg - train_losses_ADAMBS_corrected_std, train_losses_ADAMBS_corrected_avg + train_losses_ADAMBS_corrected_std, color='#8c564b', alpha=0.2)
axs[0, 0].set_title("(a) MNIST", fontsize=16)
axs[0, 0].set_xlabel("Epochs", fontsize=14)
axs[0, 0].set_ylim(0, 2.4)
axs[0, 0].set_ylabel("Train Loss", fontsize=14)
axs[0, 0].grid(True)

# Test loss for MNIST
axs[1, 0].plot(test_losses_ADAM_avg, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_ADAM_avg)), test_losses_ADAM_avg - test_losses_ADAM_std, test_losses_ADAM_avg + test_losses_ADAM_std, color='#2ca02c', alpha=0.2)
axs[1, 0].plot(test_losses_ADAMBS_avg, "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_ADAMBS_avg)), test_losses_ADAMBS_avg - test_losses_ADAMBS_std, test_losses_ADAMBS_avg + test_losses_ADAMBS_std, color='#1f77b4', alpha=0.2)
axs[1, 0].plot(test_losses_ADAMCB_avg, "^-",label='AdamCB (ours)', color='#d62728', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_ADAMCB_avg)), test_losses_ADAMCB_avg - test_losses_ADAMCB_std, test_losses_ADAMCB_avg + test_losses_ADAMCB_std, color='#d62728', alpha=0.2)
axs[1, 0].plot(test_losses_AMSGrad_avg, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_AMSGrad_avg)), test_losses_AMSGrad_avg - test_losses_AMSGrad_std, test_losses_AMSGrad_avg + test_losses_AMSGrad_std, color='#ff7f0e', alpha=0.2)
axs[1, 0].plot(test_losses_ADAMX_avg, "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_ADAMX_avg)), test_losses_ADAMX_avg - test_losses_ADAMX_std, test_losses_ADAMX_avg + test_losses_ADAMX_std, color='#9467bd', alpha=0.2)
axs[1, 0].plot(test_losses_ADAMBS_corrected_avg, "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[1, 0].fill_between(range(len(test_losses_ADAMBS_corrected_avg)), test_losses_ADAMBS_corrected_avg - test_losses_ADAMBS_corrected_std, test_losses_ADAMBS_corrected_avg + test_losses_ADAMBS_corrected_std, color='#8c564b', alpha=0.2)
axs[1, 0].set_xlabel("Epochs", fontsize=14)
axs[1, 0].set_ylim(0, 2.4)
axs[1, 0].set_ylabel("Test Loss", fontsize=14)
axs[1, 0].grid(True)

# Plot for Fashion MNIST dataset (b)
axs[0, 1].plot(train_losses_ADAM_avg_1, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_ADAM_avg_1)), train_losses_ADAM_avg_1 - train_losses_ADAM_std_1, train_losses_ADAM_avg_1 + train_losses_ADAM_std_1, color='#2ca02c', alpha=0.2)
axs[0, 1].plot(train_losses_ADAMBS_avg_1, "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_ADAMBS_avg_1)), train_losses_ADAMBS_avg_1 - train_losses_ADAMBS_std_1, train_losses_ADAMBS_avg_1 + train_losses_ADAMBS_std_1, color='#1f77b4', alpha=0.2)
axs[0, 1].plot(train_losses_ADAMCB_avg_1, "^-",label='AdamCB (ours)', color='#d62728', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_ADAMCB_avg_1)), train_losses_ADAMCB_avg_1 - train_losses_ADAMCB_std_1, train_losses_ADAMCB_avg_1 + train_losses_ADAMCB_std_1, color='#d62728', alpha=0.2)
axs[0, 1].plot(train_losses_AMSGrad_avg_1, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_AMSGrad_avg_1)), train_losses_AMSGrad_avg_1 - train_losses_AMSGrad_std_1, train_losses_AMSGrad_avg_1 + train_losses_AMSGrad_std_1, color='#ff7f0e', alpha=0.2)
axs[0, 1].plot(train_losses_ADAMX_avg_1, "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_ADAMX_avg_1)), train_losses_ADAMX_avg_1 - train_losses_ADAMX_std_1, train_losses_ADAMX_avg_1 + train_losses_ADAMX_std_1, color='#9467bd', alpha=0.2)
axs[0, 1].plot(train_losses_ADAMBS_corrected_avg_1, "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[0, 1].fill_between(range(len(train_losses_ADAMBS_corrected_avg_1)), train_losses_ADAMBS_corrected_avg_1 - train_losses_ADAMBS_corrected_std_1, train_losses_ADAMBS_corrected_avg_1 + train_losses_ADAMBS_corrected_std_1, color='#8c564b', alpha=0.2)
axs[0, 1].set_title("(b) Fashion MNIST", fontsize=16)
axs[0, 1].set_xlabel("Epochs", fontsize=14)
axs[0, 1].set_ylim(0.3, 2.4)
#axs[0, 1].set_ylabel("Training Loss")
axs[0, 1].grid(True)

# Test loss for Fashion MNIST
axs[1, 1].plot(test_losses_ADAM_avg_1, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_ADAM_avg_1)), test_losses_ADAM_avg_1 - test_losses_ADAM_std_1, test_losses_ADAM_avg_1 + test_losses_ADAM_std_1, color='#2ca02c', alpha=0.2)
axs[1, 1].plot(test_losses_ADAMBS_avg_1, "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_ADAMBS_avg_1)), test_losses_ADAMBS_avg_1 - test_losses_ADAMBS_std_1, test_losses_ADAMBS_avg_1 + test_losses_ADAMBS_std_1, color='#1f77b4', alpha=0.2)
axs[1, 1].plot(test_losses_ADAMCB_avg_1, "^-",label='AdamCB (ours)', color='#d62728', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_ADAMCB_avg_1)), test_losses_ADAMCB_avg_1 - test_losses_ADAMCB_std_1, test_losses_ADAMCB_avg_1 + test_losses_ADAMCB_std_1, color='#d62728', alpha=0.2)
axs[1, 1].plot(test_losses_AMSGrad_avg_1, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_AMSGrad_avg_1)), test_losses_AMSGrad_avg_1 - test_losses_AMSGrad_std_1, test_losses_AMSGrad_avg_1 + test_losses_AMSGrad_std_1, color='#ff7f0e', alpha=0.2)
axs[1, 1].plot(test_losses_ADAMX_avg_1 , "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_ADAMX_avg_1)), test_losses_ADAMX_avg_1 - test_losses_ADAMX_std_1, test_losses_ADAMX_avg_1 + test_losses_ADAMX_std_1, color='#9467bd', alpha=0.2)
axs[1, 1].plot(test_losses_ADAMBS_corrected_avg_1 , "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[1, 1].fill_between(range(len(test_losses_ADAMBS_corrected_avg_1)), test_losses_ADAMBS_corrected_avg_1 - test_losses_ADAMBS_corrected_std_1, test_losses_ADAMBS_corrected_avg_1 + test_losses_ADAMBS_corrected_std_1, color='#8c564b', alpha=0.2)
axs[1, 1].set_xlabel("Epochs", fontsize=14)
axs[1, 1].set_ylim(0.3, 2.4)
#axs[1, 1].set_ylabel("Test Loss")
axs[1, 1].grid(True)

# Plot for CIFAR-10 dataset (c)
axs[0, 2].plot(train_losses_ADAM_avg_2, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_ADAM_avg_2)), train_losses_ADAM_avg_2 - train_losses_ADAM_std_2, train_losses_ADAM_avg_2 + train_losses_ADAM_std_2, color='#2ca02c', alpha=0.2)
axs[0, 2].plot(train_losses_ADAMBS_avg_2, "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_ADAMBS_avg_2)), train_losses_ADAMBS_avg_2 - train_losses_ADAMBS_std_2, train_losses_ADAMBS_avg_2 + train_losses_ADAMBS_std_2, color='#1f77b4', alpha=0.2)
axs[0, 2].plot(train_losses_ADAMBS_corrected_avg_2, "^-",label='AdamCB (ours)', color='#d62728', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_ADAMBS_corrected_avg_2)), train_losses_ADAMBS_corrected_avg_2 - train_losses_ADAMBS_corrected_std_2, train_losses_ADAMCB_avg_2 + train_losses_ADAMCB_std_2, color='#d62728', alpha=0.2)
axs[0, 2].plot(train_losses_AMSGrad_avg_2, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_AMSGrad_avg_2)), train_losses_AMSGrad_avg_2 - train_losses_AMSGrad_std_2, train_losses_AMSGrad_avg_2 + train_losses_AMSGrad_std_2, color='#ff7f0e', alpha=0.2)
axs[0, 2].plot(train_losses_ADAMX_avg_2, "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_ADAMX_avg_2)), train_losses_ADAMX_avg_2 - train_losses_ADAMX_std_2, train_losses_ADAMX_avg_2 + train_losses_ADAMX_std_2, color='#9467bd', alpha=0.2)
axs[0, 2].plot(train_losses_ADAMCB_avg_2, "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[0, 2].fill_between(range(len(train_losses_ADAMCB_avg_2)), train_losses_ADAMCB_avg_2 - train_losses_ADAMCB_std_2, train_losses_ADAMCB_avg_2 + train_losses_ADAMCB_std_2, color='#8c564b', alpha=0.2)
axs[0, 2].set_title("(c) CIFAR-10", fontsize=16)
axs[0, 2].set_xlabel("Epochs", fontsize=14)
axs[0, 2].set_ylim(1.3, 2.4)
#axs[0, 2].set_ylabel("Training Loss")
axs[0, 2].grid(True)

# Test loss for CIFAR-10
axs[1, 2].plot(test_losses_ADAMCB_avg_2, "s-", label='Adam', color='#2ca02c', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_ADAMCB_avg_2)), test_losses_ADAMCB_avg_2 - test_losses_ADAMCB_std_2, test_losses_ADAMCB_avg_2 + test_losses_ADAMCB_std_2, color='#2ca02c', alpha=0.2)
axs[1, 2].plot(test_losses_ADAMBS_avg_2 , "o-", label='AdamBS', color='#1f77b4', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_ADAMBS_avg_2)), test_losses_ADAMBS_avg_2 - test_losses_ADAMBS_std_2, test_losses_ADAMBS_avg_2 + test_losses_ADAMBS_std_2, color='#1f77b4', alpha=0.2)
axs[1, 2].plot(test_losses_ADAMBS_corrected_avg_2, "^-", label='AdamCB (ours)', color='#d62728', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_ADAMBS_corrected_avg_2)), test_losses_ADAMBS_corrected_avg_2 - test_losses_ADAMBS_corrected_std_2, test_losses_ADAMBS_corrected_avg_2 + test_losses_ADAMBS_corrected_std_2, color='#d62728', alpha=0.2)
axs[1, 2].plot(test_losses_AMSGrad_avg_2, "x-", label='AMSGrad', color='#ff7f0e', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_AMSGrad_avg_2)), test_losses_AMSGrad_avg_2 - test_losses_AMSGrad_std_2, test_losses_AMSGrad_avg_2 + test_losses_AMSGrad_std_2, color='#ff7f0e', alpha=0.2)
axs[1, 2].plot(test_losses_ADAMX_avg_2, "d-", label='Adam (corrected)', color='#9467bd', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_ADAMX_avg_2)), test_losses_ADAMX_avg_2 - test_losses_ADAMX_std_2, test_losses_ADAMX_avg_2 + test_losses_ADAMX_std_2, color='#9467bd', alpha=0.2)
axs[1, 2].plot(test_losses_ADAM_avg_2, "v-", label='AdamBS (corrected)', color='#8c564b', markersize=4)
axs[1, 2].fill_between(range(len(test_losses_ADAM_avg_2)), test_losses_ADAM_avg_2 - test_losses_ADAM_std_2, color='#8c564b', alpha=0.2)
axs[1, 2].set_xlabel("Epochs", fontsize=14)
axs[1, 2].set_ylim(1.3, 2.3)
#axs[1, 2].set_ylabel("Test Loss")
axs[1, 2].grid(True)

# Add a single legend for all plots
fig.legend(handles=[line1, line5, line2, line6, line4, line3], loc='upper center', ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.05))
# Show plot
# Save the figure with tight bounding box to avoid cut-off
plt.savefig('plot_results_MLP_all_1.pdf', format='pdf', bbox_inches='tight')
plt.show()