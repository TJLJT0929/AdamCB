import torch
import torch.nn as nn
from torchvision import datasets, transforms
from utils import set_seed
from models import MNIST_MLP, MNIST_CNN, MNIST_VGG
from train import ADAM, ADAMX, AMSGrad, ADAMBS, ADAMBS_corrected, ADAMCB
import torchvision.transforms as transforms
import pickle


# Transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizes the dataset for mean and std deviation of MNIST
])

# Loading the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

loss_function = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = 128
n = len(train_dataset)
one_epoch = n // K
#one_epoch = 30
max_iter = 10 * one_epoch
seeds = [10, 20, 30, 40, 50]


train_losses_ADAM, test_losses_ADAM, train_accuracies_ADAM, test_accuracies_ADAM, times_ADAM = [], [], [], [], []
train_losses_AMSGrad, test_losses_AMSGrad, train_accuracies_AMSGrad, test_accuracies_AMSGrad, times_AMSGrad = [], [], [], [], []
train_losses_ADAMX, test_losses_ADAMX, train_accuracies_ADAMX, test_accuracies_ADAMX, times_ADAMX = [], [], [], [], []
train_losses_ADAMBS, test_losses_ADAMBS, train_accuracies_ADAMBS, test_accuracies_ADAMBS, times_ADAMBS = [], [], [], [], []
train_losses_ADAMBS_corrected, test_losses_ADAMBS_corrected, train_accuracies_ADAMBS_corrected, test_accuracies_ADAMBS_corrected, times_ADAMBS_corrected = [], [], [], [], []
train_losses_ADAMCB, test_losses_ADAMCB, train_accuracies_ADAMCB, test_accuracies_ADAMCB, times_ADAMCB = [], [], [], [], []

for i, j in enumerate(seeds):
    print("Repetition: {} starts!".format(i+1))
    print("Seed: {}".format(j))

    print("Training by ADAM starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_ADAM, test_loss_ADAM, train_accuracy_ADAM, test_accuracy_ADAM, time_ADAM = ADAM(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_ADAM.append(train_loss_ADAM)
    test_losses_ADAM.append(test_loss_ADAM)
    train_accuracies_ADAM.append(train_accuracy_ADAM)
    test_accuracies_ADAM.append(test_accuracy_ADAM)
    times_ADAM.append(time_ADAM)

    print("Training by AMSGrad starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_AMSGrad, test_loss_AMSGrad, train_accuracy_AMSGrad, test_accuracy_AMSGrad, time_AMSGrad = AMSGrad(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_AMSGrad.append(train_loss_AMSGrad)
    test_losses_AMSGrad.append(test_loss_AMSGrad)
    train_accuracies_AMSGrad.append(train_accuracy_AMSGrad)
    test_accuracies_AMSGrad.append(test_accuracy_AMSGrad)
    times_AMSGrad.append(time_AMSGrad)

    print("Training by ADAMX starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_ADAMX, test_loss_ADAMX, train_accuracy_ADAMX, test_accuracy_ADAMX, time_ADAMX = ADAMX(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_ADAMX.append(train_loss_ADAMX)
    test_losses_ADAMX.append(test_loss_ADAMX)
    train_accuracies_ADAMX.append(train_accuracy_ADAMX)
    test_accuracies_ADAMX.append(test_accuracy_ADAMX)
    times_ADAMX.append(time_ADAMX)

    print("Training by ADAMBS starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_ADAMBS, test_loss_ADAMBS, train_accuracy_ADAMBS, test_accuracy_ADAMBS, time_ADAMBS = ADAMBS(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_ADAMBS.append(train_loss_ADAMBS)
    test_losses_ADAMBS.append(test_loss_ADAMBS)
    train_accuracies_ADAMBS.append(train_accuracy_ADAMBS)
    test_accuracies_ADAMBS.append(test_accuracy_ADAMBS)
    times_ADAMBS.append(time_ADAMBS)
   
    print("Training by ADAMBS_corrected starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_ADAMBS_corrected, test_loss_ADAMBS_corrected, train_accuracy_ADAMBS_corrected, test_accuracy_ADAMBS_corrected, time_ADAMBS_corrected = ADAMBS_corrected(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_ADAMBS_corrected.append(train_loss_ADAMBS_corrected)
    test_losses_ADAMBS_corrected.append(test_loss_ADAMBS_corrected)
    train_accuracies_ADAMBS_corrected.append(train_accuracy_ADAMBS_corrected)
    test_accuracies_ADAMBS_corrected.append(test_accuracy_ADAMBS_corrected)
    times_ADAMBS_corrected.append(time_ADAMBS_corrected)
    
    print("Training by ADAMCB starts!")
    set_seed(j)
    model = MNIST_MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_ADAMCB, test_loss_ADAMCB, train_accuracy_ADAMCB, test_accuracy_ADAMCB, time_ADAMCB = ADAMCB(model, loss_function, train_dataset, test_dataset, K, max_iter, device)
    train_losses_ADAMCB.append(train_loss_ADAMCB)
    test_losses_ADAMCB.append(test_loss_ADAMCB)
    train_accuracies_ADAMCB.append(train_accuracy_ADAMCB)
    test_accuracies_ADAMCB.append(test_accuracy_ADAMCB)
    times_ADAMCB.append(time_ADAMCB)
    
    print("Repetition: {} finished!".format(i+1))
    

results_dict = {
    'train_losses_ADAM': train_losses_ADAM,
    'test_losses_ADAM': test_losses_ADAM,
    'train_accuracies_ADAM': train_accuracies_ADAM,
    'test_accuracies_ADAM': test_accuracies_ADAM,
    'times_ADAM': times_ADAM,
    'train_losses_AMSGrad': train_losses_AMSGrad,
    'test_losses_AMSGrad': test_losses_AMSGrad,
    'train_accuracies_AMSGrad': train_accuracies_AMSGrad,
    'test_accuracies_AMSGrad': test_accuracies_AMSGrad,
    'times_AMSGrad': times_AMSGrad,
    'train_losses_ADAMX': train_losses_ADAMX,
    'test_losses_ADAMX': test_losses_ADAMX,
    'train_accuracies_ADAMX': train_accuracies_ADAMX,
    'test_accuracies_ADAMX': test_accuracies_ADAMX,
    'times_ADAMX': times_ADAMX,
    'train_losses_ADAMBS': train_losses_ADAMBS,
    'test_losses_ADAMBS': test_losses_ADAMBS,
    'train_accuracies_ADAMBS': train_accuracies_ADAMBS,
    'test_accuracies_ADAMBS': test_accuracies_ADAMBS,
    'times_ADAMBS': times_ADAMBS,
    'train_losses_ADAMBS_corrected': train_losses_ADAMBS_corrected,
    'test_losses_ADAMBS_corrected': test_losses_ADAMBS_corrected,
    'train_accuracies_ADAMBS_corrected': train_accuracies_ADAMBS_corrected,
    'test_accuracies_ADAMBS_corrected': test_accuracies_ADAMBS_corrected,
    'times_ADAMBS_corrected': times_ADAMBS_corrected,
    'train_losses_ADAMCB': train_losses_ADAMCB,
    'test_losses_ADAMCB': test_losses_ADAMCB,
    'train_accuracies_ADAMCB': train_accuracies_ADAMCB,
    'test_accuracies_ADAMCB': test_accuracies_ADAMCB,
    'times_ADAMCB': times_ADAMCB,
    'seeds': seeds
}

with open('MLP_MNIST_results_all_final.pkl', 'wb') as file:
    pickle.dump(results_dict, file)