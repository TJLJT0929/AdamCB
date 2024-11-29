import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def compute_unbiased_gradient_estimate(model, loss_function, train_dataset, indices, p_t, device):
    n = len(train_dataset)
    K = len(indices)
    params = list(model.parameters())

    gradient_norms = []
    #training_loss = 0
    #correct = 0

    # Compute unbiased gradient estimate
    G_t_hat = [torch.zeros_like(p.data, device=device) for p in params]
    for k in indices:
        data, target = train_dataset[k]
        data, target = data.to(device).unsqueeze(0), torch.tensor([target], device=device, dtype=torch.long)
        model.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        grad_norm = 0
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                G_t_hat[i] += param.grad.data / (n * p_t[k])
                grad_norm += (param.grad.data ** 2).sum()
            grad_norm = torch.sqrt(grad_norm)
        gradient_norms.append(grad_norm.item())

        #training_loss += loss.item()
        #_, predicted = torch.max(output.data, 1)
        #correct += (predicted == target).sum().item()
    
    #training_loss /= K
    #accuracy = 100.0 * correct / K
    G_t_hat = [grad / K for grad in G_t_hat]

    return G_t_hat, gradient_norms #, training_loss, accuracy


def evaluate1(model, loss_function, train_dataset, device):  
    test_subset_size = 1000
    test_subset_indices = range(1000)
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        for idx in test_subset_indices:
            data, target = train_dataset[idx]
            data, target = data.to(device).unsqueeze(0), torch.tensor([target], device=device, dtype=torch.long)
            output = model(data)
            test_loss += loss_function(output, target).item()
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == target).sum().item()
    avg_test_loss = test_loss / test_subset_size
    test_accuracy = 100.0 * test_correct / test_subset_size
    return avg_test_loss, test_accuracy


def evaluate2(model, loss_function, train_dataset, device):    
    test_subset_size = 1000
    test_subset_indices = np.random.choice(len(train_dataset), test_subset_size, replace=False)
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        for idx in test_subset_indices:
            data, target = train_dataset[idx]
            data, target = data.to(device).unsqueeze(0), torch.tensor([target], device=device, dtype=torch.long)
            output = model(data)
            test_loss += loss_function(output, target).item()
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == target).sum().item()
    avg_test_loss = test_loss / test_subset_size
    test_accuracy = 100.0 * test_correct / test_subset_size
    return avg_test_loss, test_accuracy