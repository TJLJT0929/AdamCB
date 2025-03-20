import torch
import numpy as np
import time
from tqdm import tqdm
from samplers import batch_selection_adambs, batch_selection_adamcb
from utils import compute_unbiased_gradient_estimate, evaluate1, evaluate2
from optimizers import adam_update, amsgrad_update, adam_corrected_update
from update import update_sample_distribution, update_sample_weights_BS, update_sample_weights_CB


def ADAMCB(model, loss_function, train_dataset, test_dataset, K, max_iter, device, lr=0.001, betas=(0.9, 0.999), eps=1e-8, lambda_ = 1-1e-8):
    n = len(train_dataset)
    one_epoch = n // K
    #one_epoch = 30
    gamma = 0.4
    p_min = gamma / n

    # Initialize model parameters
    params = list(model.parameters())
    # Initialize first moment estimate and second moment estimate
    m_t = [torch.zeros_like(p.data, device=device) for p in params]
    v_t = [torch.zeros_like(p.data, device=device) for p in params]
    v_hat = [torch.zeros_like(p.data, device=device) for p in params]
    # Initialize sample weights
    w_t = np.ones(n)
    
    max_norm, min_norm = float('-inf'), float('inf')
    alpha_p = float(p_min**3)
    train_losses, train_accuracies, test_losses, test_accuracies, times = [], [], [], [], []
    start_time = time.time()
    for t in tqdm(range(1, max_iter + 1), desc="Training Progress"):
        model.train()
        # Select a mini-batch I_t by sampling with replacement from p_t
        indices, p_t, S_null = batch_selection_adamcb(w_t, K, gamma)
        p_t /= np.sum(p_t)
        G_t_hat, gradient_norms = compute_unbiased_gradient_estimate(model, loss_function, train_dataset, indices, p_t, device)
        train_loss, train_accuracy = evaluate1(model, loss_function, train_dataset, device)
        if max(gradient_norms) > max_norm:
            max_norm = max(gradient_norms)
        min_norm = min(min_norm, min(gradient_norms))
        sample_gradient_norms = [(x - min_norm) / (max_norm - min_norm) for x in gradient_norms]
        
        adam_corrected_update(model, G_t_hat, m_t, v_t, v_hat, t, lr, betas, lambda_, eps, device, 0)
        w_t = update_sample_weights_CB(indices, w_t, p_t, sample_gradient_norms, gamma, alpha_p, S_null)
        
        if t == 1 or t % one_epoch == 0:
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            times.append(time.time() - start_time)
            model.eval()
            test_loss, test_accuracy = evaluate2(model, loss_function, test_dataset, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f"Iteration: {t}, Time: {times[-1]:.2f}(s), Train Loss: {train_losses[-1]}, "
                f"Test Loss: {test_losses[-1]}, Train Accuracy: {train_accuracies[-1]}, "
                f"Test Accuracy: {test_accuracies[-1]}")

    return train_losses, test_losses, train_accuracies, test_accuracies, times
