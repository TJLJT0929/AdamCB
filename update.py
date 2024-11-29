import numpy as np

def update_sample_weights_BS(indices, w_t, p_t, sample_gradient_norms, gamma, alpha_p):
    n = len(p_t)
    K = len(indices)
    p_min = gamma / n
    h_hat = np.zeros(n)
    for i, idx in enumerate(indices):
        loss = - (sample_gradient_norms[i]**2 / p_t[idx]**2) + 1.0 / p_min**2
        h_hat[idx] += loss / (K*p_t[idx])
    w_t = w_t * np.exp(-alpha_p * h_hat)
    return w_t


def update_sample_weights_CB(indices, w_t, p_t, sample_gradient_norms, gamma, alpha_p, S_null):
    n = len(p_t)
    K = len(indices)
    p_min = gamma / n
    w_temp = w_t.copy()
    h_hat = np.zeros(n)
    for i, idx in enumerate(indices):
        loss = - (sample_gradient_norms[i]**2 / p_t[idx]**2) + 1.0 / p_min**2
        h_hat[idx] = loss / p_t[idx]
    w_t = w_temp * np.exp(-alpha_p * h_hat)
    for i in S_null:
        w_t[i] = w_temp[i]
    return w_t


def update_sample_distribution(indices, p_t, sample_gradient_norms, gamma, L, alpha_p):
    n = len(p_t)
    K = len(indices)
    p_min = gamma / n
    
    h_hat = np.zeros(n)
    for i, idx in enumerate(indices):
        loss = - (sample_gradient_norms[i]**2 / p_t[idx]**2) + (L**2 / p_min**2)
        h_hat[idx] += loss / (K*p_t[idx])

    # Apply clipping to prevent overflow in exp()
    #print(max(-alpha_p * h_hat), min(-alpha_p * h_hat))
    scaled_h_hat = -alpha_p * h_hat
    clipped_h_hat = np.clip(scaled_h_hat, a_min=-100, a_max=100)
    w_t = p_t * np.exp(clipped_h_hat)
    p_t = w_t / w_t.sum()
    return p_t