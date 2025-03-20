import torch


def adam_corrected_update(model, G_t_hat, m_t, v_t, v_hat, t, lr, betas, lambda_, eps, device, q):
    beta1 = betas[0] * lambda_ ** (t-1)
    beta1_prev = betas[0] * lambda_ ** (t-2)
    beta2 = betas[1]
    alpha_t = lr / t**0.5
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            m_t[i] = beta1 * m_t[i] + (1 - beta1) * G_t_hat[i]
            v_t[i] = beta2 * v_t[i] + (1 - beta2) * (G_t_hat[i] ** 2)
            if t == 1:
                v_hat[i] = v_t[i]
            else:
                v_hat[i] = torch.max(((1-beta1)**2/(1-beta1_prev)**2)*v_hat[i], v_t[i])
            
            m_hat_t = m_t[i] / (1- beta1 ** t)
            v_hat_t = v_hat[i] / (1 - beta2 ** t)
            
            update_step = alpha_t * m_hat_t / (torch.sqrt(v_hat_t) + eps)
            param.data -= update_step.to(device)
