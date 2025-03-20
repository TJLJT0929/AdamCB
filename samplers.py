import numpy as np


# ADAMCB
def distr_multiPlays(weights, K, gamma=0.0):
    n = len(weights)
    weights_sum = np.sum(weights)
    probabilities = K * ((1.0 - gamma) * (weights / weights_sum) + (gamma / n))
    return probabilities


def getAlpha(temp, w_sorted):
    # getAlpha calculates the alpha value for the sorted weight.
    sum_weight = np.sum(w_sorted)
    for i in range(len(w_sorted)):
        alpha = (temp * sum_weight) / (1.0 - i * temp)
        curr = w_sorted[i]
        if alpha > curr:
            alpha_exp = alpha
            return alpha_exp
        sum_weight = sum_weight - curr
    raise Exception("alpha not found")


def find_indices(arr, condition):
    # Function that returns the indices satisfying the condition function
    return np.nonzero(condition(arr))[0]


def DepRound(weights_p, k=1):
    p = weights_p
    n = len(p)
    # Checks
    assert k < n, f"Error (DepRound): k = {k} should be < n = {n}."
    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(p <= 1), f"Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ...(={p})"
    assert np.isclose(np.sum(p), 1), f"Error: the sum of weights p_1 + ... + p_K should be = 1 (= {np.sum(p)})"
    indices = np.random.choice(n, size=k, replace=False, p=p)
    return indices


def batch_selection_adamcb(weights, K, gamma):
    n = len(weights)
    # 1. modify the weights
    theSum = np.sum(weights)
    temp = (1.0 / K - gamma / n) * float(1.0 / (1.0 - gamma))
    w_temp = weights.copy()
    if np.max(weights) >= temp * theSum:
        w_sorted = np.sort(weights)[::-1]
        alpha_t = getAlpha(temp, w_sorted)
        S_null = find_indices(w_temp, lambda e: e >= alpha_t)
        for s in S_null:
            w_temp[s] = alpha_t
    else:
        S_null = []
    # 2. compute the probability
    p_t = distr_multiPlays(w_temp, K, gamma=gamma)
    assert False in np.isnan(np.array(p_t)
        ), "Error, probability must be a real number"
    # 3. sample K-distinct arms
    indices = DepRound(p_t, k=K)
    return indices, p_t, S_null
