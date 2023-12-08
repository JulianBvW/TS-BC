import torch

def dist_manhatten(a, b):
    diffs = a - b
    diffs = torch.abs(diffs)
    diffs = diffs.sum(1 if a.ndim == 2 else 0)
    return diffs

def dist_euclidean(a, b):
    diffs = a - b
    diffs = diffs**2
    diffs = diffs.sum(1 if a.ndim == 2 else 0)
    return diffs

def dist_chebyshev(a, b):
    diffs = a - b
    diffs = torch.abs(diffs)
    diffs = diffs.max(1 if a.ndim == 2 else 0).values
    return diffs

def dist_cosine(a, b):
    d = -1 if a.ndim == 2 else 0
    sim = torch.nn.functional.cosine_similarity(a, b, dim=d)
    return 1 - sim

def nDCG():
    pass  # TODO normalized discounted cumulative gain

DISTANCE_FUNCTIONS = {
    'manhatten': dist_manhatten,
    'euclidean': dist_euclidean,
    'chebyshev': dist_chebyshev,
    'cosine': dist_cosine
}