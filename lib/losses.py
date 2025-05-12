import torch
import torch.nn.functional as F


def infoNCELoss(q, k, queue, T):
    # positive logits: Nx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    # logits: Nx(K+1)
    logits = torch.cat([l_pos, l_neg], dim=1)
    # apply temperature
    logits /= T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

    return F.cross_entropy(logits, labels)


def kde_regularization_vmf(features, kappa=5.0):
    # Kernel
    cosine_sim = torch.matmul(features, features.T)
    kernel_vals = torch.exp(kappa * cosine_sim)

    kde_vals = torch.mean(kernel_vals, dim=1)
    log_kde_vals = torch.log(kde_vals + 1e-8)  # Add small value to prevent log(0)

    regularization_loss = torch.mean(log_kde_vals)

    return regularization_loss
