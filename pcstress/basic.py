import torch
from tqdm import tqdm


def default_PCStress(loader, augmenter, epsilons, norm_batch_size=1000, nchannel=3, imgsize=(32, 32), norm='l2'):
    """

    @return: Tensor of shape (dataset_len,epsilons_len)
    @param loader: dataset loader.
    @param epsilons: a list of epsilon values to evaluate the PC stress on.
    @param augmenter:
    @param norm_batch_size: number of points to sample.
    """
    device = augmenter.device
    normalds = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(1 * nchannel * imgsize[0] * imgsize[1]).cuda(),
        torch.eye(
            1 * nchannel * imgsize[0] * imgsize[1]).cuda()).expand(
        (norm_batch_size,))
    print("working with epsilon values of ", epsilons)
    with torch.no_grad():
        all_results = []
        for test_i, batch in enumerate(
                tqdm(loader, desc='stress loop:')):
            X, y = batch
            X, y = X.to(device), y.to(device)
            outputs = augmenter.eval_mode_pred(X)
            _, predictions = torch.max(outputs, 1)
            for ox, op in zip(X, predictions):
                ox = ox.unsqueeze(0)
                step = normalds.sample().reshape(norm_batch_size, -1)
                # project into sphere
                step_norm = torch.sqrt((step * step).sum(dim=1, keepdim=True))
                step /= step_norm
                step = step.reshape(norm_batch_size, nchannel, imgsize[0], imgsize[1])
                res = []
                for ep in epsilons:
                    epouts = augmenter.eval_mode_pred(ox + step * ep)
                    _, ep_npredicted = torch.max(epouts, 1)
                    res.append((op == ep_npredicted).sum().float().cpu())
                all_results.append(torch.as_tensor(res))
        all_results = torch.stack(all_results)
        print("PC Stress done results: shape =", all_results.shape)
        return all_results
