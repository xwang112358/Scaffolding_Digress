import torch

class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def sample_n_greater_than(self, n_min, device):
        """Sample a number of nodes that is greater than n_min for scaffold decoration.
        Args:
            n_min: Minimum number of nodes (tensor or int)
            device: Device to place the sampled tensor on
        Returns:
            A tensor with sampled number of nodes
        """

        n_min = n_min.cpu() if isinstance(n_min, torch.Tensor) else n_min
        valid_indices = torch.arange(len(self.prob)) > n_min
        valid_prob = self.prob[valid_indices]
        valid_prob = valid_prob / valid_prob.sum()
        valid_dist = torch.distributions.Categorical(valid_prob)
        
        # Sample from valid distribution and add offset
        idx = valid_dist.sample() + (n_min + 1)
        
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
