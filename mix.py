import random
import bisect
from torch.utils.data import Dataset

class MultiRatioDataset(Dataset):
    def __init__(self, datasets, ratios, total_length=10000):
        assert len(datasets) == len(ratios), "Each dataset must have a corresponding ratio"
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

        self.datasets = datasets
        self.ratios = ratios
        self.total_length = total_length
        self.lengths = [len(ds) for ds in datasets]

        # Build cumulative distribution for weighted sampling
        self.cumulative_ratios = [sum(ratios[:i+1]) for i in range(len(ratios))]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        dataset_idx = bisect.bisect_right(self.cumulative_ratios, r)
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]



import random
from torch.utils.data import Dataset
import bisect

class MultiDatasetMixer(Dataset):
    def __init__(self, datasets, ratios, seed=None):
        assert len(datasets) == len(ratios), "Must have one ratio per dataset"
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

        # Ensure all datasets are the same length
        lengths = [len(ds) for ds in datasets]
        assert all(l == lengths[0] for l in lengths), "All datasets must have the same length"

        self.datasets = datasets
        self.ratios = ratios
        self.length = lengths[0]

        # Compute cumulative distribution for sampling
        self.cumulative_ratios = [sum(ratios[:i+1]) for i in range(len(ratios))]

        # Precompute sampling source per index
        if seed is not None:
            random.seed(seed)

        self.source_indices = [
            self._sample_dataset_index(random.random())
            for _ in range(self.length)
        ]

    def _sample_dataset_index(self, r):
        # Find the right bin using binary search
        return bisect.bisect_right(self.cumulative_ratios, r)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx = self.source_indices[idx]
        return self.datasets[dataset_idx][idx]
