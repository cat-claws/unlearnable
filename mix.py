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

# class MultiDatasetMixer(Dataset):
#     def __init__(self, datasets, ratios, seed=None):
#         assert len(datasets) == len(ratios), "Must have one ratio per dataset"
#         assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

#         # Ensure all datasets are the same length
#         lengths = [len(ds) for ds in datasets]
#         assert all(l == lengths[0] for l in lengths), "All datasets must have the same length"

#         self.datasets = datasets
#         self.ratios = ratios
#         self.length = lengths[0]

#         # Compute cumulative distribution for sampling
#         self.cumulative_ratios = [sum(ratios[:i+1]) for i in range(len(ratios))]

#         # Precompute sampling source per index
#         if seed is not None:
#             random.seed(seed)

#         self.source_indices = [
#             self._sample_dataset_index(random.random())
#             for _ in range(self.length)
#         ]

#     def _sample_dataset_index(self, r):
#         # Find the right bin using binary search
#         return bisect.bisect_right(self.cumulative_ratios, r)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         dataset_idx = self.source_indices[idx]
#         return self.datasets[dataset_idx][idx]


class MultiDatasetMixer(Dataset):
    def __init__(self, datasets, ratios, seed=None):
        assert len(datasets) == len(ratios), "Must have one ratio per dataset"
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

        self.datasets = datasets
        self.ratios = ratios

        # Determine minimum dataset length among real datasets
        real_datasets = [ds for ds in datasets if ds is not None]
        min_len = min(len(ds) for ds in real_datasets)

        # Compute number of total valid samples (ignore placeholders)
        self.real_indices = []
        self.dataset_lookup = []
        self.cumulative_ratios = [0.0]
        total_real_ratio = 0.0

        for i, (ds, r) in enumerate(zip(datasets, ratios)):
            if ds is not None:
                total_real_ratio += r
                self.dataset_lookup.append(i)
                self.cumulative_ratios.append(total_real_ratio)

        self.length = int(min_len * total_real_ratio)

        if seed is not None:
            random.seed(seed)

        # Precompute sampling plan
        self.source_indices = [
            self._sample_dataset_index(random.random())
            for _ in range(self.length)
        ]

    def _sample_dataset_index(self, r):
        # Scale r to real part only
        r_scaled = r * self.cumulative_ratios[-1]
        idx = bisect.bisect_right(self.cumulative_ratios, r_scaled) - 1
        return self.dataset_lookup[idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx = self.source_indices[idx]
        return self.datasets[dataset_idx][idx]
