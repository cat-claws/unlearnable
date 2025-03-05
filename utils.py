def generate_combinations(param_dict):
    """
    Recursively generate all combinations from a nested parameter dictionary.
    """
    if not param_dict:
        return [{}]

    # Extract the first parameter key and its values
    key, values = next(iter(param_dict.items()))
    remaining_params = {k: v for k, v in param_dict.items() if k != key}

    if isinstance(values, dict):  # Handle nested dictionaries
        expanded_values = []
        for sub_key, sub_params in values.items():
            sub_combinations = generate_combinations(sub_params)
            for sub_config in sub_combinations:
                expanded_values.append({key: sub_key, f"{key}_config": sub_config})
    else:  # Handle standard parameter lists
        expanded_values = [{key: value} for value in values]

    # Recursively process the remaining parameters
    remaining_combinations = generate_combinations(remaining_params)

    # Merge current and remaining combinations
    return [{**base, **extra} for base in expanded_values for extra in remaining_combinations]

# def flatten_dict(d, parent_key='', sep='.'):

#     items = []
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

import numpy as np
import torch

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple, set)):
            v = torch.tensor(list(v)) if all(isinstance(i, (int, float, bool)) for i in v) else np.array(list(v), dtype=object)
            items.append((new_key, v))
        elif isinstance(v, np.ndarray):
            v = torch.tensor(v) if np.issubdtype(v.dtype, np.number) else v
            items.append((new_key, v))
        else:
            items.append((new_key, v))
    
    return dict(items)
