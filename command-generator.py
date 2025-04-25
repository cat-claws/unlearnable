import itertools

# Base config
base_config = {
    "python": "python",
    "script": "train.py",
    "--model": "WideResNet",
    "--model-depth": "28",
    "--model-widen_factor": "10",
    "--model-drop_rate": "0.3",
    "--optimizer": "SGD",
    "--optimizer-lr": "0.1",
    "--optimizer-momentum": "0.9",
    "--optimizer-weight_decay": "0.0005",
    "--scheduler": "CosineAnnealingLR",
    "--scheduler-T_max": "200",
    "--batch_size": "128",
}

# Variant groups
model_variants = [
    {"--model-depth": "34"},
    {"--model-drop_rate": "0.0"},
]

optimizer_variants = [
    {"--optimizer": "AdamW", "--optimizer-lr": "0.001", "--optimizer-weight_decay": "0.01"},
]

scheduler_variants = [
    {"--scheduler": "StepLR", "--scheduler-step_size": "100", "--scheduler-gamma": "0.1"},
]

variant_groups = [model_variants, optimizer_variants, scheduler_variants]

def generate_ablation_configs(base, variant_groups):
    configs = []
    for group in variant_groups:
        for variant in group:
            cfg = base.copy()
            cfg.update(variant)
            configs.append(cfg)
    return configs

def generate_pairwise_combos(base, variant_groups):
    combos = []
    for g1, g2 in itertools.combinations(variant_groups, 2):
        for v1 in g1:
            for v2 in g2:
                cfg = base.copy()
                cfg.update(v1)
                cfg.update(v2)
                combos.append(cfg)
    return combos

def save_as_runnable_sh(configs, filename):
    with open(filename, "w") as f:
        for cfg in configs:
            parts = [cfg["python"], cfg["script"]]
            for k, v in cfg.items():
                if k not in {"python", "script"}:
                    parts.append(k)
                    parts.append(str(v))
            line = "\t".join(parts)
            f.write(line + "\n")
    print(f"✅ Saved as: {filename} (runnable .sh, tab-separated)")
    print(f"💡 You can now run: bash {filename}")

# Main
if __name__ == "__main__":
    print("🔬 Generating clean runnable ablation shell script...")
    all_configs = generate_ablation_configs(base_config, variant_groups)
    # all_configs += generate_pairwise_combos(base_config, variant_groups)

    save_as_runnable_sh(all_configs, "ablations.tsv")
    print(f"🧪 Total runnable commands: {len(all_configs)}")
