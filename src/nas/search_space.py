# src/nas/search_space.py
import random

# Each layer in a candidate architecture is represented as a dict of choices.
BLOCK_TYPES = ["cnn_small", "cnn_large", "bilstm", "attention"]
CHANNEL_CHOICES = [16, 32, 64]
ATTENTION_HEADS = [2, 4, 8]
DROPOUT_CHOICES = [0.0, 0.1, 0.2]
NUM_LAYERS_RANGE = (2, 5)  # min, max layers


def sample_layer_config():
    """Randomly sample one layer's configuration."""
    block_type = random.choice(BLOCK_TYPES)
    config = {
        "block_type": block_type,
        "channels": random.choice(CHANNEL_CHOICES),
        "dropout": random.choice(DROPOUT_CHOICES),
    }
    if block_type == "attention":
        config["num_heads"] = random.choice(ATTENTION_HEADS)
    return config


def sample_architecture():
    """Randomly sample a full candidate architecture: a list of layer configs."""
    num_layers = random.randint(*NUM_LAYERS_RANGE)
    return [sample_layer_config() for _ in range(num_layers)]


def mutate_architecture(architecture, mutation_rate=0.3):
    """Randomly mutate an existing architecture — used by the evolutionary search
    in Phase 4 to generate new candidates from good ones."""
    new_arch = []
    for layer in architecture:
        if random.random() < mutation_rate:
            new_arch.append(sample_layer_config())  # replace this layer
        else:
            new_arch.append(layer)  # keep it unchanged
    return new_arch


if __name__ == "__main__":
    print("Sampling 3 random candidate architectures:\n")
    for i in range(3):
        arch = sample_architecture()
        print(f"Candidate {i+1} ({len(arch)} layers):")
        for j, layer in enumerate(arch):
            print(f"  Layer {j+1}: {layer}")
        print()

    print("Mutating candidate 1:")
    arch1 = sample_architecture()
    mutated = mutate_architecture(arch1)
    print(f"Original: {arch1}")
    print(f"Mutated:  {mutated}")