import argparse

# Define dataset-specific configurations
DATASET_CONFIGS = {
    "amazon": {
        "align_weight": 0.1,
        "lr": 0.0005,
        "weight_decay": 0.25,
        "num_neg": 50,
        "margin": 1.0
    },
    "yelp": {
        "align_weight": 0.1,
        "lr": 0.0005,
        "weight_decay": 0.2,
        "num_neg": 50,
        "margin": 1.0
    },
    "google": {
        "align_weight": 0.01,
        "lr": 0.001,
        "weight_decay": 0.05,
        "num_neg": 40,
        "margin": 2.0
    },
}

def parse_configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon", help="amazon, yelp, google")
    parser.add_argument("--align_weight", type=float, help="contrastive alignment loss weight")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=float, help="l2 regularization strength")
    parser.add_argument("--num_neg", type=int, help="number of negative samples")
    parser.add_argument("--margin", type=float, help="margin value in the metric learning loss")
    parser.add_argument("--euc_lr", type=float, default=1e-5, help="learning rate for mlp")
    parser.add_argument("--dim", type=int, default=50, help="embedding dimension")
    parser.add_argument("--scale", type=float, default=0.1, help="scale for init")
    parser.add_argument("--eval-freq", type=int, default=10, help="how often to compute val metrics (in epochs)")
    parser.add_argument("--momentum", type=float, default=0.95, help="momentum in optimizer")
    parser.add_argument("--c", type=float, default=1.0, help="hyperbolic radius")
    parser.add_argument("--k", type=int, default=2, help="proportion of nodes between layer in the hierarchy")
    parser.add_argument("--num-layers", type=int, default=4, help="number of hidden layers in encoder")
    parser.add_argument("--batch-size", type=int, default=10000, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="maximum number of epochs to train for")
    parser.add_argument("--layer", type=int, default=8, help="layer of the hierarchy, support from 1 to max_layer")
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature for control the recommendation diversity")
    parser.add_argument("--mode", type=str, help="train, test; build, evaluate, balance")
    return parser.parse_args()

args = parse_configure()
# Override default values based on the dataset
if args.dataset in DATASET_CONFIGS:
    dataset_config = DATASET_CONFIGS[args.dataset]
    for key, value in dataset_config.items():
        setattr(args, key, value)