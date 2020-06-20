from inq.quantizer import *
import torch.utils.data
import time
from examples.supported_modules import supported_modules
from examples.print_util import print_gather


# Get Min and max of x tensor, and stores it
def update_stats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
    batch_size = max_val.shape[0]

    if key not in stats:
        stats[key] = {"max_sum": max_val.sum(), "min_sum": min_val.sum(), "samples": batch_size, "max": max(max_val), "min": min(min_val)}
    else:
        stats[key]['max_sum'] += max_val.sum().item()
        stats[key]['min_sum'] += min_val.sum().item()
        stats[key]['samples'] += batch_size
        stats[key]['max'] = max(stats[key]['max'], max(max_val))
        stats[key]['min'] = min(stats[key]['min'], min(max_val))


# Compute the activation stats at each layer for 1 batch of data (x)
def gather_activation_stats(model, x, stats):
    model.eval()  # Switch to eval model (so that we properly handle the dropout layers for example)
    i = 0
    for layer in model.seq:
        if type(layer) in supported_modules:
            # x is flattened except for the sample id dimension
            update_stats(x.view(x.shape[0], -1), stats, repr(i))
            i += 1
        x = layer(x)


# Compute the activation stats at each layer. You should do that on the training set because
# we want to quantize the model without seeing the test / validation set.
def gather_stats(model, loader, device):
    model.eval()
    stats = {}

    title = 'Gathering activation stats'
    color = ''

    start_time = time.clock()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            elapsed_time = time.clock() - start_time
            print_gather(title, batch_idx, len(loader), elapsed_time, color=color, persistent=False)
            data, target = data.to(device), target.to(device)
            gather_activation_stats(model, data, stats)

        print_gather(title, batch_idx, len(loader), elapsed_time, color=color, persistent=True)
    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"avg_max": (value["max_sum"] / value["samples"]), "avg_min": (value["min_sum"] / value["samples"]),
                            "max": value["max"], "min": value["min"]}

    return final_stats


# Either load activation stats from file, or compute them using the model and store them to a file
# These stats are for example the min / max of the activation coming into each quantized layer
# They are used to compute scale_x and zp_x at each layer (or more precisely, to compute scale_x_next
# and zp_x_next at the layer before the one where the stats are computed)
def load_or_gather_stats(model, train_stats_loader, device, load, saves_path):
    stats_path = saves_path + 'stats.pth'
    if load:
        print("Loading stats from save, be sure to remove this when the quantization scheme changes")
        stats = torch.load(stats_path)

    else:
        stats = gather_stats(model, train_stats_loader, device)
        torch.save(stats, stats_path)

    return stats


# This function returns the proportion of fib weights in a given qmodel, averaged over the layers
# The idea of using weighted=False is that all layers are equally important no matter how many weights they have
# Weighted=True gives the true proportion of fib weights in the qmodel
def average_proportion_fib(qmodel, weighted=False):
    bits = qmodel.bits.item()
    proportions = []
    weights = []
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            n = np.prod(layer.weight.shape) if weighted else 1
            proportions.append(proportion_fib(layer.weight, bits=bits) * n)
            weights.append(n)

    return sum(proportions) / sum(weights)


# This function returns the average distance to fib weights in a given qmodel, averaged over the layers
# The idea of using weighted=False is that all layers are equally important no matter how many weights they have
# Weighted=True gives the true average distance to fib weights in the qmodel
def average_distance_fib(qmodel, weighted=False):
    bits = qmodel.bits.item()
    avg_distances = []
    weights = []
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            _, distances = fib_distances(layer.weight, bits)
            avg_fib_dist = distances.mean().item()
            n = np.prod(layer.weight.shape) if weighted else 1
            avg_distances.append(avg_fib_dist * n)
            weights.append(n)

    return sum(avg_distances) / sum(weights)

