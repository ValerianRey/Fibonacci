from inq.quantization import *
import pickle
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


def gather_activation_stats(model, x, stats):
    model.eval()  # Switch to eval model (so that we properly handle the dropout layers for example)
    i = 0
    for layer in model.seq:
        if type(layer) in supported_modules:
            # x is flattened except for the sample id dimension
            update_stats(x.view(x.shape[0], -1), stats, repr(i))
            i += 1
        x = layer(x)

    update_stats(x, stats, 'out')  # Useless stat, TODO: remove that


# Entry function to get stats of all functions.
def gather_stats(model, loader):
    device = 'cuda:0'
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
        final_stats[key] = {"avg_max": value["max_sum"] / value["samples"], "avg_min": value["min_sum"] / value["samples"],
                            "max": value["max"], "min": value["min"]}

    print("Gathering completed")
    return final_stats


def load_or_gather_stats(model, train_stats_loader, load):
    if load:
        print("Loading stats from save, be sure to remove this when the quantization scheme changes")
        with open('saves/stats_train.pickle', 'rb') as handle:
            stats = pickle.load(handle)

    else:
        stats = gather_stats(model, train_stats_loader)
        print("Saving stats for later use (if same quantization scheme)")
        with open('saves/stats_train.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stats


def gather_qmodel_part_means(qmodel, data, args, layers_stats):
    qmodel_forward(qmodel, data, num_bits=args.weight_bits, layers_stats=layers_stats)


def gather_qmodel_means(qmodel, args, loader):
    device = 'cuda:0'
    qmodel.eval()

    title = 'Gathering layers means'
    color = ''

    layers_stats = []
    final_means = []
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            layers_stats.append({'part3': [], 'part4': []})
            final_means.append({})

    start_time = time.clock()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            elapsed_time = time.clock() - start_time
            print_gather(title, batch_idx, len(loader), elapsed_time, color=color, persistent=False)
            data = data.to(device)
            gather_qmodel_part_means(qmodel, data, args, layers_stats)
        print_gather(title, batch_idx, len(loader), elapsed_time, color=color, persistent=True)

    final_means = [{}, {}, {}, {}]
    # Be careful, all the elements of layers_stats[i]['partj'] have the same weight in the mean, but some of them are the averages over a smaller batch (the last batch)
    # If the loader has a batch size that divides the set total number of samples we don't have this problem
    for i in range(len(layers_stats)):
        final_means[i]['part3'] = torch.mean(torch.cat(layers_stats[i]['part3']), dim=0)
        final_means[i]['part4'] = torch.mean(torch.cat(layers_stats[i]['part4']), dim=0)
    print("Gathering completed")

    return final_means


def load_or_gather_layers_means(qmodel, args, train_stats_loader, load, fibonacci_encode):
    if load:
        print("Loading layers_means from save, be sure to remove this when the quantization scheme changes")
        fib_str = 'fib' if fibonacci_encode else 'nofib'
        with open('saves/layers_means_train_' + fib_str + '.pickle', 'rb') as handle:
            layers_means = pickle.load(handle)
    else:
        layers_means = gather_qmodel_means(qmodel, args, train_stats_loader)
        print("Saving layers_means for later use (if same quantization scheme)")
        fib_str = 'fib' if fibonacci_encode else 'nofib'
        with open('saves/layers_means_train_' + fib_str + '.pickle', 'wb') as handle:
            pickle.dump(layers_means, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return layers_means

