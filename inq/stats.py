from inq.quantization import *
import pickle
import torch.utils.data


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


# Reworked Forward Pass to access activation Stats through update_stats function
def gather_activation_stats(model, x, stats):
    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.conv1(x)
    x = F.relu(x)

    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = model.conv2(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)

    update_stats(x, stats, 'fc1')
    x = model.fc1(x)
    x = F.relu(x)

    update_stats(x, stats, 'fc2')
    x = model.fc2(x)

    update_stats(x, stats, 'out')


# Gathers the stats at some different places than gather_activation_stats (does not work well currently)
def gather_activation_stats2(model, x, stats):
    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.conv1(x)
    x = F.relu(x)

    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = model.conv2(x)
    update_stats(x.clone().view(x.shape[0], -1), stats, 'fc1')
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)

    x = model.fc1(x)
    x = F.relu(x)
    update_stats(x, stats, 'fc2')

    x = model.fc2(x)
    update_stats(x, stats, 'out')


# Entry function to get stats of all functions.
def gather_stats(model, test_loader, before_layer=True):
    device = 'cuda:0'
    print("Gathering stats...")
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if before_layer:
                gather_activation_stats(model, data, stats)
            else:
                gather_activation_stats2(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"avg_max": value["max_sum"] / value["samples"], "avg_min": value["min_sum"] / value["samples"],
                            "max": value["max"], "min": value["min"]}

    print("Gathering completed")
    return final_stats


def gather_qmodel_part_means(qmodel, data, args, layers_stats):
    qmodel_forward(qmodel, data, num_bits=args.weight_bits, layers_stats=layers_stats)


def gather_qmodel_stats(qmodel, args, loader, save=False, fibonacci_encode=False):
    device = 'cuda:0'
    print("Gathering qmodel stats...")
    qmodel.eval()

    layers_stats = [{}, {}, {}, {}]  # Each element of this list corresponds to a layer
    for layer_stats in layers_stats:
        layer_stats['part1'] = []  # Each element of this list is the average (over a batch) of the tensor of part1
        layer_stats['part2'] = []
        layer_stats['part3'] = []
        layer_stats['part4'] = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            gather_qmodel_part_means(qmodel, data, args, layers_stats)

    final_means = [{}, {}, {}, {}]
    # Be careful, all the elements of layers_stats[i]['partj'] have the same weight in the mean, but some of them are the averages over a smaller batch (the last batch)
    # If the loader has a batch size that divides the set total number of samples we don't have this problem
    for i in range(len(layers_stats)):
        final_means[i]['part1'] = torch.mean(torch.cat(layers_stats[i]['part1']), dim=0)  # This is the average (over all of the dataset) of the tensor of part1
        final_means[i]['part2'] = torch.mean(torch.cat(layers_stats[i]['part2']), dim=0)
        final_means[i]['part3'] = torch.mean(torch.cat(layers_stats[i]['part3']), dim=0)
        final_means[i]['part4'] = torch.mean(torch.cat(layers_stats[i]['part4']), dim=0)

    print("Gathering completed")
    if save:
        print("Saving stats for later use (if seed fixed)")
        fib_str = 'fib' if fibonacci_encode else 'nofib'
        with open('saves/layers_means_train_' + fib_str + '.pickle', 'wb') as handle:
            pickle.dump(final_means, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return final_means


def load_layers_means(fibonacci_encode=False):
    print("Loading layers_means from save, be sure to remove this when seed is not fixed")

    fib_str = 'fib' if fibonacci_encode else 'nofib'

    with open('saves/layers_means_train_' + fib_str + '.pickle', 'rb') as handle:
        layers_means = pickle.load(handle)

    return layers_means

