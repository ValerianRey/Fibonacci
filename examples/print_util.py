from examples.supported_modules import supported_modules
from inq.fib_util import *


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    BRIGHT_GREEN = '\033[92m'
    GREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(color=''):
    print(Color.BOLD + color + ''.ljust(33) + 'Batch'.ljust(14) + 'Time'.ljust(7)
          + 'Avg loss'.ljust(15) + 'Avg acc'.ljust(12) + 'LR'.ljust(8) + Color.END)


def print_train(epoch, total_epochs, batch_idx, num_batches, batch_time, loss, top1, lr, persistent=True, color=''):
    print('\r' + color + Color.BOLD + 'Training [{0}/{1}]'.format(epoch + 1, total_epochs).ljust(33) + Color.END + color +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.avg:.4f}'.format(loss=loss).ljust(15) +
          '{top1.avg:.3f}'.format(top1=top1).ljust(12) +
          '{0:.5f}'.format(lr).rstrip('0').rstrip('.').ljust(8) + Color.END,
          end='\n' if persistent else '')


def print_test(batch_idx, num_batches, batch_time, loss, top1, persistent=True, color='', title='Test'):
    print('\r' + color + Color.BOLD + title.ljust(33) + Color.END + color +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.avg:.4f}'.format(loss=loss).ljust(15) + Color.END + color +
          '{top1.avg:.3f}'.format(top1=top1).ljust(12) + Color.END,
          end='\n' if persistent else '')


def print_layer(name, layer, print_data=False):
    title_color = ''
    data_color = Color.GRAY

    weights = layer.weight.data
    min, max = weights.min().item(), weights.max().item()
    print(title_color + name + " weights (min=" + repr(min) + ", max=" + repr(max) + ")" + Color.END)
    if print_data:
        print(data_color + repr(weights) + Color.END)

    if layer.bias is not None:
        biases = layer.bias.data
        min, max = biases.min().item(), biases.max().item()
        print(title_color + name + " biases (min=" + repr(min) + ", max=" + repr(max) + ")" + Color.END)
        if print_data:
            print(data_color + repr(biases) + Color.END)

    print()


def print_quantization_params(name, layer, bits=8):
    percentage_fib = proportion_fib(layer.weight, bits) * 100
    _, distances = fib_distances(layer.weight, bits)
    avg_fib_dist = distances.mean()
    print(name + " quantization data: " + "bits=" + repr(bits) + ", shifts=" + repr(layer.shifts)
          + ", mults=" + repr(layer.mults) + ", zp_x=" + repr(layer.zp_x)
          + ", zp_x_next=" + repr(layer.zp_x_next) + ", {0:.1f}% of Fibonacci encoded weights".format(percentage_fib)
          + ", Average fib distance: {0:.2f}".format(avg_fib_dist) + Color.END)


def print_seq_model(model, how='long'):
    if how == 'no':
        return
    elif how == 'long':
        print_data = True
    else:
        print_data = False

    layers_to_print = []
    names_to_print = []

    for layer in model.seq:
        if type(layer) in supported_modules:
            layers_to_print.append(layer)
            names_to_print.append(layer.__class__.__name__)

    for name, layer in zip(names_to_print, layers_to_print):
        print_layer(name, layer, print_data=print_data)
        if 'bits' in model.state_dict().keys():
            print_quantization_params(name, layer, bits=model.bits.item())


def print_gather(title, batch_idx, num_batches, elapsed_time, color='', persistent=False, ):
    print('\r' + color + Color.BOLD + title.ljust(33) + Color.END + color +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{0:.0f}s'.format(elapsed_time).ljust(7) + Color.END,
          end='\n' if persistent else '')


# Counts the number of values lying outside of the [min_value, max_value] range, returns this as a tuple of 2 numbers and can print them
def count_out(x, min_value, max_value, log=False):
    assert isinstance(x, torch.Tensor)
    too_low = torch.sum(x < min_value).item()
    too_high = torch.sum(x > max_value).item()
    total = np.prod(x.shape)
    percent_too_low = too_low * 100 / total
    percent_too_high = too_high * 100 / total
    color_too_low = Color.GRAY if percent_too_low < 0.1 else Color.RED
    color_too_high = Color.GRAY if percent_too_high < 0.1 else Color.RED
    if log:
        print(Color.GRAY + 'Values clamped to min: ' + color_too_low + repr(too_low) + " ({0:.1f}%)".format(percent_too_low) +
              Color.GRAY + ' - Values clamped to max: ' + color_too_high + repr(too_high) + " ({0:.1f}%)".format(percent_too_high) + Color.END)
    return too_low, too_high


def print_dataset_name(dataset):
    color = Color.BLUE if dataset == 'cifar10' else Color.RED
    text = "CIFAR-10" if dataset == 'cifar10' else "MNIST" if dataset == 'mnist' else ''
    print(color + Color.BOLD + "\n\t\t\t" + text + "\n\n" + Color.END, end='')


def print_bn(bn):
    print("Running mean: " + repr(bn.running_mean))
    print("Running variance: " + repr(bn.running_var))
    print("Weight: " + repr(bn.weight))
    print("Bias: " + repr(bn.bias))
    print("Training: " + repr(bn.training))


def format_float(value, max_decimals=5):
    return ("{0:." + repr(max_decimals) + "f}").format(value).rstrip('0').rstrip('.')


def print_quantization_epoch(qepoch, max_epoch, percent):
    print("Quantization epoch " + Color.BOLD + Color.GREEN + repr(qepoch + 1) + "/" + repr(max_epoch) + Color.END
          + " - Will quantize " + Color.BOLD + Color.GREEN + format_float(percent) + '%' + Color.END + " of the weights as fibonacci")


def print_fib_info(weighted_proportion, unweighted_proportion, weighted_distance, unweighted_distance):
    print("Proportion of fib weights: " +
          format_float(weighted_proportion * 100, max_decimals=2) + '%')
    print("Average proportion of fib weights over equally important layers: " +
          format_float(unweighted_proportion * 100, max_decimals=2) + '%')
    print("Average distance to fib weights: " +
          format_float(weighted_distance, max_decimals=3))
    print("Average average distance to fib weights over equally important layers: " +
          format_float(unweighted_distance, max_decimals=3))


def print_quantization(fib, proportions, step, bits, layer_idx, num_layers, layer_type, persistent=False):
    print("\rQuantizing model (" + ("fib " + format_float(proportions[step] * 100) + '%' if fib else "no fib") + ", {0} bits)".format(bits) +
          " - Layer [{0}/{1}] ({2})".format(layer_idx + 1, num_layers, layer_type.__name__)
          , end='\n' if persistent else '')
