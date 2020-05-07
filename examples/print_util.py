import torch
from examples.supported_modules import supported_modules


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
    print('Training the neural network')
    print(Color.BOLD + color + 'Epoch'.ljust(14) + 'Batch'.ljust(14) + 'Time'.ljust(7)
          + 'Loss (avg)'.ljust(25) + 'Top1 Acc (avg)'.ljust(22) + Color.END)


def print_train(epoch, total_epochs, batch_idx, num_batches, batch_time, loss, top1, persistent=True, color=''):
    print('\r' + color + '[{0}/{1}]'.format(epoch + 1, total_epochs).ljust(14) +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(25) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')


def print_test(batch_idx, num_batches, batch_time, loss, top1, persistent=True, color='', title='Test'):
    print('\r' + color + title.ljust(14) +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(25) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')


def print_layer(name, layer, print_data=False):
    title_color = ''
    data_color = Color.GRAY

    weights = layer.weight.data
    min, max = weights.min().item(), weights.max().item()
    print(title_color + name + " weights (min=" + repr(min) + ", max=" + repr(max) + ")" + Color.END)
    if print_data:
        print(data_color + repr(weights) + Color.END)

    biases = layer.bias.data
    min, max = biases.min().item(), biases.max().item()
    print(title_color + name + " biases (min=" + repr(min) + ", max=" + repr(max) + ")" + Color.END)
    if print_data:
        print(data_color + repr(biases) + Color.END)

    if layer.zp is None:
        print(title_color + name + " quantization data: " + "None" + Color.END)
    else:
        print(title_color + name + " quantization data: " + "shift=" + repr(layer.shift)
              + ", mult=" + repr(layer.mult) + ", zp=" + repr(layer.zp)
              + ", zp_next=" + repr(layer.zp_next) + Color.END)

    print()


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


def print_gather(title, batch_idx, num_batches, elapsed_time, color='', persistent=False, ):
    print('\r' + color + title.ljust(30) +
          '[{0}/{1}]'.format(batch_idx + 1, num_batches).ljust(14) +
          '{0:.0f}s'.format(elapsed_time).ljust(7) + Color.END,
          end='\n' if persistent else '')


# Counts the number of values lying outside of the [min_value, max_value] range, returns this as a tuple of 2 numbers and can print them
def count_out(x, min_value, max_value, log=False):
    assert isinstance(x, torch.Tensor)
    too_low = torch.sum(x < min_value).item()
    too_high = torch.sum(x > max_value).item()
    color_too_low = Color.GRAY if too_low == 0 else Color.RED
    color_too_high = Color.GRAY if too_high == 0 else Color.RED
    if log:
        print(Color.GRAY + 'Values clamped to min: ' + color_too_low + repr(too_low) + Color.GRAY + ' - Values clamped to max: '
              + color_too_high + repr(too_high) + Color.END)
    return too_low, too_high

