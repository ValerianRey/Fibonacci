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
    print(Color.BOLD + color + 'Epoch'.ljust(10) + 'Batch'.ljust(14) + 'Time'.ljust(7)
          + 'Loss (avg)'.ljust(18) + 'Top1 Acc (avg)'.ljust(22) + Color.END)


def print_train(epoch, total_epochs, batch_idx, batch_length, batch_time, loss, top1, persistent=True, color=''):
    print('\r' + color + '[{0}/{1}]'.format(epoch + 1, total_epochs).ljust(10) +
          '[{0}/{1}]'.format(batch_idx + 1, batch_length).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(18) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')


def print_test(batch_idx, batch_length, batch_time, loss, top1, persistent=True, color=''):
    print('\r' + color + 'Test'.ljust(10) +
          '[{0}/{1}]'.format(batch_idx+1, batch_length).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(18) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')

