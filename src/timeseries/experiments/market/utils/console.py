import time


def print_progress(i, test_bundles, verbose):
    if verbose >= 2 and i % 50 == 0:
        print("{}/{} - {}% predictions done".format(i, len(test_bundles), round(i * 100 / len(test_bundles))),
              end='\r')


def print_pred_time(start_time, test_bundles, verbose):
    end_time = time.time()
    if verbose >= 1:
        print("{} predictions in {}s: avg: {}s".format(len(test_bundles), round(end_time - start_time, 2),
                                                       round((end_time - start_time) / len(test_bundles), 4)))


def print_progress_loop(i, tot, process_text=''):
    print("{}/{} - {}% {} done".format(i+1, tot, round((i+1) * 100 / tot), process_text), end='\r')
