import argparse
import os
import sys
import json
import shutil

from core.worker import Worker
from core.utils import summary_angular_errors
from core.print_logger import PrintLogger
from core.cache_manager import CacheManager

parser = argparse.ArgumentParser(description='Color Constancy: Inference')

parser.add_argument('configurationfile',
                    help='path to configuration file')
parser.add_argument('dataset', help='dataset class name')
parser.add_argument('subdataset', help='subdataset name')
parser.add_argument('testfile', help='text file contraining the files to test')
parser.add_argument('checkpointfile', help='path to model checkpoint file')

parser.add_argument('-gpu', type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--save_fullres', action='store_true',
                    help='save full resolution prediction images')
parser.add_argument('--outputfolder', default='./output/', type=str,
                    help='path for the ouput folder. ')
parser.add_argument('--datapath', default='data/paths.json', type=str,
                    help='path to json file that specifies the directories of the datasets. ')

def generate_results(res):
    errors = [r.error for r in res]
    if errors[0] is None:
        print('No GT')
        return
    results = summary_angular_errors(errors)
    for k in results.keys():
        print(k +':', "{:.4f}".format(results[k]), end=' ')
    print()


def main():
    args = parser.parse_args()

    # load configuration file for this experiment
    with open(args.configurationfile, 'r') as f:
        conf = json.load(f)

    # load datapath file: paths specific to the current machine
    with open(args.datapath, 'r') as f:
        data_conf = json.load(f)

    # create output folder
    os.makedirs(args.outputfolder, exist_ok=True)

    # copy configuration file to output folder
    shutil.copy(args.configurationfile, os.path.join(args.outputfolder, os.path.basename(args.configurationfile)))

    # we overwrite the stdout and stderr (standard output and error) to
    # files in the output directory
    sys.stdout = PrintLogger(os.path.join(args.outputfolder, 'stdout.txt'), sys.stdout)
    sys.stderr = PrintLogger(os.path.join(args.outputfolder, 'stderr.txt'), sys.stderr)

    # used in core/worker.py to determine what to do
    args.evaluate = True
    args.resume = False
    args.valfile = None
    args.trainfiles = None

    # init the cache manager: this caches images from the dataset
    # to avoid reading them more than once
    cache = CacheManager(conf, no_cache = True)

    fold = 0 # no folds, but we always use fold #0 for these experiments
    worker = Worker(fold, conf, data_conf, cache, args, inference=True)
    res, _ = worker.run()

    # print angular errors statistics (mean, median, etc...)
    generate_results(res)
if __name__ == '__main__':
    main()
