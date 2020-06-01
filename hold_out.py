import argparse
import os
import random
import shutil
import time
import warnings
import sys
import json

import torch
import torch.backends.cudnn as cudnn

from core.worker import Worker
from core.utils import summary_angular_errors
from core.print_logger import PrintLogger
from core.cache_manager import CacheManager

parser = argparse.ArgumentParser(description='Color Constancy: Cross validation')

parser.add_argument('configurationfile',
                    help='path to configuration file')
parser.add_argument('dataset', help='dataset class name')
parser.add_argument('subdataset', help='subdataset name')
parser.add_argument('trainfiles', help='text file contraining the files to train')
parser.add_argument('--valfile', help='text file contraining the files to validate', type=str)
parser.add_argument('--testfile', help='text file contraining the files to test', type=str)

parser.add_argument('-gpu', type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous execution')
parser.add_argument('--pretrainedmodel', help='path to model pretrained model file')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_fullres', action='store_true',
                    help='save full resolution prediction images')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--outputfolder', default='./output/', type=str,
                    help='path for the ouput folder. ')
parser.add_argument('--datapath', default='data/paths.json', type=str,
                    help='path to json file that specifies the directories of the datasets. ')

def generate_results(res, prefix=None):
    errors = [r.error for r in res]
    results = summary_angular_errors(errors)
    if prefix is not None:
        print(prefix, end=' ')
    for k in results.keys():
        print(k +':', "{:.4f}".format(results[k]), end=' ')
    print()

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # load configuration file: epochs, loss function, etc... for this experiment
    with open(args.configurationfile, 'r') as f:
        conf = json.load(f)

    # load datapath file: paths specific to the current machine
    with open(args.datapath, 'r') as f:
        data_conf = json.load(f)

    # remove previous results
    output_dir = os.path.join(args.outputfolder, args.dataset, args.subdataset, conf['name'])
    if not args.evaluate and not args.resume:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    # create output folder
    os.makedirs(output_dir, exist_ok=True)
    args.outputfolder = output_dir

    # copy configuration file to output folder
    shutil.copy(args.configurationfile, os.path.join(output_dir, os.path.basename(args.configurationfile)))

    # we overwrite the stdout and stderr (standard output and error) to
    # files in the output directory
    sys.stdout = PrintLogger(os.path.join(output_dir, 'stdout.txt'), sys.stdout)
    sys.stderr = PrintLogger(os.path.join(output_dir, 'stderr.txt'), sys.stderr)

    fold = 0 # no folds, but we always use fold #0 for these experiments
    cache = CacheManager(conf)
    worker = Worker(fold, conf, data_conf, cache, args)
    res, _ = worker.run()

    # some datasets have no validation GT
    if len(res) > 0:
        # print angular errors statistics (mean, median, etc...)
        generate_results(res, 'test')

if __name__ == '__main__':
    main()
