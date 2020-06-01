import argparse
import os
import sys
import json
import shutil

from core.worker import Worker
from core.print_logger import PrintLogger
from core.cache_manager import CacheManager
from core.crossvalidation import Fold, Crossvalidation

parser = argparse.ArgumentParser(description='Color Constancy: Inference')

parser.add_argument('configurationfile',
                    help='path to configuration file')
parser.add_argument('datasetsplits', help='path to text file containing all the dataset description')
parser.add_argument('-checkpointfile', '--c', dest='checkpointfile', action='append', help='path to model checkpoint file', required=True)

parser.add_argument('-gpu', type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--save_fullres', action='store_true',
                    help='save full resolution prediction images')
parser.add_argument('--outputfolder', default='./output/', type=str,
                    help='path for the ouput folder. ')
parser.add_argument('--datapath', default='data/paths.json', type=str,
                    help='path to json file that specifies the directories of the datasets. ')

def main():
    args = parser.parse_args()

    # load configuration file for this experiment
    with open(args.configurationfile, 'r') as f:
        conf = json.load(f)

    # load datapath file: paths specific to the current machine
    with open(args.datapath, 'r') as f:
        data_conf = json.load(f)

    # load all splits/folds from the text file
    # 'datasetsplits' is a text file that contains a list of text files,
    # each line corresponds to one fold
    splits = None
    with open(args.datasetsplits, 'r') as f:
        splits = f.readlines()

    # the first line of 'datasetsplits', is the class name of the
    # dataset (if it is 'Nus', it will correspond to datasets/nus.py)
    splits = [e.strip() for e in splits]
    args.dataset = splits[0]
    splits = splits[1:]

    # get subdataset from filename
    subdataset = os.path.basename(args.datasetsplits).replace('.txt', '')
    args.subdataset = subdataset

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

    # init the cache manager: this caches images from the dataset
    # to avoid reading them more than once
    cache = CacheManager(conf)

    folds = []
    for fold in range(len(splits)):
        # This is the format for cross validation files:
        # DatasetClass
        # Fold1.txt [ValFold1.txt] [TestFold1.txt]
        # Fold2.txt [ValFold2.txt] [TestFold2.txt]
        # ...
        # 1. if there's only FoldX.txt, we treat this as independent folds (they should be disjoint),
        #    for each CV fold, the training will be the amalgamation of the rest of the folds and FoldX.txt
        #    will be the validation fold. Example: data/p30/noccb.txt
        # 2. if there's FoldX.txt and ValFoldX.txt, the training fold will be FoldX.txt and ValFoldX.txt will
        #    be the validation folds.
        # 3. TestFoldX.txt is assumed to be the same as the validation fold unless it is specified.
        #    Validation results are shown during training, and Test results are shown at the end.
        split_list = splits[fold].split(' ')
        if len(split_list) == 1:
            valfile = splits[fold]
            trainfiles = list(splits)
            trainfiles.remove(valfile)
            testfile = valfile
        elif len(split_list) == 2:
            trainfiles, valfile = split_list
            testfile = valfile
        elif len(split_list) == 3:
            trainfiles, valfile, testfile = split_list
        else:
            raise Exception('Wrong number of files in splits')
        # We prepare the folds with all details for processing in
        # the Crossvalidation code (core/crossvalidation.py)
        folds.append(Fold(trainfiles, valfile, testfile))

    # Go to core/crossvalidation.py for more details
    cv = Crossvalidation(cache, conf, data_conf, args, folds, inference=True)
    results = cv.run()

    for k in results.keys():
        print(k +':', "{:.4f}".format(results[k]), end=' ')
    print()
if __name__ == '__main__':
    main()
