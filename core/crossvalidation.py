from core.worker import Worker
from core.utils import summary_angular_errors

# Fold class: contains all necessary info for handling CV folds
class Fold():
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

class Crossvalidation():
    def __init__(self, cache, conf, data_conf, args, folds, inference=False, verbose=True):
        self.cache = cache
        self.conf = conf
        self.data_conf = data_conf
        self.args = args
        self.folds = folds
        self.verbose = verbose
        self.inference = inference

    def _print_results(self, results, prefix=None):
        if prefix is not None:
            print(prefix, end=' ')
        for k in results.keys():
            print(k +':', "{:.4f}".format(results[k]), end=' ')
        print()

    def run(self):
        ae_res = []
        stability_res = []

        # Cross Validation loop
        for i in range(len(self.folds)):
            # get first fold #i info
            fold = self.folds[i]

            # set train, validation and test sets for this fold iteration
            self.args.trainfiles = fold.train
            self.args.valfile = fold.validation
            self.args.testfile = fold.test

            # all training logic is inside Worker (core/worker.py)
            worker = Worker(i, self.conf, self.data_conf, self.cache, self.args,
                            verbose=self.verbose, inference=self.inference)
            res, _ = worker.run()

            # print results for each fold
            if self.verbose:
                partial_res = summary_angular_errors([r.error for r in res])
                self._print_results(partial_res, 'fold '+str(i))

            # accumulate results into lists
            ae_res += res

        # summary_angular_errors: computes mean, median, best 25%, etc...
        results = summary_angular_errors([r.error for r in ae_res])
        if self.verbose:
            self._print_results(results, 'total')

        return results
