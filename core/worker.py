import time
import sys
import math
import os
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from core.utils import *
from core.tensorboard_logger import TensorBoardLogger
from core.display import Display
from core.factory import Factory
from tqdm import tqdm

# statistics of epoch: angular error (mean, median), mean loss and time it took
class EpochStats():
    def __init__(self, mean_err, med_err, mean_loss, t):
        self.mean_err = mean_err
        self.med_err = med_err
        self.mean_loss = mean_loss
        self.time = t

class Worker():
    def __init__(self, fold, conf, data_conf, cache_manager, args, inference=False, verbose=True):
        self._args = args
        self._fold = fold
        self._conf = conf
        self._data_conf = data_conf
        self._inference = inference
        self._verbose = verbose
        self.tmp_dir = self._data_conf['tmp']

        # we save output with this folder structure:
        # output/
        #       -> tensorboard/ (tensorboard results)
        #       -> results/ (output files: images, illuminant, GT, etc...)
        #       -> checkpoint.pth.tar (checkpoint to continue training in case of failure)
        #       -> model_best.pth.tar (best checkpoint, for inference)
        self._pretrained_model = None
        if not self._inference:
            output_dir = os.path.join(self._args.outputfolder, str(self._fold))
            self._tensorboard_dir = os.path.join(output_dir, 'tensorboard')
            self._results_dir = os.path.join(output_dir, 'results')
            self._best_checkpoint_file = os.path.join(output_dir, 'model_best.pth.tar')
            self._checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
            self._pretrained_model = self._args.pretrainedmodel

            # create all directories
            os.makedirs(self._tensorboard_dir, exist_ok=True)
        else:
            # for inference all results are saved under the output directory
            # (images, illuminant, GT, etc...)
            self._results_dir = self._args.outputfolder
            if isinstance(self._args.checkpointfile, list):
                self._checkpoint_file = self._args.checkpointfile[fold]
            else:
                self._checkpoint_file = self._args.checkpointfile

        self._display = Display(self._conf)
        self._factory = Factory(self._conf, self._data_conf, cache_manager, self._args, verbose)
        self._cache_manager = cache_manager

        # create output directory
        os.makedirs(self._results_dir, exist_ok=True)

        os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'torch_model_zoo')

    # function used to determine the best epoch
    def _compute_best(self, best, train_stats, val_stats):
        metric = train_stats.mean_loss
        if 'choose_best_epoch_by' in self._conf:
            if self._conf['choose_best_epoch_by'] == 'mean_angular_error':
                metric = train_stats.mean_err
            elif self._conf['choose_best_epoch_by'] == 'median_angular_error':
                metric = train_stats.med_err
            elif self._conf['choose_best_epoch_by'] == 'mean_loss':
                metric = train_stats.mean_loss
            elif self._conf['choose_best_epoch_by'] == 'val_median_angular_error':
                metric = val_stats.med_err
            else:
                raise Exception('Invalid "choose_best_epoch_by" option')

        is_best = metric < best
        best = min(metric, best)

        return is_best, best

    # function to print the epoch info
    def _log_epoch(self, epoch, train_stats, val_stats):
        if self._verbose and epoch % self._conf['print_frequency_epoch'] == 0:
            print('Epoch [{}]: AE (mean={:.4f} med={:.4f}) loss {:.4f} time={:.1f}'.format(
                epoch, train_stats.mean_err, train_stats.med_err, train_stats.mean_loss, train_stats.time), end='')
            if val_stats is not None:
                print(' (val: AE (mean={:.4f} med={:.4f}) loss={:.4f} time={:.4f})\t'.format(
                    val_stats.mean_err, val_stats.med_err, val_stats.mean_loss, val_stats.time), end='')
            print()

        # 1. Log scalar values (scalar summary)
        info = { 'Epoch Loss': train_stats.mean_loss, 'Epoch Mean AE': train_stats.mean_err, 'Epoch Median AE': train_stats.med_err }
        if val_stats is not None:
            info.update({'Epoch Loss (validation)': val_stats.mean_loss, 'Epoch Mean AE (validation)': val_stats.mean_err, 'Epoch Median AE (validation)': val_stats.med_err })

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch)

    def run(self):
        args = self._args
        gpu = args.gpu
        self._conf['use_gpu'] = gpu is not None

        if self._verbose:
            if gpu is not None:
                print("Using GPU: {}".format(gpu))
            else:
                print("WARNING: You're training on the CPU, this could be slow!")

        # create transforms
        transforms = create_all_transforms(self, self._conf['transforms'])
        # copy FFCC histogram settings to conf (from transform settings)
        self._conf['log_uv_warp_histogram'] = find_loguv_warp_conf(transforms)

        # create model
        self.model = self._factory.get_model()

        # if we're evaluating instead of training:
        # 1. init the model (without training illuminants)
        # 2. load model weights
        if args.evaluate:
            self.model.initialize()
            if self._inference:
                checkpoint = self._checkpoint_file
            else:
                checkpoint = self._best_checkpoint_file

            # optionally resume from a checkpoint
            start_epoch, best, self.model = self._factory.resume_from_checkpoint(checkpoint, self.model, None, gpu)
        else:
            checkpoint = self._checkpoint_file

        # create validation/test transforms if defined, otherwise, the same as training
        if self._conf['transforms_valtest'] is not None:
            transforms_valtest = create_all_transforms(self, self._conf['transforms_valtest'])
        else:
            transforms_valtest = transforms

        if gpu is not None:
            torch.cuda.set_device(gpu)
            cudnn.benchmark = True

        if args.testfile is not None:
            # test loader
            test_dataset, test_loader, test_loader_cache = self._factory.get_loader(args.testfile, transforms_valtest, gpu)
            # if evaluating, copy model to GPU, evaluate and die
            if args.evaluate:
                if gpu is not None:
                    self.model = self.model.cuda(gpu)
                return self.validate(test_loader) # we finish here!

        # if validation file is defined
        if args.valfile is not None:
            # to save memory, don't do it again if valfile==testfile
            if args.valfile == args.testfile:
                val_dataset = test_dataset
                val_loader = test_loader
                val_loader_cache = test_loader_cache
            else:
                # validation loader
                val_dataset, val_loader, val_loader_cache = self._factory.get_loader(args.valfile, transforms_valtest, gpu)

        # training loader
        train_dataset, train_loader, train_loader_cache = self._factory.get_loader(args.trainfiles, transforms, gpu, train = True)

        # init model with the training set illuminants
        self.model.initialize(train_dataset.get_illuminants_by_sensor())

        # optionally pretrain model
        self._factory.pretrain_model(self._pretrained_model, self.model)

        # optionally resume from a checkpoint
        self.optimizer, optimizer_name = self._factory.get_optimizer(self.model)
        start_epoch, best, self.model = self._factory.resume_from_checkpoint(checkpoint, self.model, self.optimizer, gpu)

        # define loss function
        self.criterion = self._factory.get_criterion()

        # tensorboard logger
        self.logger = TensorBoardLogger(self._tensorboard_dir)

        # learning rate scheduler (if defined)
        scheduler, scheduler_name = self._factory.get_lr_scheduler(start_epoch, self.optimizer)

        # copy stuff to GPU
        if gpu is not None:
            self.criterion = self.criterion.cuda(gpu)
            self.model = self.model.cuda(gpu)

        # for FFCC, we reset the optimizer after some epochs
        # because they use two loss functions, ugly trick
        # TODO: fix
        reset_opt = -1
        if 'reset_optimizer_epoch' in self._conf:
            reset_opt = self._conf['reset_optimizer_epoch']

        # load data for the first time
        # we use the cache loaders, they define batch size=1
        # so that we can see the progress with tqdm
        if self._cache_manager.transforms().length > 0 and self._fold == 0:
            if self._verbose:
                print('Caching images...')
            for data in tqdm(train_loader_cache, desc="Training set", disable=not self._verbose):
                pass
            if args.testfile is not None:
                for data in tqdm(test_loader_cache, desc="Test set", disable=not self._verbose):
                    pass
            if args.valfile is not None and args.testfile != args.valfile:
                for data in tqdm(val_loader_cache, desc="Validation set", disable=not self._verbose):
                    pass

        # if epochs==0, we don't really want to train,
        # we only want to do the candidate selection process for our method
        if self._conf['epochs'] == 0:
            print('WARNING: Training 0 epochs')
            checkpoint = {
                'epoch': 0,
                'arch': self._conf['network']['arch'],
                'subarch': self._conf['network']['subarch'],
                'state_dict': self.model.state_dict(),
                'best': float("inf"),
                'optimizer': self.optimizer.state_dict()
            }
            self._factory.save_checkpoint(self._checkpoint_file, self._best_checkpoint_file, checkpoint, is_best=True)

        # epoch loop
        for epoch in range(start_epoch, self._conf['epochs']):
            # ugly trick for FFCC 2 losses
            if epoch == reset_opt:
                if self._verbose:
                    print('Reset optimizer and lr scheduler')
                best = float("inf")
                self.optimizer, optimizer_name = self._factory.get_optimizer(self.model)
                # TODO: What if lr scheduler changes its internal API?
                if scheduler is not None:
                    scheduler.optimizer = self.optimizer

            # train for one epoch
            train_stats = self.train(train_loader, epoch)

            # validation
            val_stats = None
            if args.valfile is not None:
                _, val_stats = self.validate(val_loader, epoch)

            # compute the best training epoch
            is_best, best = self._compute_best(best, train_stats, val_stats)

            # log epoch details
            self._log_epoch(epoch, train_stats, val_stats)

            # learning rate scheduler
            if scheduler is not None:
                # TODO: hardcoded
                if scheduler_name == 'ReduceLROnPlateau':
                    scheduler.step(train_stats.mean_err)
                else:
                    scheduler.step()

            # save checkpoint!
            checkpoint = {
                'epoch': epoch + 1,
                'arch': self._conf['network']['arch'],
                'subarch': self._conf['network']['subarch'],
                'state_dict': self.model.state_dict(),
                'best': best,
                'optimizer': self.optimizer.state_dict()
            }
            self._factory.save_checkpoint(self._checkpoint_file, self._best_checkpoint_file, checkpoint, is_best)

        # get results for the best model
        start_epoch, best, self.model = self._factory.load_model(self._best_checkpoint_file, self.model, self.optimizer, gpu)

        # return results from best epoch
        if args.testfile is not None:
            start_time = time.time()
            results = self.validate(test_loader)
            if self._verbose:
                print('Final inference (including generation of output files) took {:.4f}'.format(time.time()-start_time))
            return results
        else:
            # for some datasets, we have no validation ground truth,
            # so, no evaluation possible
            return [], EpochStats(-1, -1, -1, 0)

    # log iteration
    def _log_iteration(self, epoch, step, len_epoch, loss, err, data, output):
        real_step = epoch*len_epoch + step
        if self._conf['tensorboard_frequency'] != -1 and real_step % self._conf['tensorboard_frequency'] == 0:
            # Log scalar values (scalar summary)
            info = { 'Loss': loss, 'Angular Error': err }

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, real_step)

            # Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                if value.requires_grad:
                    if value.grad is None:
                        print('WARNING: variable ',tag,'.grad is None!')
                    else:
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), real_step)
                        self.logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), real_step)

            if 'confidence' in output:
                self.logger.histo_summary('confidence', output['confidence'].data.cpu().numpy().flatten(), real_step)

        if self._conf['tensorboard_frequency_im'] != -1 and real_step % self._conf['tensorboard_frequency_im'] == 0:
            # Log training images (image summary)
            info = self._display.get_images(data, output)

            for tag, images in info.items():
                self.logger.image_summary(tag, images, real_step)

    def train(self, train_loader, epoch):
        start_t = time.time() # log starting time
        self.model.train() # switch to train mode

        # angular errors and loss lists
        angular_errors = []
        loss_vec = []

        # batch loop
        for step, data in enumerate(train_loader):
            data['epoch'] = epoch # we know what's the current epoch
            err = err_m = output = loss = None
            def closure():
                nonlocal err, err_m, output, loss

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, data, self.model)
                loss.backward()
                err_m = angular_error_degrees(output['illuminant'], Variable(data['illuminant'], requires_grad=False)).detach()
                err = err_m.sum().item() / err_m.shape[0]
                return loss

            self.optimizer.step(closure)
            angular_errors += err_m.cpu().data.tolist()
            loss_value = loss.detach().item()
            loss_vec.append(loss_value)

            self._log_iteration(epoch, step, len(train_loader), loss_value, err, data, output)

        angular_errors = np.array(angular_errors)
        mean_err = angular_errors.mean()
        med_err = np.median(angular_errors)

        mean_loss = np.array(loss_vec).mean()

        t = time.time() - start_t
        return EpochStats(mean_err, med_err, mean_loss, t)

    def validate(self, val_loader, epoch=None):
        with torch.no_grad(): # don't compute gradients
            save_full_res = self._args.save_fullres
            training = epoch is not None
            start_t = time.time()
            # switch to evaluate mode
            self.model.eval()

            res = []
            angular_errors = []
            loss_vec = []

            for i, data in enumerate(val_loader):
                if training:
                    data['epoch'] = epoch
                # compute output
                output = self.model(data)

                # measure accuracy and save loss
                err = None
                if 'illuminant' in data:
                    if training:
                        loss = self.criterion(output, data, self.model)
                        loss_vec.append(loss.detach().item())
                    err = angular_error_degrees(output['illuminant'], Variable(data['illuminant'], requires_grad=False)).data.cpu().tolist()
                    angular_errors += err

                # When training, we don't want to save validation images
                if not training:
                    res += self._display.save_output(data, output, err, val_loader.dataset, self._results_dir, save_full_res)

            # some datasets have no validation GT
            mean_err = med_err = mean_loss = -1

            if len(angular_errors) > 0:
                angular_errors = np.array(angular_errors)
                mean_err = angular_errors.mean()
                med_err = np.median(angular_errors)

            if len(loss_vec) > 0:
                mean_loss = np.array(loss_vec).mean()

            t = time.time() - start_t
            return res, EpochStats(mean_err, med_err, mean_loss, t)
