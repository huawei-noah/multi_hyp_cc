from core.utils import *
import sys
import shutil
import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from datasets.base_dataset import BaseDataset

# class to create all models, optimizers, etc...
class Factory:
    def __init__(self, conf, data_conf, cache_manager, args, verbose):
        self._conf = conf
        self._data_conf = data_conf
        self._cache_manager = cache_manager
        self._args = args
        self._verbose = verbose

    # generate CNN model
    def get_model(self):
        if self._verbose:
            str_model = 'using pre-trained model' if self._conf['pretrained'] else 'creating model'
            print("=> {} '{}/{}'".format(str_model, self._conf['network']['arch'], self._conf['network']['subarch']))
        # import from /models/ directory
        model_class = import_shortcut('models', self._conf['network']['arch'])
        return model_class(self._conf, pretrained=self._conf['pretrained'], **self._conf['network']['params'])

    # generate optimizer
    def get_optimizer(self, model):
        opt_params = None
        if 'params' in self._conf['optimizer']:
            opt_params = self._conf['optimizer']['params']
        optimizer_name = self._conf['optimizer']['name']
        # from pytorch
        return create_optimizer(optimizer_name, filter(lambda p: p.requires_grad,model.parameters()), opt_params), optimizer_name

    def resume_from_checkpoint(self, checkpoint_file, model, optimizer = None, use_gpu = None):
        # optionally resume from a checkpoint
        best = float("inf")
        start_epoch = 0
        if self._args.resume or self._args.evaluate:
            if os.path.isfile(checkpoint_file):
                start_epoch, best, model = self.load_model(checkpoint_file, model, optimizer, use_gpu)
                if self._verbose:
                    print("=> loaded checkpoint '{}' (epoch {}, best {:.4f})"
                          .format(checkpoint_file, start_epoch, best))
            else:
                if self._verbose:
                    print("=> no checkpoint found at '{}'".format(checkpoint_file))
                if self._args.evaluate:
                    print("Can't evaluate without the trained models")
                    sys.exit(-1)
        return start_epoch, best, model

    def get_lr_scheduler(self, start_epoch, optimizer):
        scheduler_conf = self._conf['learning_rate_scheduler']
        last_epoch = max(start_epoch-1, 0)
        scheduler_name = None
        if scheduler_conf is None:
            scheduler = None
        else:
            scheduler_name = scheduler_conf['name']
            # create pytorch learning rate scheduler
            if last_epoch != 0:
                scheduler = lr_scheduler.__dict__[scheduler_name](optimizer, **scheduler_conf['params'], last_epoch=last_epoch)
            else:
                scheduler = lr_scheduler.__dict__[scheduler_name](optimizer, **scheduler_conf['params'])

        return scheduler, scheduler_name

    def get_criterion(self):
        # create loss function from /loss/
        return import_shortcut('loss', self._conf['loss']['name'])(self._conf, **self._conf['loss']['params'])

    def get_loader(self, dataset_file, transforms, gpu, train = False):
        required_input = None
        if 'required_input' in self._conf:
            required_input = self._conf['required_input']

        # import dataset class form /datasets/
        dataset_class = import_shortcut('datasets', self._args.dataset)

        # create BaseDataset wrapper
        base_dataset = BaseDataset(self._args.subdataset,
                    self._data_conf, dataset_file,
                    dataset_class, self._conf, gpu,
                    transforms, required_input,
                    self._cache_manager, train == False, self._verbose)

        batch_size = self._conf['batch_size'] # batch size
        shuffle_images = True # whether to randomly shuffle images

        # no need to shuffle if not training
        if train is False:
            shuffle_images = False

        # batch size=-1 means that we use the whole dataset for a batch
        if batch_size == -1:
            batch_size = len(base_dataset)
            shuffle_images = False # no need to shuffle

        sampler = None
        # import the sampler from /samplers/
        if 'sampler' in self._conf:
            sampler_class = import_shortcut('samplers', self._conf['sampler'])
            sampler = sampler_class(base_dataset, batch_size)

        # create pytorch dataloader
        if sampler is None:
            loader = torch.utils.data.DataLoader(
                base_dataset,
                batch_size=batch_size,
                shuffle=shuffle_images,
                num_workers=self._args.workers)
        else:
            loader = torch.utils.data.DataLoader(
                base_dataset,
                batch_sampler=sampler,
                num_workers=self._args.workers)


        # create a cache dataloader with batch size=1
        # we use this to show progress with tqdm
        cache_loader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=1, shuffle=False,
            num_workers=self._args.workers)

        return base_dataset, loader, cache_loader

    # load pretrained model
    def pretrain_model(self, pretrained_model, model):
        if pretrained_model is not None:
            if os.path.isfile(pretrained_model):
                if self._verbose:
                    print("=> loading pretrained model '{}'".format(pretrained_model))
                self._pretrain_model(pretrained_model, model)
                if self._verbose:
                    print("=> loaded pretrained model '{}'"
                          .format(pretrained_model))
            else:
                print("=> no pretrained model found at '{}'".format(pretrained_model))
                sys.exit(-1)

    def _pretrain_model(self, state_file, model):
        checkpoint = torch.load(state_file)
        state_dict = checkpoint['state_dict']
        strict = True
        # skip some weights according to the conf file ('pretrained_skip')
        if 'pretrained_skip' in self._conf:
            for key in self._conf['pretrained_skip']:
                if key in state_dict:
                    print('deteled key: '+ key)
                    del state_dict[key]
                    strict = False

        model.load_state_dict(state_dict, strict = strict)

    def load_model(self, state_file, model, optimizer = None, use_gpu = None):
        # load model from checkpoint file:
        # get recover the best checkpoint, last epoch, and optimizer state
        checkpoint = torch.load(state_file)
        start_epoch = checkpoint['epoch']
        best = checkpoint['best']
        model.load_state_dict(checkpoint['state_dict'])
        if use_gpu is not None:
            model = model.cuda(use_gpu)
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return start_epoch, best, model

    # save checkpoint
    def save_checkpoint(self, checkpoint_file, best_checkpoint_file, state, is_best):
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, best_checkpoint_file)
