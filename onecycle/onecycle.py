from torch.optim import Optimizer
from detectron2.engine.train_loop import HookBase
import detectron2.engine.hooks as hooks
from detectron2.engine import DefaultPredictor, DefaultTrainer, BestCheckpointer
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.utils import comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.config import CfgNode
from fvcore.common.param_scheduler import ParamScheduler, CosineParamScheduler, CompositeParamScheduler
import pycocotools.mask as mask_util
import detectron2.data.transforms as T
import warnings
from functools import wraps
import weakref
import torch
from typing import List, Dict
from collections import Counter
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from onecycle.lrfinder import LRFinder

class _MultiParamScheduler(object):

    def __init__(self, optimizer, param_names = ['lr'], last_epoch = -1, verbose = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_names = param_names

        # Initialize epoch and base parameters
        if last_epoch == -1:
            for group in optimizer.param_groups:
                for param in param_names:
                    group.setdefault('initial_' + param, group[param])
        else:
            for i, group in enumerate(optimizer.param_groups):
                for param in param_names:
                    if param not in group:
                        raise KeyError("param {param} is not specified "
                                    "in param_groups[{}] when resuming an optimizer".format(param, i))

        self.base_params = {param: [group['initial_' + param] for group in optimizer.param_groups] for param in param_names}
        self.last_epoch = last_epoch

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_param(self, param):
        """ Return last computed parameter by current scheduler.
        """
        return self._last_params[param]

    def get_param(self, param):
        # Compute parameter using chainable form of the scheduler
        raise NotImplementedError

    def print_param(self, is_verbose, param, group, value, epoch = None):
        """Display the current parameter.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting {} of group {} to {:4e}.'.format(param, group, value))
            else:
                print('Epoch {:5d}: adjusting {} of group {} to {:4e}.'.format(epoch, param, group, value))

    def step(self, epoch = None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                            "initialization. Please, make sure to call `optimizer.step()` before "
                            "`lr_scheduler.step()`. See more details at "
                            "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                            "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                            "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                            "will result in PyTorch skipping the first value of the learning rate schedule. "
                            "See more details at "
                            "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_param_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_param_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_param_called_within_step = False

        with _enable_get_param_call(self):
            if epoch is None:
                self.last_epoch += 1
                param_values = {param: self.get_param(param) for param in self.param_names}
            else:
                warning.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                param_values = {}
                for param in self.param_names:
                    if hasattr(self, "_get_closed_form_param"):
                        param_values[param] = self._get_closed_form_param(param)
                    else:
                        param_values[param] = self.get_param(param)
        last_params = {}
        for param in self.param_names:
            for i, param_group in enumerate(self.optimizer.param_groups):
                value = param_values[param][i]
                param_group[param] = value
                self.print_param(self.verbose, param, i, value, epoch)
            last_params[param] = [group[param] for group in self.optimizer.param_groups]
        self._last_params = last_params

class MultiParamScheduler(_MultiParamScheduler):
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        schedulers: Dict[str, ParamScheduler], #### This shouldnt work?
        max_iter: int, 
        last_iter: int = -1
    ):
        self.schedulers = schedulers
        self._max_iter = max_iter
        super().__init__(optimizer, param_names = self.schedulers.keys(), last_epoch = last_iter)
    
    def state_dict(self):
        state_dict = {'base_' + param: self.base_params[param] for param in self.param_names}
        state_dict['last_epoch'] = self.last_epoch
        return state_dict

    def get_param(self, param) -> List[float]:
        multiplier = self.schedulers[param](self.last_epoch / self._max_iter)
        return [multiplier for _ in self.base_params[param]]

class MultiParamHook(HookBase):

    def __init__(self, optimizer=None, schedulers = None):

        self._optimizer = optimizer
        self._schedulers = schedulers
        self.schedulers = None

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        self.schedulers = MultiParamScheduler(
                self._optimizer,
                self._schedulers,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
        )
        self._best_param_group_id = MultiParamHook.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        best_param_group = self._optimizer.param_groups[self._best_param_group_id]
        values = {param: best_param_group[param] for param in self.schedulers.param_names}
        for param in self.schedulers.param_names:
            self.trainer.storage.put_scalar(param, values[param], smoothing_hint=False)
        self.schedulers.step()

    def state_dict(self):
        if isinstance(self.schedulers, _MultiParamScheduler):
            return self.schedulers.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.schedulers, _MultiParamScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.schedulers.load_state_dict(state_dict)

class CycleTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        #logger = logging.getLogger(__name__)
        #logger.info("Model:\n{}".format(model))
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.get('AUGS', False):
            augs = []
            ## Geometric augmentations
            if cfg.AUGS.rotation:
                augs.append(T.RandomRotation([-cfg.AUGS.rotation, cfg.AUGS.rotation], expand = False))
            if cfg.AUGS.crop_size:
                augs.append(T.RandomCrop(crop_type = 'absolute', crop_size = cfg.AUGS.crop_size))
            if cfg.AUGS.shortest_edge_size:
                augs.append(T.ResizeShortestEdge(cfg.AUGS.shortest_edge_size, cfg.INPUT.MAX_SIZE_TRAIN, 'choice'),)
            ## Flip augmentations
            if cfg.AUGS.hor_flip:
                augs.append(T.RandomFlip(prob = 0.5, horizontal=True, vertical = False))
            if cfg.AUGS.vert_flip:
                augs.append(T.RandomFlip(prob = 0.5, horizontal=False, vertical = True))
            ## Lighting augmentations
            if cfg.AUGS.contrast:
                augs.append(T.RandomContrast(1.0 - cfg.AUGS.contrast, 1.0 + cfg.AUGS.contrast))
            if cfg.AUGS.brightness:
                augs.append(T.RandomBrightness(1.0 - cfg.AUGS.brightness, 1.0 + cfg.AUGS.brightness))
            if cfg.AUGS.saturation:
                augs.append(T.RandomSaturation(1.0 - cfg.AUGS.saturation, 1.0 + cfg.AUGS.saturation))
            
            mapper = DatasetMapper(
                cfg, 
                is_train=True, 
                augmentations=augs, 
                use_instance_mask = True, 
                instance_mask_format = cfg.INPUT.MASK_FORMAT,
                recompute_boxes = True,
                image_format = 'BGR'
            )
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)
    
    def build_hooks(self):
        ret = super().build_hooks()
        
        ## Replace default LRScheduler (second in default hooks list) with 1cycle scheduler
        cfg = self.cfg.clone()
        lr_start = cfg.ONECYCLE.LR_MAX / cfg.ONECYCLE.DIV
        lr_middle = cfg.ONECYCLE.LR_MAX
        lr_end = cfg.ONECYCLE.LR_MAX / cfg.ONECYCLE.DIV_FINAL
        pct_start = cfg.ONECYCLE.PCT or 0.3
        moms = cfg.ONECYCLE.MOMS or (0.95, 0.85, 0.95)
        
        ret[1] = MultiParamHook(optimizer = None,
                                schedulers = {
                                    'lr': CompositeParamScheduler([CosineParamScheduler(lr_start, lr_middle),
                                                                   CosineParamScheduler(lr_middle, lr_end)],
                                                                  [pct_start, 1 - pct_start],
                                                                  ['rescaled', 'rescaled']
                                                                 ), 
                                    'momentum': CompositeParamScheduler([CosineParamScheduler(moms[0], moms[1]),
                                                                         CosineParamScheduler(moms[1], moms[2])],
                                                                        [pct_start, 1 - pct_start],
                                                                        ['rescaled', 'rescaled']
                                                                       )
                                }
                               ) 
        ret.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                          DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                          "MaP IoU",
                                          "max",
                                         )
                    )
        return ret

    def lr_find(self, start_lr = 1e-8, end_lr = 1e-2, method = 'exponential', max_iter = 100):
        assert(method in ['exponential', 'linear']), f'Unknown method: "{method}". Use either "exponential" or "linear".'
        cfg = self.cfg.clone()
        
        cfg.LRFINDER = CfgNode({'START_LR': start_lr, 'END_LR': end_lr, 'METHOD': method})
        
        cfg.SOLVER.MAX_ITER = max_iter
        ## values from lr_scheduler are multiplied by base_lr, base_lr = 1 to keep them unchanged
        cfg.SOLVER.BASE_LR = 1.
        ## Turn of evaluation
        cfg.TEST.EVAL_PERIOD = 0
        cfg.INPUT.MIN_SIZE_TRAIN = 0
        
        ## Save the current model to be used in LRFinder
        self.checkpointer.save('tmp')
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, 'tmp.pth')
        lr_finder = LRFinder(cfg)
        ## Load current model from original trainer
        #lr_finder.checkpointer.load(checkpoint_path)
        lr_finder.resume_or_load(False)
        print(cfg.SOLVER)
        print(cfg.LRFINDER)
        lr_finder.train()

        ## Visualize loss
        lrs = [x[0] for x in lr_finder.storage._history['lr']._data]
        losses = [x[0] for x in lr_finder.storage._history['total_loss']._data]
        losses = np.array(losses).cumsum()/np.arange(1,len(losses)+1) ## Average loss at step
        
        fig, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ## Clean up
        os.remove(checkpoint_path)
        return lrs
    
    
####
# Metrics
####
    
# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}
    
class AugTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        #logger = logging.getLogger(__name__)
        #logger.info("Model:\n{}".format(model))
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.get('AUGS', False):
            augs = []
            ## Geometric augmentations
            if cfg.AUGS.rotation:
                augs.append(T.RandomRotation([-cfg.AUGS.rotation, cfg.AUGS.rotation], expand = False))
            if cfg.AUGS.crop_size:
                augs.append(T.RandomCrop(crop_type = 'absolute', crop_size = cfg.AUGS.crop_size))
            if cfg.AUGS.shortest_edge_size:
                augs.append(T.ResizeShortestEdge(cfg.AUGS.shortest_edge_size, cfg.INPUT.MAX_SIZE_TRAIN, 'choice'),)
            ## Flip augmentations
            if cfg.AUGS.hor_flip:
                augs.append(T.RandomFlip(prob = 0.5, horizontal=True, vertical = False))
            if cfg.AUGS.vert_flip:
                augs.append(T.RandomFlip(prob = 0.5, horizontal=False, vertical = True))
            ## Lighting augmentations
            if cfg.AUGS.contrast:
                augs.append(T.RandomContrast(1.0 - cfg.AUGS.contrast, 1.0 + cfg.AUGS.contrast))
            if cfg.AUGS.brightness:
                augs.append(T.RandomBrightness(1.0 - cfg.AUGS.brightness, 1.0 + cfg.AUGS.brightness))
            if cfg.AUGS.saturation:
                augs.append(T.RandomSaturation(1.0 - cfg.AUGS.saturation, 1.0 + cfg.AUGS.saturation))
            
            mapper = DatasetMapper(
                cfg, 
                is_train=True, 
                augmentations=augs, 
                use_instance_mask = True, 
                instance_mask_format = cfg.INPUT.MASK_FORMAT,
                recompute_boxes = True,
                image_format = 'BGR'
            )
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)
