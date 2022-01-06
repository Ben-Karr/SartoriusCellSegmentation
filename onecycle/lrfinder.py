from fvcore.common.param_scheduler import ExponentialParamScheduler, ParamScheduler, LinearParamScheduler
from fvcore.common.config import CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.hooks import LRScheduler
import detectron2.engine.hooks as hooks
from detectron2.engine import DefaultTrainer
from detectron2.utils import comm

class CancelTrainException(Exception): pass

class LRFindScheduler(LRScheduler):
    
    best_loss = float('inf')
    
    def after_step(self):
        super().after_step()
        loss = self.trainer.storage._latest_scalars['total_loss'][0]
        if loss < self.best_loss:
            self.best_loss = loss
        if loss > 5 * self.best_loss:
            raise CancelTrainException

class LRFinder(DefaultTrainer):
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        start_lr = cfg.LRFINDER.START_LR
        end_lr = cfg.LRFINDER.END_LR
        if cfg.LRFINDER.METHOD == 'exponential':
            return ExponentialParamScheduler(start_lr, end_lr/start_lr)
        else:
            return LinearParamScheduler(start_lr, end_lr)
    
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
    

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            LRFindScheduler(), # Instead of LRScheduler
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ###ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
    
    def train(self):
        try:
            super().train()
        except:
            print('Stopping Criterion reached')