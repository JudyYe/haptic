import logging
import os
import os.path as osp
from typing import Callable, Dict, Mapping, Optional, Sequence, Union
from pytorch_lightning import Callback
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LoggerCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def _on_batch_start(self, trainer, pl_module) -> None:
        if hasattr(pl_module.logger, 'commit'):
            pl_module.logger.commit(step=pl_module.global_step)
        else:
            logging.info('no commit avaialbe')

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs) -> None:
        return self._on_batch_start(trainer, pl_module)
    
    def on_validation_batch_start(self, trainer, pl_module, *args, **kwargs) -> None:
        return self._on_batch_start(trainer, pl_module)



class LFSLogger(Logger):
    """
    local system logger
    """
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__() 
        self._save_dir = save_dir + '/log'
        os.makedirs(self._save_dir, exist_ok=True)
        self._name = name
        self._version = version

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def log_dir(self) -> Optional[str]:
        return self._save_dir
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> Union[int, str]:
        return self._version
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        print('Global Step [%05d] ' % step)
        for k,v in metrics.items():
            print(k, v)
            # if isinstance(v, wandb.Image):
            #     fname = os.path.join(self.save_dir, '%d_%s.jpg' % (step, k.replace('/', '_')))
            #     os.makedirs(osp.dirname(fname), exist_ok=True)
            #     v.image.save(fname)
    
    def watch(self, *args, **kwargs):
        return 

    def unwatch(self, *args, **kwargs):
        return 

    @rank_zero_only
    def commit(self, step=None):
        return

    @rank_zero_only
    def log_hyperparams(self,*args, **kwargs) -> None:    
        return
    
    
class MyWandbLogger(WandbLogger):
    def __init__(self, name: Optional[str] = None, save_dir: Optional[str] = None, offline: Optional[bool] = False, id: Optional[str] = None, anonymous: Optional[bool] = None, version: Optional[str] = None, project: Optional[str] = None, log_model: Union[str, bool] = False, experiment=None, prefix: Optional[str] = "", agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None, agg_default_func: Optional[Callable[[Sequence[float]], float]] = None, **kwargs):
        super().__init__(name, save_dir, offline=offline, id=id, anonymous=anonymous, version=version, 
            project=project, log_model=log_model, experiment=experiment, prefix=prefix, **kwargs)
        self.to_commit = {}
        self.to_commit_step = -1
    
    @rank_zero_only
    def commit(self, step=None):
        """the original log_metrics"""
        prev_step = self.to_commit_step
        if step > prev_step: 
            if len(self.to_commit) > 0:
                if step is not None:
                    self.to_commit["trainer/global_step"] = max(prev_step, 0)
                self.experiment.log(self.to_commit, step=max(0, prev_step))
                self.to_commit.clear()
            self.to_commit_step = step
        else:
            # continue to accumulate log
            pass
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """change to: cache to self.to_commit"""
        # assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        # metrics = {f'{self.}'}
        # metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.to_commit.update(metrics)



def build_logger(expname, save_dir, log='none', project_name='hawc_'):
    if log == 'none':
        from .logger import LFSLogger
        log = LFSLogger(project=project_name + osp.dirname(expname),
            name=osp.basename(expname),
            save_dir=save_dir,
        )
    elif log == 'wandb':
        os.makedirs(save_dir + '/log/wandb', exist_ok=True)
        kwargs = {}

        runid = None
        if os.path.exists(f"{save_dir}/runid.txt"):
            runid = open(f"{save_dir}/runid.txt").read().strip()
        log = MyWandbLogger(
            project=project_name + osp.dirname(expname),
            name=osp.basename(expname),
            save_dir=osp.join(save_dir, 'log'),
            id=runid,
            save_code=True,
            settings=wandb.Settings(start_method='fork'),
            **kwargs,
        )
        runid = log.experiment.id
        @rank_zero_only
        def save_runid():
            with  open(f"{save_dir}/runid.txt", 'w') as fp:
                fp.write(runid)

        save_runid()
    else:
        raise NotImplementedError

    return log

