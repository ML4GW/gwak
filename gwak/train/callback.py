import lightning.pytorch as pl
import torch

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams['init_args']
        )

        module.model.eval()

        trace = torch.jit.script(module.model.to("cpu"))
        save_dir = trainer.logger.log_dir or trainer.logger.save_dir

        with open(os.path.join(save_dir, "model_JIT.pt"), "wb") as f:
            torch.jit.save(trace, f)

class ValidationCallback(pl.Callback):
    def __init__(self):
        self.global_validation_step = 0

    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx, 
    ):

        # Store it in the trainer (or another accessible place)
        trainer.global_validation_step = self.global_validation_step

        # Increment global validation step
        self.global_validation_step += 1
        
