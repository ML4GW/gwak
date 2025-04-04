import lightning.pytorch as pl

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
        
