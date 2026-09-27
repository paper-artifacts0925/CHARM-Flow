"""Checkpoint callback for non-periodic source curriculum boundaries."""

from pathlib import Path

import pytorch_lightning as pl


class ScheduledStepCheckpoint(pl.Callback):
    """Save full trainer state at an explicit set of completed steps."""

    def __init__(self, dirpath, steps, filename="step={step}.ckpt"):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.steps = {int(step) for step in steps}
        self.filename = str(filename)
        self.saved_steps = set()

    @property
    def state_key(self):
        return f"{self.__class__.__qualname__}:{self.dirpath}"

    def state_dict(self):
        return {"saved_steps": sorted(self.saved_steps)}

    def load_state_dict(self, state_dict):
        self.saved_steps = {int(step) for step in state_dict.get("saved_steps", [])}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step not in self.steps or step in self.saved_steps:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / self.filename.format(step=step)
        trainer.save_checkpoint(str(path), weights_only=False)
        self.saved_steps.add(step)


__all__ = ["ScheduledStepCheckpoint"]
