from src.training.base_trainer import BaseTrainer


class TrainerDecorator(BaseTrainer):
    """Base class for all OOP trainer decorators.

    Delegates any attribute not defined in the decorator to the wrapped
    trainer, traversing the full chain transparently. This means every
    decorator automatically exposes model, optimizer, device, etc.
    """

    def __init__(self, trainer: BaseTrainer):
        self._trainer = trainer

    def __getattr__(self, name: str):
        return getattr(self._trainer, name)
