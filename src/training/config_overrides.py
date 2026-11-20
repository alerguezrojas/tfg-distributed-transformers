"""Sobrescrituras de la config desde la línea de comandos, comunes a los scripts de entrenamiento."""

import logging

logger = logging.getLogger(__name__)


def add_common_overrides(parser) -> None:
    """Añade al parser las opciones que sobrescriben la config en todas las estrategias."""
    parser.add_argument("--data-root", type=str, help="Override data.root")
    parser.add_argument("--seed", type=int, help="Override training.seed (repetitions)")


def apply_common_overrides(cfg: dict, args) -> None:
    """Aplica ``--data-root``, ``--epochs`` y ``--seed`` sobre ``cfg`` (in situ).

    Si ``--epochs`` deja ``epochs <= warmup_epochs`` se baja ``warmup_epochs`` a
    ``epochs - 1`` (mínimo 0) en vez de dejar que el validador falle; un YAML que
    ya traiga esa combinación sigue fallando en el validador.
    """
    if getattr(args, "data_root", None):
        cfg["data"]["root"] = args.data_root
    if getattr(args, "seed", None) is not None:
        cfg["training"]["seed"] = args.seed
    if getattr(args, "epochs", None):
        train = cfg["training"]
        train["epochs"] = args.epochs
        warmup = train.get("warmup_epochs")
        if isinstance(warmup, int) and warmup >= args.epochs:
            train["warmup_epochs"] = max(0, args.epochs - 1)
            logger.warning(
                "--epochs %d <= warmup_epochs %d: se baja warmup_epochs a %d",
                args.epochs, warmup, train["warmup_epochs"])
