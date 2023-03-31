from typing import Callable, Tuple, Dict, Any

import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel

from tml.ml_logging.torch_logging import logging

DEVICE = torch.device("cuda:0")


class ModelAndLoss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, loss_fn: Callable) -> None:
        """
        Initialize ModelAndLoss class.

        Args:
            model: torch module to wrap.
            loss_fn: Function for calculating loss, should accept logits and labels.
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch: "RecapBatch") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Runs model forward and calculates loss according to given loss_fn.

        Args:
            batch: Batch of data to process.

        Returns:
            Tuple of losses and a dictionary containing the updated outputs.
        """
        outputs = self.model_forward(batch)
        losses = self.calculate_loss(outputs, batch.labels.float(), batch.weights.float())

        outputs.update({"loss": losses, "labels": batch.labels, "weights": batch.weights})

        return losses, outputs

    def model_forward(self, batch: "RecapBatch") -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def calculate_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(logits, labels, weights)


def maybe_shard_model(model: torch.nn.Module, device: torch.device = DEVICE) -> torch.nn.Module:
    """
    Set up and apply DistributedModelParallel to a model if running in a distributed environment.

    Args:
        model: Model to wrap in DistributedModelParallel.
        device: Device to use for the model.

    Returns:
        Wrapped model if in a distributed environment, otherwise returns the input model directly.
    """
    if dist.is_initialized():
        logging.info("***** Wrapping in DistributedModelParallel *****")
        logging.info(f"Model before wrapping: {model}")
        model = DistributedModelParallel(module=model, device=device)
        logging.info(f"Model after wrapping: {model}")

    return model


def log_sharded_tensor_content(
    weight_name: str, table_name: str, weight_tensor: torch.Tensor
) -> None:
    """
    Log the content of EBC embedding layer.
    Only works for single GPU machines.

    Args:
        weight_name: Name of tensor, as defined in the model.
        table_name: Name of the EBC table the weight is taken from.
        weight_tensor: Embedding weight tensor.
    """
    logging.info(f"{weight_name}, {table_name}", rank=-1)
    logging.info(f"{weight_tensor.metadata()}", rank=-1)
    output_tensor = torch.zeros(*weight_tensor.size(), device=DEVICE)
    weight_tensor.gather(out=output_tensor)
    logging.info(f"{output_tensor}", rank=-1)
