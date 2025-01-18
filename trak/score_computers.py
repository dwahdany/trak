"""
Computing scores for the TRAK algorithm from pre-computed projected gradients
involves a number of matrix multiplications. This module contains classes that
perform these operations. The :code:`AbstractScoreComputer` class defines the
interface for score computers. Then, we provide two implementations:
- :class:`BasicSingleBlockScoreComputer`: A bare-bones implementation, mostly for
    testing purposes.
- :class:`BasicScoreComputer`: A more sophisticated implementation that does
    block-wise matrix multiplications to avoid OOM errors.

"""

import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .utils import get_matrix_mult

ch = torch


class AbstractScoreComputer(ABC):
    """
    The :code:`ScoreComputer` class
    Implementations of the ScoreComputer class must implement three methods:
    - :code:`get_xtx`
    - :code:`get_x_xtx_inv`
    - :code:`get_scores`
    """

    @abstractmethod
    def __init__(self, dtype, device) -> None:
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def get_xtx(self, grads: Tensor) -> Tensor:
        """Computes :math:`X^\top X`, where :math:`X` is the matrix of projected
        gradients. Here, the shape of :math:`X` is :code:`(n, p)`, where
        :math:`n` is the number of training examples and :math:`p` is the
        dimension of the projection.


        Args:
            grads (Tensor): projected gradients of shape :code:`(n, p)`.

        Returns:
            Tensor: :math:`X^\top X` of shape :code:`(p, p)`.
        """

    @abstractmethod
    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        """Computes :math:`X(X^\top X)^{-1}`, where :math:`X` is the matrix of
        projected gradients. Here, the shape of :math:`X` is :code:`(n, p)`,
        where :math:`n` is the number of training examples and :math:`p` is the
        dimension of the projection. This function takes as input the
        pre-computed :math:`X^\top X` matrix, which is computed by the
        :code:`get_xtx` method.

        Args:
            grads (Tensor): projected gradients :math:`X` of shape :code:`(n, p)`.
            xtx (Tensor): :math:`X^\top X` of shape :code:`(p, p)`.

        Returns:
            Tensor: :math:`X(X^\top X)^{-1}` of shape :code:`(n, p)`.
        """

    @abstractmethod
    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> None:
        """Computes the scores for a given set of features and target gradients.
        In particular, this function takes in a matrix of features
        :math:`\Phi=X(X^\top X)^{-1}`, computed by the :code:`get_x_xtx_inv`
        method, and a matrix of target (projected) gradients :math:`X_{target}`.
        Then, it computes the scores as :math:`\Phi X_{target}^\top`.  The
        resulting matrix has shape :code:`(n, m)`, where :math:`n` is the number
        of training examples and :math:`m` is the number of target examples.

        The :code:`accumulator` argument is used to store the result of the
        computation. This is useful when computing scores for multiple model
        checkpoints, as it allows us to re-use the same memory for the score
        matrix.

        Args:
            features (Tensor): features :math:`\Phi` of shape :code:`(n, p)`.
            target_grads (Tensor):
                target projected gradients :math:`X_{target}` of shape
                :code:`(m, p)`.
            accumulator (Tensor): accumulator of shape :code:`(n, m)`.
        """


class BasicSingleBlockScoreComputer(AbstractScoreComputer):
    """A bare-bones implementation of :code:`ScoreComputer` that will likely
    OOM for almost all applications. Here for testing purposes only. Unless you
    have a good reason not to, you should use :func:`BasicScoreComputer`
    instead.
    """

    def get_xtx(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        # torch.linalg.inv does not support float16
        return grads @ ch.linalg.inv(xtx.float()).to(self.dtype)

    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> None:
        accumulator += (features @ target_grads.T).detach().cpu()


class BasicScoreComputer(AbstractScoreComputer):
    """An implementation of :code:`ScoreComputer` that computes matmuls in a
    block-wise manner.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        CUDA_MAX_DIM_SIZE: int = 20_000,
        logging_level=logging.INFO,
        lambda_reg: float = 0.0,
    ) -> None:
        """
        Args:
            dtype (torch.dtype):
            device (Union[str, torch.device]):
                torch device to do matmuls on
            CUDA_MAX_DIM_SIZE (int, optional):
                Size of block for block-wise matmuls. Defaults to 100_000.
            logging_level (logging level, optional):
                Logging level for the logger. Defaults to logging.info.
            lambda_reg (int):
                regularization term for l2 reg on xtx
        """
        super().__init__(dtype, device)
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE
        self.logger = logging.getLogger("ScoreComputer")
        self.logger.setLevel(logging_level)
        self.lambda_reg = lambda_reg

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = ch.zeros(
            self.proj_dim, self.proj_dim, dtype=self.dtype, device=self.device
        )
        blocks = ch.split(
            grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0
        )

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        xtx_reg = xtx + self.lambda_reg * torch.eye(
            xtx.size(dim=0), device=xtx.device, dtype=xtx.dtype
        )
        xtx_inv = ch.linalg.inv(xtx_reg.to(ch.float32))

        # center X^TX inverse a bit to avoid numerical issues when going to float16
        xtx_inv /= xtx_inv.abs().mean()
        xtx_inv = xtx_inv.to(self.dtype)

        # Move everything to CPU first to avoid device conflicts
        grads_cpu = grads.cpu().numpy()
        xtx_inv_cpu = xtx_inv.cpu().numpy()

        # Create JAX arrays on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            grads_jax = jnp.array(grads_cpu)
            xtx_inv_jax = jnp.array(xtx_inv_cpu)

            n = grads_jax.shape[0]
            factors = [i for i in range(1, n + 1) if n % i == 0]
            smallest_factor_over_100 = next(
                f for f in sorted(factors) if f > 100
            )
            grads_blocks = jnp.split(grads_jax, smallest_factor_over_100)

        # Move xtx_inv to GPU once before the loop
        xtx_inv_gpu = jax.device_put(xtx_inv_jax, jax.devices("gpu")[0])

        # Process blocks on GPU
        result_blocks = []
        for block in tqdm(grads_blocks, desc="Processing blocks"):
            # Move block to GPU, compute, then back to CPU
            block_gpu = jax.device_put(block, jax.devices("gpu")[0])
            result_gpu = block_gpu @ xtx_inv_gpu
            result_blocks.append(jax.device_get(result_gpu))

        # Concatenate results on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            result_jax = jnp.concatenate(result_blocks)

        # Convert back to PyTorch tensor
        result = ch.from_numpy(np.array(result_jax)).to(dtype=self.dtype)

        return result

    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        self.logger.debug(f"{train_dim=}, {target_dim=}")

        accumulator += (
            get_matrix_mult(features=features, target_grads=target_grads)
            .detach()
            .cpu()
        )
