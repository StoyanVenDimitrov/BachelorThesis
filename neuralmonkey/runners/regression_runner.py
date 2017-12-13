from typing import Dict, List, Callable, Set, cast

import numpy as np
import tensorflow as tf

from typeguard import check_argument_types
from neuralmonkey.decoders.sequence_regressor import SequenceRegressor
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
# pylint: disable=too-few-public-methods


class RegressionRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: SequenceRegressor,
                 postprocess: Callable[[float], float] = None) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> Executable:
        decoder = cast(SequenceRegressor, self._decoder)

        if compute_losses:
            fetches = {"mse": decoder.cost}
        else:
            fetches = {}

        fetches["prediction"] = decoder.predictions

        return RegressionRunExecutable(self.all_coders, fetches,
                                       self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["mse"]


class RegressionRunExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: Dict[str, tf.Tensor],
                 postprocess: Callable[[float], float] = None) -> None:

        self.all_coders = all_coders
        self._fetches = fetches
        self._postprocess = postprocess
        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        predictions_sum = np.zeros_like(results[0]["prediction"])
        mse_loss = 0.

        for sess_result in results:
            if "mse" in sess_result:
                mse_loss += sess_result["mse"]

            predictions_sum += sess_result["prediction"]

        predictions = predictions_sum / len(results)

        if self._postprocess is not None:
            predictions = self._postprocess(predictions)

        self.result = ExecutionResult(
            outputs=predictions.tolist(),
            losses=[mse_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
