from typing import Callable, List, Dict, Optional, Set, cast

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.beam_search_decoder import (BeamSearchDecoder,
                                                       SearchStepOutput)
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN


class BeamSearchExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_encoders: Set[ModelPart],
                 bs_outputs: SearchStepOutput,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Callable]) -> None:

        self._rank = rank
        self._all_encoders = all_encoders
        self._bs_outputs = bs_outputs
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        return self._all_encoders, {"bs_outputs": self._bs_outputs}, {}

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) > 1:
            raise ValueError("Beam search runner does not support ensembling.")

        evaluated_bs = results[0]["bs_outputs"]
        max_time = evaluated_bs.scores.shape[0]

        # pick the end of the hypothesis based on its rank
        hyp_index = np.argpartition(
            -evaluated_bs.scores[-1], self._rank - 1)[self._rank - 1]
        bs_score = evaluated_bs.scores[-1][hyp_index]

        # now backtrack
        output_tokens = []  # type: List[str]
        for time in reversed(range(max_time)):
            token_id = evaluated_bs.token_ids[time][hyp_index]
            token = self._vocabulary.index_to_word[token_id]
            output_tokens.append(token)
            hyp_index = evaluated_bs.parent_ids[time][hyp_index]

        output_tokens.reverse()

        before_eos_tokens = []  # type: List[str]
        for tok in output_tokens:
            if tok == END_TOKEN:
                break
            before_eos_tokens.append(tok)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess([before_eos_tokens])
        else:
            decoded_tokens = [before_eos_tokens]

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[bs_score],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class BeamSearchRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        if rank < 1 or rank > decoder.beam_size:
            raise ValueError(
                ("Rank of output hypothesis must be between 1 and the beam "
                 "size ({}), was {}.").format(decoder.beam_size, rank))

        self._rank = rank
        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> BeamSearchExecutable:
        decoder = cast(BeamSearchDecoder, self._decoder)

        return BeamSearchExecutable(
            self._rank, self.all_coders, decoder.outputs,
            decoder.vocabulary, self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]

    @property
    def decoder_data_id(self) -> Optional[str]:
        return None


def beam_search_runner_range(output_series: str,
                             decoder: BeamSearchDecoder,
                             max_rank: int = None,
                             postprocess: Callable[
                                 [List[str]], List[str]]=None
                            ) -> List[BeamSearchRunner]:
    """Return beam search runners for a range of ranks from 1 to max_rank.

    This means there is max_rank output series where the n-th series contains
    the n-th best hypothesis from the beam search.

    Args:
        output_series: Prefix of output series.
        decoder: Beam search decoder shared by all runners.
        max_rank: Maximum rank of the hypotheses.
        postprocess: Series-level postprocess applied on output.

    Returns:
        List of beam search runners getting hypotheses with rank from 1 to
        max_rank.
    """
    check_argument_types()

    if max_rank is None:
        max_rank = decoder.beam_size

    if max_rank > decoder.beam_size:
        raise ValueError(
            ("The maximum rank ({}) cannot be "
             "bigger than beam size {}.").format(
                 max_rank, decoder.beam_size))

    return [BeamSearchRunner("{}.rank{:03d}".format(output_series, r),
                             decoder, r, postprocess)
            for r in range(1, max_rank + 1)]
