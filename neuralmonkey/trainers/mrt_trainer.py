
from typing import Any, Dict, List, Callable, Iterable, Tuple
from itertools import takewhile
from collections import Counter

import tensorflow as tf
import numpy as np


from neuralmonkey.evaluators.gleu import GLEUEvaluator
from neuralmonkey.evaluators.bleu import BLEUEvaluator
from neuralmonkey.trainers.generic_trainer import (GenericTrainer,
                                                   Objective, NextExecute)
from neuralmonkey.runners.base_runner import (Executable, ExecutionResult)

from neuralmonkey.vocabulary import PAD_TOKEN, END_TOKEN


from typeguard import check_argument_types


def shen_mrt_loss_per_sentence(
        alpha,
        sample_logprobs,
        metric_scores,
        name='mrt_loss'):
    with tf.name_scope(name):
        alpha = tf.constant(alpha)
        sample_logprobs=tf.multiply(sample_logprobs,alpha)
        sample_logprobs=tf.subtract(sample_logprobs,tf.reduce_max(sample_logprobs))
        sample_probs=tf.exp(sample_logprobs)
        q_distr=tf.div(sample_probs,tf.reduce_sum(sample_probs))
        loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(tf.negative(metric_scores)), q_distr))

        return loss 

def rose_annealing_loss_per_sentence(
        sample_logprobs,
        metric_scores,
        name='annealing_loss'):
    with tf.name_scope(name):
        sample_logprobs=tf.subtract(sample_logprobs,tf.reduce_max(sample_logprobs))
        sample_probs=tf.exp(sample_logprobs)
        #sample_probs=tf.div(sample_probs,tf.reduce_sum(sample_probs))
        entropy=tf.reduce_sum(sample_probs*sample_logprobs)
        loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(tf.negative(metric_scores)), sample_probs))

        return loss,entropy
   

def mrt_objective(decoder, loss, weight=None, name='objective') -> Objective:
    """Get the mrt objective with mrt loss"""
    return Objective(
        name="{} - minimum-risk".format(decoder.name),
        decoder=decoder,
        loss=loss,
        gradients=None,
        weight=weight,
    )


class MinRiskTrainer(GenericTrainer):

    def __init__(self,
                 batch_size: int,
                 decoders: List[Any],
                 postprocess: Callable[[List[str]], List[str]],
                 num_of_samples: int,
                 annealing:bool,
                 target_also_in_samples:bool,
                 alpha: float,
                 l1_weight=False, l2_weight=False,
                 clip_norm=False, optimizer=None, global_step=None) -> None:
        self.decoders = decoders
        self.batch_size = batch_size
        self._num_of_samples = num_of_samples
        self.alpha = alpha
        self._postprocess = postprocess
        self.placeholders=[]
        
        self.placeholders.append(tf.placeholder(
            tf.float32,name="temperature"))
        
        assert check_argument_types()
        def _score_with_reward_function(references: np.array,
                                        hypotheses: np.array) -> np.array:
            """Score (time, batch) arrays with sentence-based reward function.

            Parts of the sentence after generated <pad> or </s> are ignored.
            BPE-postprocessing is also included.

            :param references: array of indices of references, shape (time, batch)
            :param hypotheses: array of indices of hypotheses, shape (time, batch)
            :return: an array of batch length with float rewards
            """
            rewards = []
            for refs, hyps in zip(references.transpose(), hypotheses.transpose()):
                ref_seq = []
                hyp_seq = []
                for r_token in refs:
                    token = self.decoders[0].vocabulary.index_to_word[r_token]
                    if token == END_TOKEN or token == PAD_TOKEN:
                        break
                    ref_seq.append(token)
                for h_token in hyps:
                    token = self.decoders[0].vocabulary.index_to_word[h_token]
                    if token == END_TOKEN or token == PAD_TOKEN:
                        break
                    hyp_seq.append(token)
                # join BPEs, split on " " to prepare list for evaluator
                refs_tokens = " ".join(ref_seq).replace("@@ ", "").split(" ")
                hyps_tokens = " ".join(hyp_seq).replace("@@ ", "").split(" ")
                reward = float(GLEUEvaluator.gleu([hyps_tokens], [[refs_tokens]]))
                rewards.append(reward)
            return np.array(rewards, dtype=np.float32)

        #The loop for executing the decoding, to obtain candidates and their probabilities
        with tf.name_scope('sampling'):
            self.sample_logprobs = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0, name="sample_loprobs")
            self.score = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0, name="scores")
            self.targets = self.decoders[0].train_inputs
            self.ids = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0, name="ids")
            self.entropy = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0, name="cand_entropy")


            for i in range(0,self._num_of_samples):
                if target_also_in_samples and i==0: 
                    logits, outputs, mask, sampled_ids = self.decoders[0]._decoding_loop(train_mode=True)
                else:
                    logits, outputs, mask, sampled_ids = self.decoders[0]._decoding_loop(train_mode=False,sample=True) # see decoder.py
                logprobs=-tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_ids,logits=logits)
                sentence_sum_logprobs = tf.reduce_sum(logprobs,0)
                score=tf.py_func(_score_with_reward_function,[self.targets,sampled_ids],tf.float32)
                self.score = self.score.write(i, score)
                self.sample_logprobs = self.sample_logprobs.write(i, sentence_sum_logprobs)
                self.ids = self.ids.write(i, sampled_ids)
                # Computing the entropy components needed for DA
                logs=tf.nn.log_softmax(logits)
                probs=tf.exp(logs)
                sen_len= self.decoders[0].max_output_len
                entropy=-tf.div(tf.reduce_sum(tf.reduce_sum(probs*logs, 2),0),sen_len)
                self.entropy=self.entropy.write(i,entropy)
            self.score = tf.transpose(self.score.stack())
            self.sample_logprobs = tf.transpose(self.sample_logprobs.stack())
            self.ids =tf.transpose(self.ids.stack())
            self.entropy = tf.reduce_sum(tf.transpose(self.entropy.stack()))
            self.temp =[]

        
        with tf.name_scope('compute_loss'):
          if annealing:
              output = tf.reduce_sum(
                  [rose_annealing_loss_per_sentence(
                      self.sample_logprobs[i],
                      self.score[i],
                      'Rose_Loss') for i in range(self.batch_size)],0)

              loss=output[0]+self.placeholders[0]*self.entropy
              tf.summary.scalar('loss', loss,
                                collections=["summary_train"])


          else:
              loss = tf.reduce_sum(
                  [shen_mrt_loss_per_sentence(
                      self.alpha,
                      self.sample_logprobs[i],
                      self.score[i],
                      'Shen_loss') for i in range(self.batch_size)],0)
 
              tf.summary.scalar('loss', loss,
                                collections=["summary_train"])
  
        objective = [mrt_objective(self.decoders[0], loss, None, 'objective')]

        # After defining the MRT objective, the gradients and optimization are done with the generic_trainer
        super().__init__(
            objective, l1_weight, l2_weight, clip_norm=clip_norm,
            optimizer=optimizer, global_step=global_step)

    def get_executable(
            self, compute_losses=True, summaries=True,num_sessions=1):
        assert compute_losses
        # all tensors to be executed
        return MRtrainExecutable(
            self.all_coders,
            num_sessions,
            self.train_op,
            self.losses,
            self.targets,
            self.decoders[0].vocabulary,
            self._postprocess,
            self.placeholders,
            self.temp,
            self.scalar_summaries if summaries else None,
            self.histogram_summaries if summaries else None)


class MRtrainExecutable(Executable):
    def __init__(
            self,
            all_coders,
            num_sessions,
            train_op,          
            loss,
            tar,
            vocabulary,
            postprocess,
            placeholder,
            temp,
            scalar_summaries,
            histogram_summaries):
        self.all_coders = all_coders
        self.num_sessions = num_sessions
        self.train_op = train_op
        self.tar = tar
        self.loss = loss
        self._vocabulary = vocabulary
        self._postprocess = postprocess
        self.placeholder = placeholder
        self.temp = temp
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries

        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        fetches = {}
        fetches['temp'] = self.temp
        fetches['train_op'] = self.train_op
        fetches['loss'] = self.loss
        fetches['tar'] = self.tar
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries

        return self.all_coders, fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        for sess_result in results:
            temps = sess_result['temp']
            targets = sess_result['tar']
            decoded_targets = self._vocabulary.vectors_to_sentences(
                targets)

            mrt_loss = sess_result['loss']

            if self.scalar_summaries is None:
                scalar_summaries = None
                histogram_summaries = None
            else:
                # TODO collect summaries from different sessions
                scalar_summaries = results[0]['scalar_summaries']
                histogram_summaries = results[0]['histogram_summaries']
                
        self.result = ExecutionResult(
            outputs=[temps],
            losses=mrt_loss,
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)

