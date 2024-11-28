import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from norm import math_normalizer as math_norm
from norm import numbered

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.utils import ModelOutput
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import generate_model_card, peft_module_casting_to_bf16

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


@contextmanager
def unwrap_model_for_generation(
        model: Union["DistributedDataParallel", "DeepSpeedEngine"], accelerator: "Accelerator",
        is_peft_model: bool = False
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    yield unwrapped_model


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path + '/policy', **kwargs)
        self.value_model.save_pretrained(path + '/value', **kwargs)



class PPOBatch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PPOStats():
    def __init__(self, stats_shape, device='cuda'):
        self.approxkl_stats = torch.zeros(stats_shape, device=device)
        self.pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.pg_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.ratio_stats = torch.zeros(stats_shape, device=device)


class PPOTrainerWrapper():
    def __init__(
            self,
            config: PPOConfig,
            processing_class,
            accelerator,
    ):
        self.args = config
        self.processing_class = processing_class
        self.accelerator = accelerator
        self.stats_shape = (self.args.num_ppo_epochs, self.args.num_mini_batches, self.args.gradient_accumulation_steps)
        self.stats = PPOStats(self.stats_shape, )

    def rollout(self, data, model, ref_policy, reward_model=None, tokenizer=None, clean=True):
        device = 'cuda'
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(self.args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        with torch.no_grad():
            queries = data["input_ids"].to(device)
            context_length = queries.shape[1]
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []
            values = []
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                if 'rollout' in data:
                    query_responses = data["rollout"]
                    # logitss = unwrapped_model.policy(query_responses)
                    out = forward(unwrapped_model.policy, query_responses, self.processing_class.pad_token_id)
                    logitss = out.logits[:, context_length - 1: -1]
                else:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        queries.size(0),
                        self.processing_class.pad_token_id,
                        generation_config,
                    )
            # print(queries.shape[0], self.args.local_rollout_forward_batch_size)
            # print('Done Generate')
            for i in range(0, queries.shape[0], self.args.local_rollout_forward_batch_size):
                # print('Cache')
                query = queries[i: i + self.args.local_rollout_forward_batch_size]
                query_response = query_responses[i: i + self.args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                logits = logitss[i: i + self.args.local_rollout_forward_batch_size]
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()

                if ref_policy is None:
                    with self.null_ref_context():
                        ref_output = forward(model.policy, query_response, self.processing_class.pad_token_id)
                else:
                    ref_output = forward(ref_policy, query_response, self.processing_class.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1: -1]
                ref_logits /= self.args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if self.args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.args.stop_token_id, self.processing_class.pad_token_id, response
                    )

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == self.processing_class.pad_token_id) - 1
                unwrapped_value_model = self.accelerator.unwrap_model(model).value_model
                full_value, _, _ = get_reward(
                    unwrapped_value_model, query_response, self.processing_class.pad_token_id, context_length
                )
                value = full_value[:, context_length - 1: -1].squeeze(-1)
                if reward_model is not None:
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, self.processing_class.pad_token_id, context_length
                    )
                else:
                    text = tokenizer.batch_decode(postprocessed_response)
                    answers = data['answers'][i: i + self.args.local_rollout_forward_batch_size]
                    score = []
                    for txt, ans in zip(text, answers):
                        txt = txt.replace('<|eot_id|>', '').replace('<|finetune_right_pad_id|>', '')
                        score.append(float(math_norm(txt) == str(ans)))
                    score = torch.tensor(score).to(postprocessed_response.device)

                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                values.append(value)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            # print(scores)
            values = torch.cat(values, 0)
            if clean:
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
            torch.cuda.empty_cache()
            gc.collect()

            # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
            # Completions not passing that filter will receive a lower score.
            contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
            if self.args.missing_eos_penalty is not None:
                scores[~contain_eos_token] -= self.args.missing_eos_penalty
            # print(scores)
            # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            sequence_lengths_p1 = sequence_lengths + 1
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            values = torch.masked_fill(values, padding_mask_p1, 0)

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            non_score_reward = -self.args.kl_coef * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            rewards[[actual_start, actual_end]] += scores

            # 5. whiten rewards
            if self.args.whiten_rewards:
                rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)
            torch.cuda.empty_cache()
        return PPOBatch(
            advantages=advantages,
            responses=responses,
            query_responses=query_responses,
            logprobs=logprobs,
            returns=returns,
            values=values,
            context_length=context_length,
            padding_mask=padding_mask,
            padding_mask_p1=padding_mask_p1,
            non_score_reward=non_score_reward,
            kl=kl,
            scores=scores,
        )

    def train_step(self, batch: PPOBatch, model, optimizer, lr_scheduler=None, clean=True):
        advantages = batch.advantages
        responses = batch.responses
        query_responses = batch.query_responses
        logprobs = batch.logprobs
        returns = batch.returns
        values = batch.values
        context_length = batch.context_length
        padding_mask = batch.padding_mask
        padding_mask_p1 = batch.padding_mask_p1

        for ppo_epoch_idx in range(self.args.num_ppo_epochs):
            b_inds = np.random.permutation(self.args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, self.args.local_batch_size, self.args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + self.args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, self.args.local_mini_batch_size,
                                               self.args.per_device_train_batch_size):
                    with self.accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + self.args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        output, vpred_temp = forward(model, mb_query_responses, self.processing_class.pad_token_id)
                        logits = output.logits[:, context_length - 1: -1]
                        logits /= self.args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(
                            new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                        )
                        vpred = vpred_temp[:, context_length - 1: -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - self.args.cliprange_value,
                            mb_values + self.args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                        vf_clipfrac = masked_mean(
                            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                        )
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - self.args.cliprange,
                                                                 1.0 + self.args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        loss = pg_loss + self.args.vf_coef * vf_loss
                        self.accelerator.backward(loss)
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff ** 2).mean()
                            self.stats.approxkl_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            self.stats.pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                pg_clipfrac
                            )
                            self.stats.pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            self.stats.vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            self.stats.vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                vf_clipfrac
                            )
                            self.stats.entropy_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            self.stats.ratio_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if clean:
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

        if lr_scheduler is not None:
            lr_scheduler.step()

    def log(self, batch: PPOBatch, clean=True):
        kl = batch.kl
        logprobs = batch.logprobs
        non_score_reward = batch.non_score_reward
        scores = batch.scores
        responses = batch.responses

        with torch.no_grad():
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            rlhf_reward = mean_non_score_reward + scores.mean()
            metrics = {}
            metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
            metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
            metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
            metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
            metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
            metrics["policy/approxkl_avg"] = self.accelerator.gather(self.stats.approxkl_stats).mean().item()
            metrics["policy/clipfrac_avg"] = self.accelerator.gather(self.stats.pg_clipfrac_stats).mean().item()
            metrics["loss/policy_avg"] = self.accelerator.gather(self.stats.pg_loss_stats).mean().item()
            metrics["loss/value_avg"] = self.accelerator.gather(self.stats.vf_loss_stats).mean().item()
            metrics["val/clipfrac_avg"] = self.accelerator.gather(self.stats.vf_clipfrac_stats).mean().item()
            metrics["policy/entropy_avg"] = self.accelerator.gather(self.stats.entropy_stats).mean().item()
            metrics["val/ratio"] = self.accelerator.gather(self.stats.ratio_stats).mean().item()
            metrics["val/ratio_var"] = self.accelerator.gather(self.stats.ratio_stats).var().item()
            metrics["val/num_eos_tokens"] = (responses == self.processing_class.eos_token_id).sum().item()

        if clean:
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, non_score_reward
            del batch
            torch.cuda.empty_cache()
            gc.collect()
        return metrics




