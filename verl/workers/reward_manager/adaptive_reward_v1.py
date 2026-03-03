"""
Adaptive Reward V1 Manager for verl GRPO training.

Key difference from other reward managers:
- Decodes with skip_special_tokens=False so that <perception>, <reasoning>,
  <answer> tags are preserved in decoded text for format detection.
"""

from collections import defaultdict
from typing import Any
import sys
import os
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("adaptive_reward_v1")
class AdaptiveRewardV1Manager(AbstractRewardManager):
    """
    Reward manager for AdaptiveRewardV1.

    Decodes responses with skip_special_tokens=False to preserve
    <perception>, <reasoning>, <answer> tags for format detection.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        # AdaptiveRewardV1 parameters (passed through)
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.2,
        type3_error_penalty: float = 0.0,
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,
        normalize_answers: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Add reward_functions directory to path
        reward_functions_path = os.path.join(os.path.dirname(__file__), '../../../../reward_functions')
        if not os.path.exists(reward_functions_path):
            reward_functions_path = os.path.join(os.getcwd(), 'reward_functions')

        if os.path.exists(reward_functions_path):
            reward_functions_path = os.path.abspath(reward_functions_path)
            if reward_functions_path not in sys.path:
                sys.path.insert(0, reward_functions_path)

        try:
            from adaptive_reward_v1 import AdaptiveRewardV1
            self.reward_fn = AdaptiveRewardV1(
                correct_reward=correct_reward,
                incorrect_reward=incorrect_reward,
                type1_format_bonus=type1_format_bonus,
                type2_format_bonus=type2_format_bonus,
                type3_format_bonus=type3_format_bonus,
                type1_error_penalty=type1_error_penalty,
                type2_error_penalty=type2_error_penalty,
                type3_error_penalty=type3_error_penalty,
                length_threshold=length_threshold,
                ideal_length=ideal_length,
                min_scalar=min_scalar,
                normalize_answers=normalize_answers,
            )
            print("Loaded AdaptiveRewardV1 (skip_special_tokens=False)")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import AdaptiveRewardV1 from reward_functions/adaptive_reward_v1.py. "
                f"Error: {e}"
            )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of responses.

        CRITICAL: Decodes with skip_special_tokens=False to preserve
        <perception>, <reasoning>, <answer> tags.
        """
        # If pre-computed RM scores exist, return directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        responses = []
        ground_truths = []
        valid_response_lengths = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # CRITICAL: skip_special_tokens=False to preserve format tags
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            responses.append(response_str)
            ground_truths.append(ground_truth)
            valid_response_lengths.append(valid_response_length)

            # Debug printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str[:200])
                print("[response]", response_str[:300])
                print("[ground_truth]", ground_truth)

        # Compute rewards and metrics for the entire batch
        result = self.reward_fn(responses, ground_truths, return_dict=True)
        rewards = result['rewards']
        batch_metrics = result['metrics']

        # Fill reward tensor
        for i, (reward, valid_length) in enumerate(zip(rewards, valid_response_lengths)):
            reward_tensor[i, valid_length - 1] = reward
            reward_extra_info['score'].append(reward)

        if return_dict:
            for metric_name, metric_value in batch_metrics.items():
                reward_extra_info[metric_name] = metric_value

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
