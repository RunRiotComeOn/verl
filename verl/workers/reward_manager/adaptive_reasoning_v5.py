"""
Adaptive Reasoning Reward Manager V5 with Batch Metrics Support

This reward manager computes rewards for adaptive reasoning tasks with:
- Reversed format bonus (Type 1 > Type 2 > Type 3)
- Error penalties for Type 1 and Type 2
- NO diversity scaling (simplified)
- Comprehensive batch-level metrics for TensorBoard monitoring
"""

from collections import defaultdict
from typing import Any
import sys
import os
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("adaptive_reasoning_v5")
class AdaptiveReasoningV5RewardManager(AbstractRewardManager):
    """
    Reward manager for adaptive reasoning V5 with simplified rewards.

    This manager processes batches of responses and computes detailed metrics:
    - Format distribution (Type 1/2/3 ratios)
    - Per-type accuracy and average length
    - Reward component breakdown (base, format bonus, error penalties)
    - NO diversity scaling (removed for simplification)
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        # AdaptiveReasoningRewardV5 parameters
        type1_format_bonus: float = 0.2,
        type2_format_bonus: float = 0.1,
        type3_format_bonus: float = 0.0,
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.3,
        type3_error_penalty: float = 0.0,
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,
        normalize_answers: bool = True,
    ) -> None:
        """Initialize the AdaptiveReasoningV5RewardManager."""
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Import and initialize the adaptive reasoning V4 reward function (uses v4 reward function)
        # Add the reward_functions directory to path
        reward_functions_path = os.path.join(os.path.dirname(__file__), '../../../../reward_functions')
        if os.path.exists(reward_functions_path):
            sys.path.insert(0, os.path.abspath(reward_functions_path))

        try:
            from adaptive_reasoning_reward import AdaptiveReasoningRewardV5

            self.reward_fn = AdaptiveReasoningRewardV5(
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
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import AdaptiveReasoningRewardV5 from reward_functions/adaptive_reasoning_reward.py. "
                f"Error: {e}"
            )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of responses with metrics support.

        Args:
            data: DataProto containing batch of responses
            return_dict: If True, return dict with reward_tensor and metrics

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # If there is rm score, directly return it
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Collect all responses and ground truths for batch processing
        decoded_responses = []
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

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            decoded_responses.append(response_str)
            ground_truths.append(ground_truth)
            valid_response_lengths.append(valid_response_length)

            # Print for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

        # Compute rewards with metrics
        result = self.reward_fn(decoded_responses, ground_truths, return_dict=True)
        rewards = result["rewards"]
        batch_metrics = result["metrics"]

        # Fill reward tensor
        for i, (reward, valid_length) in enumerate(zip(rewards, valid_response_lengths)):
            reward_tensor[i, valid_length - 1] = reward
            reward_extra_info["score"].append(reward)

        if return_dict:
            # Add batch metrics as scalar values (not lists)
            for metric_name, metric_value in batch_metrics.items():
                reward_extra_info[metric_name] = metric_value

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def aggregate_metrics(self, batch_metrics_list: list[dict]) -> dict[str, float]:
        """
        Aggregate metrics from multiple batches.

        Args:
            batch_metrics_list: List of metric dicts from multiple batches

        Returns:
            Aggregated metrics
        """
        if not batch_metrics_list:
            return {}

        # Aggregate metrics
        aggregated = defaultdict(list)

        for batch_metrics in batch_metrics_list:
            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    aggregated[key].append(value)

        # Compute mean for each metric
        final_metrics = {}
        for key, values in aggregated.items():
            if values:
                final_metrics[key] = sum(values) / len(values)

        return final_metrics
