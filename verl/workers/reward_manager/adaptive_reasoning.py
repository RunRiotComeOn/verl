"""
Adaptive Reasoning Reward Manager V6/V7 with Type1 Bonus Decay and Ratio Control

This reward manager supports:
- V6: Type1 bonus decay over training steps
- V7: Type1 bonus decay + Ratio control mechanism

Key Features:
- Tracks training steps via metadata or internal counter
- Passes step information to reward function
- Supports multiple decay strategies (linear, exponential, cosine)
- V7: Dynamic ratio-based penalty to prevent format over-exploitation
- Logs current bonus value and ratio control metrics
"""

from collections import defaultdict
from typing import Any
import sys
import os
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("adaptive_reasoning")
@register("adaptive_reasoning_v6")
class AdaptiveReasoningV6RewardManager(AbstractRewardManager):
    """
    Reward manager for adaptive reasoning V6/V7.

    This manager processes batches of responses and computes detailed metrics:
    - Format distribution (Type 1/2/3 ratios)
    - Per-type accuracy and average length
    - Reward component breakdown (base, format bonus, error penalties)
    - Decay tracking (current type1 bonus value and training step)
    - V7: Ratio control metrics and penalties
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        # Version selection
        reward_version: str = "v6",  # "v6" or "v7"
        # AdaptiveReasoningReward parameters
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = 0.0,
        type3_error_penalty: float = 0.0,
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,
        normalize_answers: bool = True,
        # Decay parameters
        enable_bonus_decay: bool = False,
        decay_strategy: str = "linear",
        decay_start_step: int = 0,
        decay_end_step: int = 30,
        type1_bonus_min: float = 0.0,
        decay_rate: float = 0.95,
        # V7 specific: Ratio control parameters
        enable_ratio_penalty: bool = False,
        ratio_penalty_start_step: int = 60,
        target_type1_ratio: float = 0.3,
        target_type2_ratio: float = 0.4,
        target_type3_ratio: float = 0.3,
        ratio_tolerance: float = 0.15,
        ratio_penalty_min_scalar: float = 0.5,
        ratio_window_size: int = 256,
    ) -> None:
        """Initialize the AdaptiveReasoningV6RewardManager with support for V6 and V7."""
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reward_version = reward_version
        self.training_step = 0  # Track training steps

        # Import and initialize the adaptive reasoning reward function
        # Add the reward_functions directory to path
        # Method 1: Relative to this file
        reward_functions_path = os.path.join(os.path.dirname(__file__), '../../../../reward_functions')
        # Method 2: From current working directory (fallback)
        if not os.path.exists(reward_functions_path):
            reward_functions_path = os.path.join(os.getcwd(), 'reward_functions')

        if os.path.exists(reward_functions_path):
            reward_functions_path = os.path.abspath(reward_functions_path)
            if reward_functions_path not in sys.path:
                sys.path.insert(0, reward_functions_path)

        # Common parameters for both V6 and V7
        common_params = {
            "correct_reward": correct_reward,
            "incorrect_reward": incorrect_reward,
            "type1_format_bonus": type1_format_bonus,
            "type2_format_bonus": type2_format_bonus,
            "type3_format_bonus": type3_format_bonus,
            "type1_error_penalty": type1_error_penalty,
            "type2_error_penalty": type2_error_penalty,
            "type3_error_penalty": type3_error_penalty,
            "length_threshold": length_threshold,
            "ideal_length": ideal_length,
            "min_scalar": min_scalar,
            "normalize_answers": normalize_answers,
            "enable_bonus_decay": enable_bonus_decay,
            "decay_strategy": decay_strategy,
            "decay_start_step": decay_start_step,
            "decay_end_step": decay_end_step,
            "type1_bonus_min": type1_bonus_min,
            "decay_rate": decay_rate,
        }

        try:
            if reward_version == "v7":
                # V7 with ratio control
                from adaptive_reasoning_reward_v7 import AdaptiveReasoningRewardV7

                self.reward_fn = AdaptiveReasoningRewardV7(
                    **common_params,
                    enable_ratio_penalty=enable_ratio_penalty,
                    ratio_penalty_start_step=ratio_penalty_start_step,
                    target_type1_ratio=target_type1_ratio,
                    target_type2_ratio=target_type2_ratio,
                    target_type3_ratio=target_type3_ratio,
                    ratio_tolerance=ratio_tolerance,
                    ratio_penalty_min_scalar=ratio_penalty_min_scalar,
                    ratio_window_size=ratio_window_size,
                )
                print(f"✓ Loaded AdaptiveReasoningRewardV7 with ratio control")
            elif reward_version == "v6":
                # V6 without ratio control
                from adaptive_reasoning_reward_v6 import AdaptiveReasoningRewardV6

                self.reward_fn = AdaptiveReasoningRewardV6(**common_params)
                print(f"✓ Loaded AdaptiveReasoningRewardV6")
            else:
                raise ValueError(f"Unknown reward version: {reward_version}. Must be 'v6' or 'v7'.")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import AdaptiveReasoningReward{reward_version.upper()} from reward_functions/. "
                f"Error: {e}"
            )

    def set_training_step(self, step: int):
        """
        Update the current training step.

        This should be called at the beginning of each training iteration.
        """
        self.training_step = step
        self.reward_fn.set_training_step(step)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of responses with metrics support.

        Args:
            data: DataProto containing batch of responses
            return_dict: If True, return dict with reward_tensor and metrics

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # Try to extract training step from metadata if available
        if hasattr(data, 'meta_info') and 'training_step' in data.meta_info:
            step = data.meta_info['training_step']
            self.set_training_step(step)

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

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            responses.append(response_str)
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

        # Compute rewards and metrics for the entire batch
        result = self.reward_fn(responses, ground_truths, return_dict=True)
        rewards = result['rewards']
        batch_metrics = result['metrics']

        # Fill reward tensor
        for i, (reward, valid_length) in enumerate(zip(rewards, valid_response_lengths)):
            reward_tensor[i, valid_length - 1] = reward
            reward_extra_info['score'].append(reward)

        if return_dict:
            # Add batch metrics as scalar values (not lists)
            # The modified ray_trainer.py will automatically extract these
            for metric_name, metric_value in batch_metrics.items():
                reward_extra_info[metric_name] = metric_value

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
