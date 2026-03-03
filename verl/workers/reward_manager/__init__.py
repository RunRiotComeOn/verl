# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager

# Import custom reward managers
try:
    from .adaptive_reasoning import AdaptiveReasoningV6RewardManager
    AdaptiveReasoningRewardManager = AdaptiveReasoningV6RewardManager  # Alias for latest version
except ImportError:
    AdaptiveReasoningV6RewardManager = None
    AdaptiveReasoningRewardManager = None

try:
    from .adaptive_reasoning_v5 import AdaptiveReasoningV5RewardManager
except ImportError:
    AdaptiveReasoningV5RewardManager = None

try:
    from .adaptive_reward_v1 import AdaptiveRewardV1Manager
except ImportError:
    AdaptiveRewardV1Manager = None

try:
    from .adaptive_reward_v2 import AdaptiveRewardV2Manager
except ImportError:
    AdaptiveRewardV2Manager = None

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "AdaptiveReasoningRewardManager",
    "AdaptiveReasoningV5RewardManager",
    "AdaptiveReasoningV6RewardManager",
    "AdaptiveRewardV1Manager",
    "AdaptiveRewardV2Manager",
    "register",
    "get_reward_manager_cls",
]
