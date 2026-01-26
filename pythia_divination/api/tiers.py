"""Tier configuration for user access control."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class UserTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierConfig:
    """Configuration for each user tier."""
    tier: UserTier
    daily_rate_limit: Optional[int]  # None = unlimited
    allowed_models: List[str]
    allowed_tasks: List[str]
    max_stocks_per_request: int
    max_historical_period_days: int
    premium_features: List[str]


TIER_CONFIGS = {
    UserTier.FREE: TierConfig(
        tier=UserTier.FREE,
        daily_rate_limit=10,
        allowed_models=["gradient_boosting", "linear_regression"],
        allowed_tasks=["classifier"],
        max_stocks_per_request=5,
        max_historical_period_days=30,
        premium_features=[],
    ),
    UserTier.PRO: TierConfig(
        tier=UserTier.PRO,
        daily_rate_limit=100,
        allowed_models=["gradient_boosting", "linear_regression", "random_forest", "lstm"],
        allowed_tasks=["classifier", "regressor"],
        max_stocks_per_request=20,
        max_historical_period_days=365,
        premium_features=["advanced_charts", "export_csv", "multi_model_comparison"],
    ),
    UserTier.ENTERPRISE: TierConfig(
        tier=UserTier.ENTERPRISE,
        daily_rate_limit=None,  # Unlimited
        allowed_models=["gradient_boosting", "linear_regression", "random_forest", "lstm"],
        allowed_tasks=["classifier", "regressor"],
        max_stocks_per_request=150,
        max_historical_period_days=3650,  # 10 years
        premium_features=[
            "advanced_charts",
            "export_csv",
            "multi_model_comparison",
            "custom_models",
            "api_access",
            "priority_support",
        ],
    ),
}


def get_tier_config(tier: str) -> TierConfig:
    """Get configuration for a tier."""
    tier_enum = UserTier(tier)
    return TIER_CONFIGS[tier_enum]
