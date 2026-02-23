from enum import Enum, auto
from typing import Dict, List, Optional

class BloomTier(Enum):
    KNOWLEDGE = 1
    COMPREHENSION = 2
    APPLICATION = 3
    ANALYSIS = 4
    SYNTHESIS = 5
    EVALUATION = 6

    @classmethod
    def get_ordered_tiers(cls) -> List['BloomTier']:
        return sorted(list(cls), key=lambda x: x.value)

class PedagogicalController:
    """
    Manages the training curriculum based on Bloom's Taxonomy and Mastery Learning principles.
    """
    def __init__(self, mastery_threshold: float = 0.8, alpha: float = 0.2):
        """
        Initialize the controller.
        
        Args:
            mastery_threshold: The score (0.0 to 1.0) required to consider a tier mastered.
            alpha: Smoothing factor for mastery score updates (Exponential Moving Average).
        """
        self.mastery_threshold = mastery_threshold
        self.alpha = alpha
        self.mastery_scores: Dict[BloomTier, float] = {
            tier: 0.0 for tier in BloomTier
        }
        self.current_focus_tier = BloomTier.KNOWLEDGE

    def update_mastery(self, tier: BloomTier, performance_score: float):
        """
        Update the mastery score for a specific tier using an exponential moving average.
        
        Args:
            tier: The BloomTier to update.
            performance_score: The observed performance (0.0 to 1.0).
        """
        if not (0.0 <= performance_score <= 1.0):
            raise ValueError("Performance score must be between 0.0 and 1.0")
            
        current_score = self.mastery_scores[tier]
        # EMA update: new_score = (1 - alpha) * current + alpha * observation
        # This handles noise in individual evaluation steps
        new_score = (1 - self.alpha) * current_score + self.alpha * performance_score
        self.mastery_scores[tier] = new_score
        
        self._update_current_focus()

    def _update_current_focus(self):
        """
        Determine the next tier to focus on based on mastery levels.
        We progress to the next tier only if the current one (and all previous) 
        meet the mastery threshold.
        """
        ordered_tiers = BloomTier.get_ordered_tiers()
        
        for tier in ordered_tiers:
            if self.mastery_scores[tier] < self.mastery_threshold:
                self.current_focus_tier = tier
                return
        
        # If all tiers are mastered, stay on the highest one (or could return None)
        self.current_focus_tier = ordered_tiers[-1]

    def get_current_tier(self) -> BloomTier:
        """Returns the tier the agent should currently be training on."""
        return self.current_focus_tier

    def get_mastery_report(self) -> Dict[str, float]:
        """Returns a human-readable report of mastery scores."""
        return {tier.name: score for tier, score in self.mastery_scores.items()}

    def reset(self):
        """Reset all mastery scores to zero."""
        for tier in BloomTier:
            self.mastery_scores[tier] = 0.0
        self.current_focus_tier = BloomTier.KNOWLEDGE
