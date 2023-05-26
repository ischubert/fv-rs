"""
Classes for reward shaping
"""
import numpy as np


class PlanBasedGaussian():
    """
    Plan-based reward shaping with Gaussians
    for discrete reward settings
    """

    def __init__(
            self,
            default_width=0.05
    ):
        self.default_width = default_width

    def shaped_reward_function(
            self,
            state,
            reward,
            plan=None,
            shaping_mode=None,
            previous_state=None,
            gamma=None,
            width=None,
            add_1_to_reward=True
    ):
        """
        Plan-based reward shaping using Gaussians
        """
        assert shaping_mode in [
            None,
            'potential_based',
            'asympt_equivalent'
        ]

        if width is None:
            width = self.default_width

        if add_1_to_reward:
            reward += 1

        if shaping_mode is None:
            return reward

        if reward:
            return reward

        assert plan is not None

        if shaping_mode == 'potential_based':
            assert previous_state is not None
            assert gamma is not None
            return reward + gamma * self.plan_based_reward(
                state,
                plan,
                width
            ) - self.plan_based_reward(
                previous_state,
                plan,
                width
            )

        if shaping_mode == 'asympt_equivalent':
            return reward + self.plan_based_reward(
                state,
                plan,
                width
            )

    def plan_based_reward(self, state, plan, width):
        """
        quantify value of state based on distance to plan and
        how far advanced in the plan the corresponding state is
        """

        exponential_dists = np.exp(
            -np.linalg.norm(
                state[None, :] - plan[:, :],
                axis=-1
            )**2/2/width**2
        )

        smallest_dist = np.argmax(exponential_dists)

        return 0.5 * exponential_dists[
            smallest_dist
        ] * smallest_dist/len(plan)
