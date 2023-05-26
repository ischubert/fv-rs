"""
RL with DDPG
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class DDPGBase():
    """
    Base class for RL with DDPG
    """

    def __init__(
            self,
            actor,
            critic,
            gamma=0.9,
            tau=0.1,
            critic_epochs=10,
            actor_epochs=10,
            critic_batch_size=32,
            actor_batch_size=32,
            clipping=None
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.critic_epochs = critic_epochs
        self.actor_epochs = actor_epochs
        self.critic_batch_size = critic_batch_size
        self.actor_batch_size = actor_batch_size
        self.clipping = clipping

        if self.clipping is not None:
            assert len(clipping) == 2

        # create target networks by cloning
        # the original networks
        self.actor_target = keras.models.clone_model(
            self.actor
        )
        self.critic_target = keras.models.clone_model(
            self.critic
        )

    def get_action(
            self,
            state,
            goal
    ):
        """
        query optimal action from actor
        """
        return np.array(self.actor([
            state.reshape((1, -1)),
            goal.reshape((1, -1))
        ])).reshape(-1)

    def train(
            self,
            states,
            actions,
            goals,
            rewards,
            next_states,
            verbose=0
    ):
        """
        Train both actor and critic based on data from the buffer
        """
        # critic target from target networks
        critic_targets = rewards.reshape(-1, 1) + self.gamma*np.array(self.critic_target([
            next_states,
            # next actions:
            self.actor_target([
                next_states,
                goals
            ]),
            goals
        ]))
        if self.clipping is not None:
            critic_targets = np.clip(
                critic_targets,
                self.clipping[0],
                self.clipping[1]
            )
        if verbose:
            print(
                'critic_targets: min={}, mean={}, max={}'.format(
                    np.min(critic_targets), np.mean(
                        critic_targets), np.max(critic_targets)
                )
            )

        # update critic
        self.critic.fit(
            # X
            [
                states,
                actions,
                goals
            ],
            # Y
            critic_targets,
            epochs=self.critic_epochs,
            batch_size=self.critic_batch_size,
            verbose=verbose
        )

        # update actor
        for epoch_count in range(self.actor_epochs):

            indices = np.random.permutation(len(states))
            batches = indices[
                :self.actor_batch_size*(len(states)//self.actor_batch_size)
            ].reshape(-1, self.actor_batch_size)

            mean_loss = 0

            for batch in batches:

                with tf.GradientTape() as tape:
                    actor_actions = self.actor([
                        states[batch],
                        goals[batch]
                    ])
                    critic_value = self.critic([
                        states[batch],
                        actor_actions,
                        goals[batch]
                    ])
                    actor_loss = -tf.math.reduce_mean(critic_value)
                    actor_grad = tape.gradient(
                        actor_loss, self.actor.trainable_variables)

                self.actor.optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables)
                )

                mean_loss += float(actor_loss) * (-1/len(batches))

            if (epoch_count % 10 == 1) or (verbose > 0):
                print('Epoch {}/{}: Actor mean return {}'.format(
                    epoch_count, self.actor_epochs, mean_loss
                ))

        # update targets
        self.actor_target.set_weights(
            [
                self.tau * weights
                + (1 - self.tau) * target_weights
                for weights, target_weights in zip(
                    self.actor.get_weights(),
                    self.actor_target.get_weights()
                )
            ]
        )
        self.critic_target.set_weights(
            [
                self.tau * weights
                + (1 - self.tau) * target_weights
                for weights, target_weights in zip(
                    self.critic.get_weights(),
                    self.critic_target.get_weights()
                )
            ]
        )
