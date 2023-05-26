# %%
"""
Pick-and-place example, no reward shaping
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import ryenv
import ddpg

# %%
# Create environment object
# and interactive view
ENV = ryenv.PickAndPlaceEnv()
ENV.view()

# %%
# define start conditions and goal
ENDEFF_POS = [-0.8, -0.3]
DISK_POS = [0.5, -0.6]
GOAL_POS = [1.0, 0.7]

ENV.reset(
    ENDEFF_POS,
    disk_position=DISK_POS,
    goal_position=GOAL_POS
)

# create approximate plan (sequence of states)
ACTIONS = [
    [0, 0, 0.05],
    [0, -0.05, 0],
    [0.05, 0, 0],
    [0, 0, -0.05],
    [0, 0, 0.05],
    [0.05, 0, 0],
    [0, 0.05, 0],
    [0, 0, -0.05]
]
DURATIONS = [5, 6, 26, 4, 4, 10, 26, 4]
assert len(ACTIONS) == len(DURATIONS)

OBS_VEC = []
REWARDS = []
for action, duration in zip(ACTIONS, DURATIONS):
    for _ in range(duration):
        observation, reward, done, info = ENV.step(action)
        OBS_VEC.append(observation['observation'])
        REWARDS.append(reward)

OBS_VEC = np.array(OBS_VEC)

PLAN = OBS_VEC.copy()
GOAL = observation['desired_goal']

# %%
# DDPG RL policy architecture
# define critic network
STATE_IN = keras.Input(shape=7)
ACTION_IN = keras.Input(shape=3)
GOAL_IN = keras.Input(shape=3)
FX = keras.layers.Concatenate(axis=-1)([
    STATE_IN,
    ACTION_IN,
    GOAL_IN
])
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(50, activation='relu')(FX)
FX = keras.layers.Dense(20, activation='relu')(FX)
FX = keras.layers.Dense(10, activation='relu')(FX)
FX = keras.layers.Dense(1, activation='linear')(FX)
CRITIC = keras.Model(
    inputs=[STATE_IN, ACTION_IN, GOAL_IN],
    outputs=FX
)
CRITIC.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.003),
    loss='mse'
)

# define actor network
STATE_IN = keras.Input(shape=7)
GOAL_IN = keras.Input(shape=3)
FX = keras.layers.Concatenate(axis=-1)([
    STATE_IN,
    GOAL_IN
])
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(100, activation='relu')(FX)
FX = keras.layers.Dense(50, activation='relu')(FX)
FX = keras.layers.Dense(30, activation='relu')(FX)
FX = keras.layers.Dense(20, activation='relu')(FX)
FX = keras.layers.Dense(10, activation='relu')(FX)
FX = keras.layers.Dense(3, activation='linear')(FX)
FX = tf.keras.layers.LayerNormalization(
    axis=1,
    scale=False
)(FX)
ACTOR = keras.Model(
    inputs=[STATE_IN, GOAL_IN],
    outputs=FX
)
ACTOR.compile(
    tf.keras.optimizers.Adadelta(lr=0.01),
    loss='mse'
)

# create DDPG policy
GAMMA = 0.9
POL = ddpg.DDPGBase(
    ACTOR,
    CRITIC,
    gamma=GAMMA,
    critic_epochs=1,
    actor_epochs=1,
    tau=0.1,
    clipping=None
)

# %%
# Train the agent
ENV.reset(
    ENDEFF_POS,
    disk_position=DISK_POS,
    goal_position=GOAL_POS
)

# Parameters:
# number of training rollouts
# maximum length (time steps) of single rollout
# epsilon for epsilon-greedy exploration strategy
ROLLOUTS = 2000
ROLLOUT_MAX_LEN = 300
EPSILON = .8

STATES = np.full(
    (ROLLOUTS*ROLLOUT_MAX_LEN, 7),
    np.nan
)
ACTIONS = np.full(
    (ROLLOUTS*ROLLOUT_MAX_LEN, 3),
    np.nan
)
REWARDS = np.full(
    ROLLOUTS*ROLLOUT_MAX_LEN,
    np.nan
)
NEXT_STATES = np.full(
    (ROLLOUTS*ROLLOUT_MAX_LEN, 7),
    np.nan
)
GOALS = np.full(
    (ROLLOUTS*ROLLOUT_MAX_LEN, 3),
    np.nan
)

ACTION = ENV.sample_action()

COUNTER = -1

for rollout in range(ROLLOUTS):
    # environment is reset to the initial configuration
    # at the start of each rollout
    ENV.reset(
        ENDEFF_POS,
        disk_position=DISK_POS,
        goal_position=GOAL_POS
    )

    observation, reward, done, info = ENV.step([0, 0, 0])
    state = observation['observation']

    for step in range(int(np.ceil(ROLLOUT_MAX_LEN * np.random.rand()))):
        COUNTER += 1
        GOALS[
            COUNTER,
            :
        ] = GOAL

        STATES[
            COUNTER,
            :
        ] = state.copy()

        # eps-greedy action selection
        if np.random.rand() > EPSILON:
            ACTION = POL.get_action(
                state,
                GOAL
            )
        else:
            ACTION = ENV.sample_action()

        ACTIONS[
            COUNTER,
            :
        ] = ACTION

        observation, reward, done, info = ENV.step(
            np.append(
                ACTION,
                0
            )
        )
        state = observation['observation']
        REWARDS[
            COUNTER
        ] = reward

        NEXT_STATES[
            COUNTER,
            :
        ] = state.copy()

    # policy is trained using the data buffer of size 30000
    buffer_start = max(
        0,
        COUNTER-30000
    )
    for ___ in range(10):
        POL.train(
            STATES[buffer_start:COUNTER],
            ACTIONS[buffer_start:COUNTER],
            GOALS[buffer_start:COUNTER],
            REWARDS[buffer_start:COUNTER],
            NEXT_STATES[buffer_start:COUNTER],
            verbose=0
        )
