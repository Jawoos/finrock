import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import layers, models

from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv
from finrock.scalers import MinMaxScaler, ZScoreScaler
from finrock.reward import SimpleReward, AccountValueChangeReward
from finrock.metrics import DifferentActions, AccountValue, MaxDrawdown, SharpeRatio
from finrock.indicators import BolingerBands, RSI, PSAR, SMA, MACD

from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import MemoryManager
from rockrl.tensorflow import PPOAgent
from rockrl.utils.vectorizedEnv import VectorizedEnv

df = pd.read_csv('Datasets/upbit_btc_data.csv')
df = df[:-5000]

pd_data_feeder = PdDataFeeder(
    df,
    indicators = [
        BolingerBands(data=df, period=20, std=2),
        RSI(data=df, period=14),
        PSAR(data=df),
        MACD(data=df),
        SMA(data=df, period=7),
    ]
)

num_envs = 10
env = VectorizedEnv(
    env_object = TradingEnv,
    num_envs = num_envs,
    data_feeder = pd_data_feeder,
    output_transformer = ZScoreScaler(),
    initial_balance = 1000.0,
    max_episode_steps = 1000,
    window_size = 50,
    reward_function = AccountValueChangeReward(),
    metrics = [
        DifferentActions(),
        AccountValue(),
        MaxDrawdown(),
        SharpeRatio(),
    ]
)

action_space = env.action_space
input_shape = env.observation_space.shape

def transformer_block(inputs, num_heads, key_dim, ff_dim, rate=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = layers.Dropout(rate)(attn_output)
    out1 = layers.Add()([inputs, attn_output])
    out1 = layers.LayerNormalization(epsilon=1e-6)(out1)

    ffn_output = layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)
    out2 = layers.Add()([out1, ffn_output])
    return layers.LayerNormalization(epsilon=1e-6)(out2)

def actor_model(input_shape, action_space, num_heads=2, key_dim=32, ff_dim=128, rate=0.1):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(inputs)
    x = layers.Reshape((input_shape[0], -1))(x)
    x = transformer_block(x, num_heads, key_dim, ff_dim, rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(action_space, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def critic_model(input_shape, num_heads=2, key_dim=32, ff_dim=128, rate=0.1):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(inputs)
    x = layers.Reshape((input_shape[0], -1))(x)
    x = transformer_block(x, num_heads, key_dim, ff_dim, rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs=inputs, outputs=outputs)

agent = PPOAgent(
    actor = actor_model(input_shape, action_space),
    critic = critic_model(input_shape),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    batch_size=128,
    lamda=0.95,
    kl_coeff=0.5,
    c2=0.01,
    writer_comment='ppo_sinusoid_discrete',
)

pd_data_feeder.save_config(agent.logdir)
env.env.save_config(agent.logdir)

memory = MemoryManager(num_envs=num_envs)
meanAverage = MeanAverage(best_mean_score_episode=1000)
states, infos = env.reset()
rewards = 0.0

with tqdm(total=10000) as pbar:
    while agent.epoch < 10000:
        action, prob = agent.act(states)

        next_states, reward, terminated, truncated, infos = env.step(action)
        memory.append(states, action, reward, prob, terminated, truncated, next_states, infos)
        states = next_states

        for index in memory.done_indices():
            pre_epoch = agent.epoch

            env_memory = memory[index]
            history = agent.train(env_memory)
            mean_reward = meanAverage(np.sum(env_memory.rewards))

            if meanAverage.is_best(agent.epoch):
                agent.save_models('ppo_sinusoid')

            if history['kl_div'] > 0.05 and agent.epoch > 1000:
                agent.reduce_learning_rate(0.995, verbose=False)

            info = env_memory.infos[-1]
            print(agent.epoch, np.sum(env_memory.rewards), mean_reward, info["metrics"]['account_value'], history['kl_div'])
            agent.log_to_writer(info['metrics'])
            states[index], infos[index] = env.reset(index=index)

            if pre_epoch != agent.epoch:
                pre_epoch = agent.epoch
                pbar.update()

        if agent.epoch >= 10000:
            break

env.close()
exit()
