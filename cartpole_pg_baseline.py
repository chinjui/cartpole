import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import random

DISCOUNT = 0.99
LEARNING_RATE = 0.05
N_OBS = 4
N_ACTIONS = 2
N_BATCH_EPISODES = 5

""" Policy Network """
w_init = tf.truncated_normal_initializer(stddev=5e-2)
b_init = tf.constant_initializer(value=0.0)

x = tf.placeholder(tf.float32, shape=[None, N_OBS])
mc_discounted_rewards = tf.placeholder(tf.float32, shape=[None])

network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DenseLayer(network, n_units=128, act=tf.nn.tanh,
        W_init=w_init, b_init=b_init, name='fc0')
network = tl.layers.DenseLayer(network, n_units=N_ACTIONS, act=tf.identity,
        W_init=w_init, b_init=b_init)

y = network.outputs
prob = tf.nn.softmax(y)
log_prob = tf.log(prob)

# optimizer
action_taken = tf.placeholder(tf.int32, shape=[None])
action_one_hot = tf.one_hot(action_taken, N_ACTIONS)
good_log_prob = tf.reduce_sum(tf.multiply(action_one_hot, log_prob), reduction_indices=[1])
goal = tf.reduce_sum(good_log_prob * mc_discounted_rewards)
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(-goal)

""" End of network """

env = gym.make('CartPole-v0')
sess = tf.Session()
tl.layers.initialize_global_variables(sess)

# TODO some of lines are intended for 2-action games
# train for 2000 times

batch_trans = []
for i_episode in range(50000):
    obs = env.reset()
    states = []
    actions = []
    rewards = []
    for step in range(200):
        # env.render()
        prob_action = sess.run(prob, {x: [obs]})[0]
        prob_action_0 = prob_action[0]
        action = 0 if random.random() < prob_action_0 else 1
        new_obs, reward, done, info = env.step(action)

        # add to memory
        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = new_obs

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, step + 1))
            trans = {"states": states, "actions": actions, "rewards": rewards}
            batch_trans.append(trans)
            print(i_episode + 1 % N_BATCH_EPISODES)
            # update every N_BATCH_EPISODES
            if (i_episode + 1) % N_BATCH_EPISODES == 0:
                # concatenate all transitions in a batch
                b_lens = [len(t["states"]) for t in batch_trans]
                b_states = np.concatenate([t["states"] for t in batch_trans])
                b_actions = np.concatenate([t["actions"] for t in batch_trans])

                # calculate advantage with baseline
                b_returns = [t["rewards"] for t in batch_trans]
                for returns in b_returns:
                    for idx in reversed( range(len(returns) - 1) ):
                        returns[idx] += DISCOUNT * returns[idx + 1]
                max_len = max(b_lens)
                b_returns = np.array([np.lib.pad(returns, (0, max_len - len(returns)),
                                'constant', constant_values=(0, 0)) for returns in b_returns])
                baseline = np.mean(b_returns, axis=0)
                b_returns -= baseline
                b_returns = [b_returns[idx, :b_lens[idx]] for idx in range(N_BATCH_EPISODES)]
                b_returns = np.concatenate(b_returns)
                feed_dict = {x: b_states, action_taken: b_actions,
                            mc_discounted_rewards: b_returns}
                sess.run(train_op, feed_dict)

                batch_trans = []
            break
    #print('\n')

