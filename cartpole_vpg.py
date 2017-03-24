import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import random

DISCOUNT = 0.99
LEARNING_RATE = 0.05
N_OBS = 4
N_ACTIONS = 2

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
transitions = []
for i_episode in range(50000):
    obs = env.reset()
    for step in range(200):
        # env.render()
        prob_action = sess.run(prob, {x: [obs]})[0]
        prob_action_0 = prob_action[0]
        #print(prob_action)
        action = 0 if random.random() < prob_action_0 else 1
        new_obs, reward, done, info = env.step(action)
        transitions.append((obs, action, reward))
        obs = new_obs

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, step + 1))

            # train PG network
            n_steps = len(transitions)
            discounted_rewards = np.zeros(n_steps)
            # no need to calculate the return for the last state
            # since it's always 0, the gradients to adjust the NN are also 0s
            rewards = np.array([t[2] for t in transitions])
            discounted_rewards += rewards
            for i in reversed(range(n_steps - 1)):
                discounted_rewards[i] += DISCOUNT * discounted_rewards[i + 1]

            # normalize discounted rewards
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            states = [t[0] for t in transitions]
            actions = [t[1] for t in transitions]
            feed_dict = {x: states, action_taken: actions,
                        mc_discounted_rewards: discounted_rewards}
            sess.run(train_op, feed_dict)

            break
    #print('\n')

