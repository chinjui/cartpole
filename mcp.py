import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gym
from gym import wrappers

# initialize constants
batch_size = 5 # number of episodes after which parameter update is done
train_episodes = 50000 # total number of training episodes
hidden_units = 50 # number of hidden units in neural network
discount_factor = 0.99 # reward discounting factor
epsilon = 1e-9 # epsilon of RMSProp update
learning_rate = 0.05 # stepsize in RMSProp update
decay_rate = 0.9 # decay rate of RMSProp

# build environment
env = gym.make("CartPole-v0")
n_obv = env.observation_space.shape[0] # size of observation vector from environment
n_acts = env.action_space.n # number of actions possible in the given environment  
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(7)

# create model
w_init = tf.truncated_normal_initializer(stddev=5e-2)
b_init = tf.constant_initializer(value=0.0)

obv = tf.placeholder(tf.float32, shape=[None, n_obv])
adv = tf.placeholder(tf.float32, shape=[None])

network = tl.layers.InputLayer(obv, name='input_layer')
network = tl.layers.DenseLayer(network, n_units=128, act=tf.nn.tanh,
        W_init=w_init, b_init=b_init, name='fc0')
network = tl.layers.DenseLayer(network, n_units=n_acts, act=tf.identity,
        W_init=w_init, b_init=b_init)

y = network.outputs
y = tf.nn.softmax(y)
log_prob = tf.log(y)

# optimizer
acts = tf.placeholder(tf.int32, shape=[None])
action_one_hot = tf.one_hot(acts, n_acts)
good_log_prob = tf.reduce_sum(tf.multiply(action_one_hot, log_prob), reduction_indices=[1])
loss = -tf.reduce_sum(good_log_prob * adv)
params = network.all_params
#train_step = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay = decay_rate, epsilon = epsilon).minimize(loss, var_list=params)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

"""
obv = tf.placeholder(tf.float64)
acts = tf.placeholder(tf.int32)
adv = tf.placeholder(tf.float64)
W0 = tf.Variable(tf.random_uniform([n_obv, hidden_units], dtype=tf.float64))
b0 = tf.Variable(tf.zeros([hidden_units], dtype=tf.float64))
W1 = tf.Variable(tf.random_uniform([hidden_units, n_acts], dtype=tf.float64))
b1 = tf.Variable(tf.zeros(n_acts, dtype=tf.float64))
params = [W0, b0, W1, b1]
N = tf.shape(obv)[0]
y = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(obv, W0) + b0[None, :]), W1) + b1[None, :])
idx_flattened = tf.range(0, tf.shape(y)[0]) * tf.shape(y)[1] + acts
yy = tf.gather(tf.reshape(y, [-1]), idx_flattened)
loss = -tf.reduce_sum(tf.multiply(tf.log(yy), adv)) / tf.cast(N, tf.float64)
train_step = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay = decay_rate, epsilon = epsilon).minimize(loss, var_list=params)
"""
def act(obvs, sess):
    obvs = obvs.reshape(1, -1)
    probs = sess.run(y, feed_dict={obv: obvs})
    probs = np.asarray(probs)
    return probs.argmax()

# computes the return R = r[i] + discount_factor * r[i + 1] + discount_factor^2 * r[i + 2]
def get_discounted_return(r, discount_factor):
    y = np.zeros(len(r), 'float64')
    y[-1] = r[-1]
    for i in reversed(range(len(r)-1)):
        y[i] = r[i] + discount_factor * y[i + 1]
    return y

observation = env.reset()
trajectories = [] # list of dictionary of rewards, actions and observations for each episode
observations = [] # list of observations for each episode
rewards = [] # list of discounted rewards for each episode
actions = [] # list of actions for each episode
episode = 0 # episode iterator
episode_time = 0 # measure time of each episode
net_reward = 0 # net reward of each episode
iteration = 0 # number of times parameters are updated

# start tf session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

while True:

    # take action and record everything
    action = act(observation, sess)
    (observation, reward, done, _) = env.step(action)
    observations.append(observation)
    actions.append(action)
    rewards.append(reward)
    net_reward += reward
    episode_time += 1

    if done:
        # stores the trajectory of episode
        episode += 1
        trajectory = {"rew" : np.array(rewards), "obv" : np.array(observations), "acts" : np.array(actions)}
        trajectories.append(trajectory)

        if episode % batch_size == 0:
            iteration += 1
            # get list of all (observations, discounted rewards, actions) of the batch
            batch_obv = np.concatenate([trajectory["obv"] for trajectory in trajectories])
            batch_rets = [get_discounted_return(trajectory["rew"], discount_factor) for trajectory in trajectories]
            batch_acts = np.concatenate([trajectory["acts"] for trajectory in trajectories])
            max_episode_length = max(len(reward) for reward in batch_rets)
            batch_rew = [np.concatenate([reward, np.zeros(max_episode_length - len(reward))]) for reward in batch_rets]

            # compute advantages with time dependent baselines
            baselines = np.mean(batch_rew, axis = 0)
            batch_adv = np.concatenate([reward - baselines[:len(reward)] for reward in batch_rets])

            # compute loss and do policy gradient step
            sess.run(train_step, feed_dict={obv: batch_obv, acts: batch_acts, adv: batch_adv})
            trajectories = []

        print("Episode %d finished after %d timesteps, reward = %d" % (episode, episode_time, net_reward))
        observation = env.reset()
        observations = []
        actions = []
        rewards = []
        episode_time = 0
        net_reward = 0

        if episode > train_episodes:
             break
