import warnings
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from Energy_Models.utils.rl_utils import ReplayBuffer
import tensorflow.keras as keras

class DDPG_Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01,
                 mem_size=500_000, fname="deuqling_dqn.h5", fc1_dims=32,
                 fc2_dims=256,fc3_dims=512,fc4_dims=256, fc5_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma


        # epsilon is controlled in rl_agent
        # self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size
        self.state_size = input_dims[0]

        self.learn_step_counter = 0
        self.replay_memory = ReplayBuffer(mem_size, input_dims)

        self.upper_bound = float(1)
        self.lower_bound = 0
        self.actor = self.get_actor(fc1_dims, fc2_dims)
        self.critic = self.get_critic(fc1_dims, fc2_dims)

        self.target_actor = self.get_actor(fc1_dims, fc2_dims)
        self.target_critic = self.get_critic(fc1_dims, fc2_dims)

        self.actor_loss = []
        self.critic_loss = []

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001
        # TODO find reasonable value for tau
        self.tau = 0.005

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Noise_object:
        std_dev = 0.5
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        self.scaled = False


    def get_actor(self,fc1_dims,fc2_dims):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_size,))
        out = layers.Dense(300, activation="relu")(inputs)
        out = layers.Dense(600, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self, fc1_dims,fc2_dims):
        warnings.warn("Number of neurons needs to be adapted")

        # State as input
        state_input = layers.Input(shape=(self.state_size))
        state_out = layers.Dense(600, activation="relu")(state_input)
        state_out = layers.Dense(300, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.n_actions))
        action_out = layers.Dense(1, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(600, activation="relu")(concat)
        out = layers.Dense(300, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state):
        """
        choose action based on given state
        :param state:
        :return: action
        """

        state = tf.convert_to_tensor(state)
        state = tf.reshape(state,shape=(1,len(state)))
        sampled_actions = tf.squeeze(self.actor(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return legal_action

    def choose_action(self,state, epsilon):
        # epsilon is not used in ddpg

        action = self.policy(state)

        return action

    # ggf @tf.function
    def train(self):

        """
        trains DDPG-Agent based on input-data collected in replay_memory
        actor and critic networks (with target-networks) are uppdated

        :return:
        """
        if self.replay_memory.mem_cntr < self.batch_size:
            return

        # sample inputs
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.sample_buffer(self.batch_size)

        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        action_batch = tf.reshape(action_batch,(len(action_batch),1))
        reward_batch = tf.convert_to_tensor(reward_batch)
        next_state_batch = tf.convert_to_tensor(next_state_batch)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch,training=True)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch,target_actions],
                                                               training = True)
            critic_value = self.critic([state_batch,action_batch],training =True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        self.critic_loss.append(critic_loss.numpy())
        critig_grad = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critig_grad,self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch,actions],training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        self.actor_loss.append(actor_loss.numpy())
        actor_grad = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))

        self.update_target(self.target_actor.variables,self.actor.variables)
        self.update_target(self.target_critic.variables,self.critic.variables)

    def train_2(self):
        if self.replay_memory.mem_cntr < self.batch_size:
            return

        # sample inputs
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.sample_buffer(self.batch_size)

        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        action_batch = tf.reshape(action_batch,(len(action_batch),1))
        reward_batch = tf.convert_to_tensor(reward_batch)
        next_state_batch = tf.convert_to_tensor(next_state_batch)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            critic_value_ = tf.squeeze(self.target_critic([next_state_batch, target_actions]))
            critic_value = tf.squeeze(self.critic([state_batch, action_batch],1))
            target = reward_batch + self.gamma *critic_value_ * (1-done_batch)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state_batch)
            actor_loss = -self.critic([state_batch, new_policy_actions])
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradient,self.actor.trainable_variables))

    # ggf @tf.function
    def update_target(self, target_weights,weights):
        for(a,b) in zip(target_weights,weights):
            a.assign(b * self.tau + a * (1-self.tau))

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.replay_memory.update_replay_memory(state, action, reward, new_state, done)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.x_prev = 0

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
