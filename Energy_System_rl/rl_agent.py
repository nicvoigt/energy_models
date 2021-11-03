from Energy_Models.Energy_System_rl.Agents.DDPG_1 import DDPG_Agent


class RL_Agent:
    def __init__(self, SoC_start, action_range):
        self.battery_charge = SoC_start
        lr = 0.001
        gamma = 0.95
        n_actions = 1
        epsilon = 1
        batch_size = 64
        input_dims = [3]
        self.agent = DDPG_Agent(lr, gamma, n_actions, epsilon, batch_size,
                 input_dims)
        self.action_range = action_range

    def choose_action(self, state, epsilon):
        """

        :param state: current rl state of model
        :param epsilon: unused in DDPG
        :return:
        """
        action = self.agent.choose_action(state, epsilon)
        action = self.translate_action(action)

        return action


    def translate_action(self, action):

        # from 0 to 1 changes to from 0 to 10
        action = action[0]
        # print(action)
        untere_aktion = self.action_range[0]
        obere_action = self.action_range[1]
        action *=abs(untere_aktion) + obere_action
        action -= self.action_range[1]
        return action


    def update_replay_memory(self,last_rl_state, last_action, reward, current_rl_state, done):
        self.agent.update_replay_memory(last_rl_state,last_action,reward, current_rl_state,done )


    def train_agent(self):
        print("Training")
        self.agent.train()


