import os
import pandas as pd
import numpy as np
from Energy_Models.Energy_System_rl.rl_agent import RL_Agent
from Energy_Models.Base_Models.Energy_Models import EnergyBaseModel_rl


class Energy_system_rl(EnergyBaseModel_rl):
    def __init__(self, model_characs, input_data, output_name, no_episodes, epsilon_decay):
        EnergyBaseModel_rl.__init__(self, model_characs, input_data, output_name,no_episodes, epsilon_decay)

        ## Energy System Parameters



        # RL-Parameters
        self.no_episodes = no_episodes
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.timestep = 0
        self.action_range = (-self.P_max, self.P_max)
        self.agent = RL_Agent(SoC_start=self.SoC_start, action_range=self.action_range)

        self.len_ts = len(self.input_data.iloc[:,0])
        self.action_range = (-self.P_max, self.P_max)
        self.last_timestep = self.len_ts


        # RL- Variables
        self.current_rl_state = [0,0,0]
        self.last_rl_state = self.current_rl_state
        self.episode_rewards = []
        self.training_rewards = []


        ## Data
        self.pv_generation = self.input_data["Erzeugung"]
        self.electricity_prices = self.input_data["Preise"]
        self.feed_in_tarrif = self.input_data["Erloese"]
        self.load_forecast = self.input_data["Prognose"]
        self.load_real = self.input_data["Reale Last"]


        ## Control variables
        self.charging = 0
        self.soc = self.SoC_start
        self.last_soc = 0


    def choose_action(self):
        action = self.agent.choose_action(self.current_rl_state, self.epsilon)
        return action

    def get_state(self, action=0):

        self.charging = action

        self.last_soc = self.soc
        self.soc = self.last_soc + self.charging

        self.last_rl_state = self.current_rl_state

        self.current_rl_state = [self.soc, self.pv_generation[self.timestep], self.load_real[self.timestep]]

        reward = self.calc_reward()

        if self.timestep == self.last_timestep:
            done = True
        else:
            done = False
        self.last_action = action

        if self.timestep >0:
            self.agent.update_replay_memory(
                *(self.last_rl_state, self.last_action, float(reward), self.current_rl_state, done))



    def calc_reward(self):
        grid_demand = 0
        feed_in = 0
        if self.pv_generation[self.timestep] - self.load_real[self.timestep] - self.charging >= 0:
            feed_in = self.pv_generation[self.timestep] - self.load_real[self.timestep] - self.charging
            reward = self.feed_in_tarrif[self.timestep]*feed_in
        elif self.pv_generation[self.timestep] - self.load_real[self.timestep] - self.charging < 0:
            grid_demand = self.pv_generation[self.timestep] - self.load_real[self.timestep] - self.charging
            reward = self.electricity_prices[self.timestep]*grid_demand

        self.episode_rewards.append(reward)

        return reward


    def get_last_state(self):
        reward = self.calc_reward()


    def end_episode(self):
        self.training_rewards.append(sum(self.episode_rewards))
        self.episode_rewards = []


    def run_simulation(self):
        self.training_runs()
        self.save_outputs()

    def save_outputs(self):
        output_df = pd.DataFrame()
        output_df = pd.concat([output_df, pd.Series(self.training_rewards)], axis=1)
        output_df.to_csv(os.path.join(self.output_name+ ".csv"))

    def training_runs(self):
        for episode in range(self.no_episodes):
            print(f"EpisodenNummer ={episode} ")
            for self.timestep in range(self.last_timestep):

                if self.timestep == 0:
                    self.get_state(action=0)
                    self.choose_action()
                elif self.timestep < self.len_ts:
                    action = self.choose_action()
                    self.get_state(action)
                elif self.timestep == self.len_ts:
                    self.get_last_state()

                if self.timestep %400==0:
                    print(self.timestep)
                    self.agent.train_agent()


            self.end_episode()





            self.epsilon *= (1-self.epsilon_decay)


