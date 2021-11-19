import pandas as pd
import numpy as np

from Energy_Models.Base_Models.Energy_Models import EnergyBaseModel

class EnergyModel(EnergyBaseModel):
    def __init__(self, model_characs, input_data, output_name):
        EnergyBaseModel.__init__(self, model_characs,input_data,output_name)
        self.soc = np.zeros(len(input_data[0]))


    def run_simulation(self):
        """
        input: energy_system data [prices, fe-tariff, generation, load]
        runs simulation for given timeseries

        creates and solves model for given stepsize until end of simulation horizon is reached.
        saves output file
        :return:
        trajectory of control_data
        """
        for timestep in range(50):

            pv_generation, load_forecast, load_real = self.extract_current_data(timestep)
            signal = self.control_model(pv_generation, load_forecast, load_real, timestep)
            self.update_state(timestep, signal)


    def extract_current_data(self, timestep):
        pv_generation = self.input_data["Erzeugung"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        electricity_prices = self.input_data["Preise"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        feed_in_tarrif = self.input_data["Erloese"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        load_forecast = self.input_data["Prognose"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        load_real = self.input_data["Reale Last"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()

        Index = [timestep]
        curr_data_named = {"names": ["pv_generation","electricity_prices", "feed_in_tarrif", "Last", "Reale Last","Index"],
                           "values": [pv_generation, electricity_prices, feed_in_tarrif, load_forecast, load_real, Index]}

        return pv_generation, load_forecast, load_real

    def control_model(self, pv_generation, load_forecast, load_real, timestep):

        if pv_generation == load_real:
            return 0

        elif pv_generation > load_real:
            surplus = pv_generation - load_real

            if self.soc[timestep] == self.SoC_max:
                feed_in = surplus
                return 0            # return surplus = 0
            elif surplus * self.Eff + self.soc[timestep-1] <= self.SoC_max:        # alles einspeichern
                return surplus * self.Eff
            elif surplus * self.Eff + self.soc[timestep-1] > self.SoC_max:
                max_charge = self.SoC_max - self.soc[timestep -1]
                real_charge = surplus / self.Eff
                feed_in = self.SoC_max - surplus * self.Eff - self.soc[timestep-1]
                return

        elif load_real > pv_generation:
            deficit = load_real - pv_generation

            if self.soc[timestep -1] == 0:
                grid_demand = deficit

            elif self.soc[timestep -1] > 0:
                discharge = s


        return ""

    def update_state(self, timestep, signal):



        self.soc[timestep] = self.soc[timestep-1] + signal



        
