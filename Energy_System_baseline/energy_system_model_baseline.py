import pandas as pd

from Energy_Models.Base_Models.Energy_Models import EnergyBaseModel

class EnergyModel(EnergyBaseModel):
    def __init__(self, model_characs, input_data, output_name):
        EnergyBaseModel.__init__(self, model_characs,input_data,output_name)

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

            current_data_named = self.extract_current_data(timestep)


            self.update_signal(model, timestep, current_data_named)

        self.dif_fixer.create_and_save_output(self.output_name)

    def extract_current_data(self, timestep):
        pv_generation = self.input_data["Erzeugung"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        electricity_prices = self.input_data["Preise"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        feed_in_tarrif = self.input_data["Erloese"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        load_forecast = self.input_data["Prognose"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()
        load_real = self.input_data["Reale Last"].iloc[timestep:(self.step_size + timestep)].reset_index(drop = True).to_dict()

        Index = [x for x in range(24)]
        curr_data_named = {"names": ["pv_generation","electricity_prices", "feed_in_tarrif", "Last", "Reale Last","Index"],
                           "values": [pv_generation, electricity_prices, feed_in_tarrif, load_forecast, load_real, Index]}

        return curr_data_named

    def control_model(self, current_data_named):
        
