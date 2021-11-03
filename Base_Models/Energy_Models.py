import os
from Energy_Models.utils.paths import paths


class EnergyBaseModel:
    def __init__(self, model_characs, input_data, output_name):
        assert model_characs["names"] == ["PV_scale", "SoC_min", "SoC_max", "P_max",
                                          "Battery_Efficiency", "SoC_Start", "Stepsize"],\
                "Check Model input characteristics."

        self.SoC_min = model_characs["values"][model_characs["names"].index("SoC_min")]
        self.SoC_max = model_characs["values"][model_characs["names"].index("SoC_max")]
        self.P_max = model_characs["values"][model_characs["names"].index("P_max")]
        self.Eff = model_characs["values"][model_characs["names"].index("Battery_Efficiency")]
        self.SoC_start = model_characs["values"][model_characs["names"].index("SoC_Start")]
        self.output_name = os.path.join(paths.output_mpc, output_name)
        pv_scale = model_characs["values"][model_characs["names"].index("PV_scale")]

        self.input_data = input_data
        self.input_data["Prognose"] /= 50
        self.input_data["Reale Last"] /= 50
        self.input_data["Erzeugung"] *= pv_scale
        self.step_size = model_characs["values"][model_characs["names"].index("Stepsize")]

class EnergyBaseModel_rl(EnergyBaseModel):
    def __init__(self, model_characs, input_data, output_name, no_episodes, epsilon_decay):
        EnergyBaseModel.__init__(self, model_characs, input_data, output_name)
        self.episodes = no_episodes
        self.output_name = os.path.join(paths.output_rl, output_name)

        self.pv_generation = self.input_data["Erzeugung"]
        self.electricity_prices = self.input_data["Preise"]
        self.feed_in_tarrif = self.input_data["Erloese"]
        self.load_forecast = self.input_data["Prognose"]
        self.load_real = self.input_data["Reale Last"]

        self.epsilon = 1
        self.epsilon_decay = epsilon_decay