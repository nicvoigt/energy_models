import os
from Energy_Models.Energy_System_mpc.energy_system_model_mpc import EnergyModel
import pandas as pd
from Energy_Models.utils.paths import paths

df = pd.read_csv(os.path.join(paths.data_dir, "df_merged_2.csv"), index_col=0)
model_characs = {"names": ["PV_scale","SoC_min", "SoC_max", "P_max", "Battery_Efficiency", "SoC_Start", "Stepsize"],
                 "values": [30, 0, 20, 5, 1, 5, 24]}

if __name__ == "__main__":

    em = EnergyModel(model_characs=model_characs, input_data=df, output_name="test")
    em.run_simulation()
