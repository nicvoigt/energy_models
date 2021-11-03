import pandas as pd
import os
from Energy_Models.utils.paths import paths
from Energy_Models.Energy_System_rl.energy_system_model_rl import Energy_system_rl




df = pd.read_csv(os.path.join(paths.data_dir, "df_merged_2.csv"), index_col=0)
model_characs = {"names": ["PV_scale","SoC_min", "SoC_max", "P_max", "Battery_Efficiency", "SoC_Start", "Stepsize"],
                 "values": [30, 0, 20, 5, 1, 5, 24]}

if __name__ == "__main__":

    em = Energy_system_rl(model_characs=model_characs, input_data=df, output_name="test", no_episodes=30, epsilon_decay=0.05)
    em.run_simulation()
