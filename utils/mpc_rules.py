# hier einfügen, dass die
import pandas as pd
from pyomo.environ import *

class Difference_fixer:
    def __init__(self, model_characs):
        self.einspeisen_e = []                              # expected feed-in
        self.netzbezug_e = []                               # expected grid-demand
        self.SoC_e = []                                     # expected SoC
        self.posladen_e = []                                # expected charging
        self.negladen_e = []                                # expected discharging
        self.direktverbrauch_e = []                         # direct energy usage form pv

        self.feed_in_r = []                                 # feed-in real
        self.grid_demand_r = []
        self.charge_r = []
        self.discharge_r = []
        self.SoC_r = [model_characs["values"][model_characs["names"].index("SoC_Start")]]
        self.direct_usage_r = []

        ## read model factors:
        self.eff = model_characs["values"][model_characs["names"].index("Battery_Efficiency")]
        self.SoC_max = model_characs["values"][model_characs["names"].index("SoC_max")]
        self.SoC_min = model_characs["values"][model_characs["names"].index("SoC_min")]
        self.P_max = model_characs["values"][model_characs["names"].index("P_max")]

    def adjustment(self, model, timestep, real_data_slice):
        generation = real_data_slice["values"][real_data_slice["names"].index("pv_generation")]
        load_forecast = real_data_slice["values"][real_data_slice["names"].index("Last")]
        load_real = real_data_slice["values"][real_data_slice["names"].index("Reale Last")]
        rewards = real_data_slice["values"][real_data_slice["names"].index("feed_in_tarrif")]
        prices = real_data_slice["values"][real_data_slice["names"].index("electricity_prices")]
        index = [x for x in range(24)]


        einspeisen_e = [model.feed_in[i].value for i in index]
        netzbezug_e = [model.grid_demand[i].value for i in index]
        SoC_e = [model.SoC[i].value for i in index]
        posladen_e = [model.charge[i].value for i in index]
        negladen_e = [model.discharge[i].value for i in index]
        direktv_real = []


        if generation[0] > load_real[0]:
            if value(model.charge[0]) < 0:
                self.charge_r.append(0)
                self.SoC_r.append(self.SoC_r[timestep])
                self.feed_in_r.append(max(min(generation[0] - load_real[0], 9999), 0))
                self.discharge_r.append(0)
                self.grid_demand_r.append(0)



            if value(model.charge[0])>0:
                if (generation[0] - load_real[0])*self.eff > self.SoC_max - self.SoC_r[timestep]:
                    self.charge_r.append(max(min((self.SoC_max - self.SoC_r[timestep]) / self.eff, self.P_max), 0))
                    self.feed_in_r.append(max(min(generation[0] - load_real[0] - self.charge_r[timestep], 9999), 0))
                    self.SoC_r.append(max(min(self.charge_r[timestep] * self.eff + self.SoC_r[timestep], self.SoC_max), 0))

                    self.discharge_r.append(0)
                    self.grid_demand_r.append(0)


                if (generation[0] -load_real[0])*self.eff <=self.SoC_max - self.SoC_r[timestep]:
                    self.charge_r.append(max(min(generation[0] - load_real[0], self.P_max), 0))
                    self.SoC_r.append(max(min(self.charge_r[timestep] * self.eff + self.SoC_r[timestep], self.SoC_max), 0))

                    self.discharge_r.append(0)
                    self.grid_demand_r.append(0)
                    self.feed_in_r.append(0)


            if value(model.charge[0]) == 0:
                self.feed_in_r.append(max(min(generation[0] - load_forecast[0], 9999), 0))

                self.discharge_r.append(0)
                self.grid_demand_r.append(0)
                self.SoC_r.append(self.SoC_r[timestep])
                self.charge_r.append(0)

        if load_real[0] > generation[0] :

            if value(model.discharge[0]) <0:
                self.discharge_r.append(0)
                self.SoC_r.append(self.SoC_r[timestep])
                self.grid_demand_r.append(load_real[0] - generation[0])
                self.feed_in_r.append(0)
                self.charge_r.append(0)


            if value(model.discharge[0]) >0:
                if load_real[0]-generation[0] > (self.SoC_r[timestep] - self.SoC_min)*self.eff:
                    self.discharge_r.append(max(min(self.SoC_r[timestep] - self.SoC_min, self.P_max), 0))
                    self.SoC_r.append(max(min(self.SoC_r[timestep] - self.discharge_r[timestep], self.SoC_max), self.SoC_min))
                    self.grid_demand_r.append(load_real[0] - generation[0] - (self.discharge_r[timestep]) * self.eff) #multipliziert mit Effizienz

                    self.feed_in_r.append(0)
                    self.charge_r.append(0)


                if load_real[0]-generation[0] <= (self.SoC_r[timestep] - self.SoC_min)*self.eff:
                    self.discharge_r.append(max(min(load_real[0] - generation[0], self.P_max), 0))
                    self.SoC_r.append(max(min(self.SoC_r[timestep] - self.discharge_r[timestep] / self.eff, self.SoC_max), self.SoC_min))

                    self.feed_in_r.append(0)
                    self.charge_r.append(0)
                    self.grid_demand_r.append(0)


            if value(model.discharge[0]) == 0:
                self.grid_demand_r.append(load_real[0] - generation[0])

                self.SoC_r.append(self.SoC_r[timestep])
                self.feed_in_r.append(0)
                self.charge_r.append(0)
                self.discharge_r.append(0)


        if load_real[0] == generation[0]:
            self.discharge_r.append(0)
            self.charge_r.append(0)
            self.grid_demand_r.append(0)
            self.feed_in_r.append(0)
            self.SoC_r.append(self.SoC_r[timestep])


        return self.SoC_r[-1]


    def create_and_save_output(self, output_name):
        output_ts = pd.DataFrame(list(zip(self.SoC_r, self.charge_r, self.discharge_r, self.feed_in_r,
                                          self.grid_demand_r)), columns = ["SoC", "posladen", "negladen", "einsp", "netz"])
        # output_ts.

        output_ts.to_csv(f"{output_name}.csv")

        return output_ts