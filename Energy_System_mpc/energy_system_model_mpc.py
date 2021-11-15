from pyomo.environ import Constraint, ConcreteModel, Set, Param, Var, Objective, SolverFactory,\
    NonNegativeReals, minimize
from Energy_Models.utils.mpc_rules import Difference_fixer
from Energy_Models.utils.paths import paths
import os
import pandas as pd

from Energy_Models.Base_Models.Energy_Models import EnergyBaseModel

class EnergyModel(EnergyBaseModel):
    def __init__(self, model_characs, input_data, output_name):
        EnergyBaseModel.__init__(self, model_characs,input_data,output_name)
        self.em = ConcreteModel()
        self.dif_fixer = Difference_fixer(model_characs)

    def create_model(self, timeseries, SoC_start):
        index = timeseries["values"][timeseries["names"].index("Index")]
        electricity_prices = timeseries["values"][timeseries["names"].index("electricity_prices")]
        feed_in_tarrif = timeseries["values"][timeseries["names"].index("feed_in_tarrif")]
        load_forecast = timeseries["values"][timeseries["names"].index("Last")]
        pv_generation = timeseries["values"][timeseries["names"].index("pv_generation")]
        load_real = timeseries["values"][timeseries["names"].index("Reale Last")]
        model = ConcreteModel()
        model.Set = Set(initialize=index)
        model.generation = Param(model.Set, initialize=pv_generation, name="PV-Generation of Household system")

        model.price = Param(model.Set, initialize=electricity_prices,
                            name="Price to purchase electricity from the grid")
        model.reward = Param(model.Set, initialize=feed_in_tarrif, name="Reward for grid feed-in")
        model.last = Param(model.Set, initialize=load_forecast, name="etst")

        # Variables
        model.feed_in = Var(model.Set, within=NonNegativeReals, name="Amount of Energy to feed back into grid")
        model.grid_demand = Var(model.Set, within=NonNegativeReals, name="Amount Energy bought from the grid")

        # Variables to control the storage
        model.charge = Var(model.Set, bounds=(0, self.P_max), name="Energy flow into battery")
        model.discharge = Var(model.Set, bounds=(0, self.P_max), name="Energy flow from battery to the system")
        model.SoC = Var(model.Set, bounds=(self.SoC_min, self.SoC_max), name="Battery's State of Charge")

        model.self_consumption = Var(model.Set, within=NonNegativeReals, name="Energy consumed from PV")

        # Define SoC in the first hour
        model.SoC[0] = SoC_start

        def energy_flow_rule(model, t):
            return model.feed_in[t] + model.charge[t] + model.last[t] == model.generation[t] + model.grid_demand[t] + \
                   model.discharge[t]

        model.energy_flows = Constraint(model.Set, rule=energy_flow_rule)

        def state_of_charge_rule(model, t):
            if t == 0:
                return model.SoC[t] == SoC_start + model.charge[t]*self.Eff - model.discharge[t]/self.Eff
            if t >= 0:
                return model.SoC[(t - 1)] + model.charge[t] * self.Eff - model.discharge[t] / self.Eff == model.SoC[t]

        model.SoC_rule = Constraint(model.Set, rule=state_of_charge_rule)

        def Objective_Function(model):
            return (-sum(model.feed_in[t] * feed_in_tarrif[t] for t in model.Set) + sum(
                (electricity_prices[t]) * model.grid_demand[t] for t in model.Set))

        model.obj_fct = Objective(rule=Objective_Function, sense=minimize)

        return model

    def solve_model(self, model):
        results = SolverFactory("glpk").solve(model)
        # results.write()

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

    def update_signal(self, timestep, model, real_data_sclice):
        """
        updating control trajectory if forecast errors occur
        :param timestep: index to slice the data
        :param model: solved model of the energy system with its variables
        :param real_data_sclice: real data containing the load to adapt the control trajectory if necessary
        :return:
        """
        self.SoC_start = self.dif_fixer.adjustment(timestep, model, real_data_sclice)


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

            model = self.create_model(current_data_named, self.SoC_start)
            self.solve_model(model)
            self.update_signal(model, timestep, current_data_named)

        self.dif_fixer.create_and_save_output(self.output_name)
