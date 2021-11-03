import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

file_path= r"C:\Users\nicoj\netcase\1-Start-UP\1-Post Masterarbeit\Programming-Artikel\Pyomo-Serie"
df = pd.read_csv(os.path.join(file_path,"Input_Daten.csv"), index_col = 0)

#extract data from df to list / dict
Index = df.index.tolist()
electricity_prices= df["Price"].to_dict()
feed_in_tarrif = df["Revenue"].to_dict()
Last = df["Energy"].to_dict()
pv_generation = df["Generation"].to_dict()


# Battery characteristics
SoC_max = 10    # Max Capacity in kWh
SoC_min = 0     # Min Capacity in kWh
P_max = 5       # Max Load in kW
Eff = 0.9       # Battery Efficiency

SoC_start = 0   # Capacity at t=0 in kWh


# Creation of model
model = ConcreteModel()

# Initialize Set with Index
model.Set = Set(initialize=Index)

# Parameters

model.generation = Param(model.Set, initialize=pv_generation, name="PV-Generation of Household system")

model.price = Param(model.Set, initialize = electricity_prices, name="Price to purchase electricity from the grid")
model.reward = Param(model.Set, initialize = feed_in_tarrif, name="Reward for grid feed-in")
model.wert = Param(model.Set, initialize=Last, name="")

# Variables
model.feed_in = Var(model.Set, within=NonNegativeReals, name="Amount of Energy to feed back into grid")
model.grid_demand = Var(model.Set, within=NonNegativeReals, name="Amount Energy bought from the grid")

# Variables to control the storage
model.charge = Var(model.Set, bounds=(0, P_max), name="Energy flow into battery")
model.discharge = Var(model.Set, bounds=(0, P_max), name="Energy flow from battery to the system")
model.SoC = Var(model.Set, bounds=(SoC_min, SoC_max), name="Battery's State of Charge")

model.self_consumption = Var(model.Set, within=NonNegativeReals, name="Energy consumed from PV")



# Define SoC in the first hour
model.SoC[0] = SoC_start

# define battery discharging in the first hour
def load_first_hour(model, t):
    return model.discharge[0] == 0

model.load_first_t = Constraint(model.Set, rule=load_first_hour)


def energy_flow_rule(model,t):
    return model.feed_in[t] + model.charge[t] + model.wert[t] == model.generation[t] + model.grid_demand[t] + model.discharge[t]

model.energy_flows = Constraint(model.Set, rule=energy_flow_rule)


def State_of_Charge_rule(model, t):
    if t == 0:
        return model.SoC[t] == SoC_start
    if t > 0:
        return model.SoC[(t - 1)] + model.charge[t] * Eff - model.discharge[t] / Eff == model.SoC[t]

model.SoC_rule = Constraint(model.Set, rule=State_of_Charge_rule)



def Objective_Function(model):
    return (-sum(model.feed_in[t] * (feed_in_tarrif[t]) for t in model.Set) + sum(
        (electricity_prices[t]) * model.grid_demand[t] for t in model.Set))

model.obj_fct = Objective(rule=Objective_Function, sense=minimize)

# solving the model
results = SolverFactory('glpk').solve(model)
results.write()

feed_in_e = [model.feed_in[t].value for t in Index]
grid_demand_e = [model.grid_demand[t].value for t in Index]
SoC_e = [model.SoC[t].value for t in Index]
charge_e = [model.charge[t].value for t in Index]
discharge_e = [model.discharge[t].value for t in Index]

plt.plot(SoC_e)
