from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory
import numpy as np 
import scipy as sp 
import pandas as pd
import numpy.matlib
import gurobipy

### Defining the Solver
solver = 'gurobi'
solver_io = 'python'

zone2_opt = SolverFactory(solver,solver_io=solver_io)

# Error
if zone2_opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for %s "\
          "using the %s interface" % (solver, solver_io))
    print("")
    exit(1)

###

### Data for Optimal Power Flow Problem
# Base values for Test Network
s_base = 1.0 #kVA
v_base = 4.16 #kV
z_base = (v_base**2)/s_base
T = range(1, 25)

# Test Network Data
#basic_data = pd.read_csv('Emunsing_Data/basic_data.csv',header=None)#, columns=['f','t','r','x'])
fields = ['Node A','Node B','R','X']
basic_data = pd.read_csv(r'Path to file\Line_Data.csv',usecols=fields) # Change your file path
basic_data.columns = ['f','t','r','x']

cutNode = len(basic_data) # include the whole network
cutNode = 16  # Cut the network off into different zones. This is Zone:2
Time = np.int64(24) # if there is no time coupling, leave at 1 for low solve time.
dt = np.int64(1) # time step in hours

# Convert into a model of the network
f = np.array(basic_data['f'].iloc[:cutNode])-1
t = np.array(basic_data['t'].iloc[:cutNode])-1
r = np.array(basic_data['r'].iloc[:cutNode])/z_base
x = np.array(basic_data['x'].iloc[:cutNode])/z_base
#r = np.insert(r,0,0)/z_base
#x = np.insert(x,0,0)/z_base
R = np.diag(r)
X = np.diag(x)
Rsub = R[1:,1:] # need these for constraints later
Xsub = X[1:,1:]

# Network Properties
nb = np.int64(len(t)+1)  # Number of buses
nl = np.int64(len(t))    # Number of lines
no_of_nodes = range(1, nb+1)

# Price of electricity from the grid (if we want grid-connected)
LMP_price = np.array(pd.read_csv(r'Path to file\EnergyPrice.csv',header=None))
price = LMP_price / np.max(LMP_price) # Scale to [0,1]

## Hourly Load data
data_col = ['Load (kW)']

# Number of Houses on each node
noh = 8

load_data = pd.read_csv(r'Path to file\Load_Data_Hourly.csv',usecols=data_col)
load_data = noh * np.array(load_data['Load (kW)'].iloc[:Time]) # Considering 7 houses at each node
fixed_load_p = np.vstack((np.zeros((1,Time)),np.full((nl,Time), load_data)))/s_base
fixed_load_q = np.vstack((np.zeros((1,Time)),noh*(100 + (200-1002)*np.random.rand(nl,Time))))/(s_base*1e3)

# Scale loads using the price profile, with random noise 20% of the signal
fixed_load_p = 0.3 * fixed_load_p + 0.7 * np.dot(np.asmatrix(np.mean(fixed_load_p,axis=1)).T,price.transpose())
fixed_load_q = 0.3 * fixed_load_q + 0.7 * np.dot(np.asmatrix(np.mean(fixed_load_q,axis=1)).T,price.transpose())
##

## Hourly PV Data
# Solar data - NSRDB 2017 database, evaluated in SAM for PV Panel - SunPower SPR-X21-335
PV_data_col = ['PV (kW)']
p_sol = pd.read_csv(r'Path to file\PV_hourly_data.csv',usecols=PV_data_col) # Just take solar output (not hours)
p_sol = noh * np.array(p_sol['PV (kW)'].iloc[:Time]) # Considering 7 houses at each node
p_sol = np.vstack((np.zeros((1,Time)),np.full((nl,Time), p_sol)))# Stack for each node
##

## Battery Energy Storage Model and Data
eff_in = 0.95 # Battery efficiency
eff_out = 0.95 
p_battmax = np.max(fixed_load_p,axis=1)*0.5
p_battmax = np.array(p_battmax.T)
p_battmax = p_battmax.ravel()
hours = 5
battsize = hours * p_battmax          # Battery size in Watts*dt (i.e. Wh if dt=1 hour)
E_min = 0.2 * battsize  # SOC lower limits on batteries
E_max = 0.8 * battsize  # SOC lower limits on batteries
# E_init = 0.5 * battsize # Starting state of charge
eps = 0.001              # Acceptable band for ending SOC
##

## Shapeable load parameters
P_shpmin = 0 * np.max(fixed_load_p,axis=1)
P_shpmax = 2 * np.max(fixed_load_p,axis=1)
E_dem = 3 * np.multiply(np.random.rand(nb,1),P_shpmax)    # Effectively how many hours of demand we have
P_shpmin = np.array(P_shpmin.T)
P_shpmin = P_shpmin.ravel()
P_shpmax = np.array(P_shpmax.T)
P_shpmax = P_shpmax.ravel()
E_dem = np.array(E_dem)
E_dem = E_dem.ravel()
minStartTime = 9
minStartTime = np.min([Time-4,minStartTime]) # Deal with short time horizons
startProbPeriod = 9 # Start times will be uniformly distributed over this range
startAfter = minStartTime + np.round(startProbPeriod * np.random.rand(nb,1).flatten())
chargeTime = 4 + np.round(np.min([6,Time-4]) * np.random.rand(nb,1).flatten())
endBy = np.minimum(startAfter + chargeTime,Time)
##

## Generator Coefficients
alpha = np.random.rand(nb,1)
alpha = np.array(alpha.T)
alpha = alpha.ravel()
beta = np.vstack((LMP_price.T,20*np.random.rand(nb-1,Time)))
Pg_min = np.zeros((nb,Time)) # These are setup to give feeder lots of power, other buses less so.
Pg_max = (0.5e3 + (1e3-0.5e3)*np.random.rand(nl,1))
Pg_max = np.insert(Pg_max,0,nl*12e3)/(s_base*1.0e3)
Pg_max = numpy.matlib.repmat(np.asmatrix(Pg_max).transpose(),1,Time)
##

# Create a matrix where infeasible hours are 0 - buses in rows, times in columns
selectShp = np.zeros((nb,Time))
for i in range(nb):# = 1:nb:
    tooEarly = np.ones(  ( 1,int(startAfter[i]) ) )
    okTimes  = np.zeros( ( 1,int(endBy[i]-startAfter[i]) ) )
    tooLate  = np.ones(  ( 1,int(Time-endBy[i]) ) )
    selectShp[i,:] = np.hstack(( tooEarly, okTimes, tooLate))
##

# Create a Pandas Dataframe to Generate a OPF Model
df = pd.DataFrame({"T": T})
df1 = pd.DataFrame({"Nodes": no_of_nodes})
df2 = pd.DataFrame({"From_Nodes": f+1,
                    "To_Nodes": t+1
})
df3 = pd.DataFrame({"FN": f+1,
                    "TN": t+1,
                    "r": r,
                    "x": x
})
df4 = pd.DataFrame({"T": T})
df4 = df4.T.apply(pd.Series)
df5 = pd.DataFrame(fixed_load_p)
df6 = pd.DataFrame({"P_Load": no_of_nodes})
df7 = pd.DataFrame({"T": T})
df7 = df7.T.apply(pd.Series)
df8 = pd.DataFrame(p_sol)
df9 = pd.DataFrame({"P_PV": no_of_nodes})
df10 = pd.DataFrame({"T": T})
df10 = df10.T.apply(pd.Series)
df11 = pd.DataFrame(fixed_load_q)
df12 = pd.DataFrame({"Q_Load": no_of_nodes})
df13 = pd.DataFrame({"Nodes": no_of_nodes,
                    "P_battmax": p_battmax.T,
                    "battsize": battsize.T,
                    "E_min": E_min.T,
                    "E_max": E_max
})
df14 = pd.DataFrame({"Nodes": no_of_nodes,
                    "P_shpmax": P_shpmax,
                    "P_shpmin": P_shpmin,
                    "E_dem": E_dem
})
df15 = pd.DataFrame({"T": T})
df15 = df15.T.apply(pd.Series)
df16 = pd.DataFrame(selectShp)
df17 = pd.DataFrame({"selectShp": no_of_nodes})
df18 = pd.DataFrame({"Nodes": no_of_nodes,
                    "alpha": alpha
})
df19 = pd.DataFrame({"T": T})
df19 = df19.T.apply(pd.Series)
df20 = pd.DataFrame(beta)
df21 = pd.DataFrame({"beta": no_of_nodes})
df22 = pd.DataFrame({"T": T})
df22 = df22.T.apply(pd.Series)
df23 = pd.DataFrame(Pg_min)
df24 = pd.DataFrame({"Pg_min": no_of_nodes})
df25 = pd.DataFrame({"T": T})
df25 = df25.T.apply(pd.Series)
df26 = pd.DataFrame(Pg_max)
df27 = pd.DataFrame({"Pg_max": no_of_nodes})
df28 = pd.DataFrame({"eff_in": [eff_in],
                    "eff_out": [eff_out],
                    "dt": [dt],
                    "eps":[eps],
                    "v_min": [3.952],
                    "v_max": [4.368]
})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter("zone2_linear_opt_model.xlsx", engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object. Note that we turn off
# the default header and skip one row to allow us to insert a user defined
# header.
df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0)
df1.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T))
df2.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T))
df3.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T))
df4.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 2)
df5.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 2, startrow=len(df4) + 1, header=False)
df6.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 1, startrow=len(df4))
df7.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 2, startrow=len(df4) + len(df5) + 2)
df8.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 2, startrow=len(df4) + len(df5) + len(df7) + 3, header=False)
df9.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + 1, startrow=len(df4) + len(df5) + 3)
df10.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + 4)
df11.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + 4, startrow=len(df4) + 1, header=False)
df12.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + 3, startrow=len(df4))
df13.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + 5)
df14.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + 6)
df15.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + 8)
df16.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + 8, startrow=len(df15) + 1, header=False)
df17.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + 7, startrow=len(df15))
df18.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + 9)
df19.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + 11)
df20.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + 11, startrow=len(df19) + 1, header=False)
df21.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + 10, startrow=len(df19))
df22.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 13)
df23.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 13, startrow=len(df22) + 1, header=False)
df24.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 12, startrow=len(df22))
df25.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 13, startrow=len(df22) + len(df23) + 2)
df26.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 13, startrow=len(df22) + len(df23) + len(df25) + 3, header=False)
df27.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0 + len(df.T) + len(df1.T) + len(df2.T) + len(df3.T) + len(df4.T) + len(df10.T) + len(df13.T) + len(df14.T) + len(df15.T) + len(df18.T) + len(df19.T) + 12, startrow=len(df22) + len(df23) + 3)
df28.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0, startrow=len(df) + 2)

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book
workbook.define_name('T', '=Sheet1!$A$1:$A$25')
workbook.define_name('Nodes', '=Sheet1!$B$1:$B$18')
workbook.define_name('FT_Nodes', '=Sheet1!$C$1:$D$17')
workbook.define_name('RX', '=Sheet1!$E$1:$H$17')
workbook.define_name('P_Load', '=Sheet1!$J$2:$AH$19')
workbook.define_name('Q_Load', '=Sheet1!$AJ$2:$BH$19')
workbook.define_name('P_PV', '=Sheet1!$J$22:$AH$39')
workbook.define_name('BESS', '=Sheet1!$BJ$1:$BN$18')
workbook.define_name('P_Shp_Load', '=Sheet1!$BP$1:$BS$18')
workbook.define_name('selectShp', '=Sheet1!$BU$2:$CS$19')
workbook.define_name('alpha', '=Sheet1!$CU$1:$CV$18')
workbook.define_name('beta', '=Sheet1!$CX$2:$DV$19')
workbook.define_name('Pg_min', '=Sheet1!$DX$2:$EV$19')
workbook.define_name('Pg_max', '=Sheet1!$DX$22:$EV$39')
workbook.define_name('eff_in', '=Sheet1!$A$28')
workbook.define_name('eff_out', '=Sheet1!$B$28')
workbook.define_name('dt', '=Sheet1!$C$28')
workbook.define_name('eps', '=Sheet1!$D$28')
workbook.define_name('v_min', '=Sheet1!$E$28')
workbook.define_name('v_max', '=Sheet1!$F$28')


# Close the Pandas Excel writer and output the Excel file.
writer.save()


### Create a Non-Linear OPF Model for Zone-2 for IEEE-123 Node Network
model = AbstractModel()

## Declaring Sets
model.T = Set() # Time period
model.Nodes = Set() # Number of Nodes
model.FT_Nodes = Set(within=model.Nodes * model.Nodes) # Network of nodes
##

## Creating Parameters
model.R = Param(model.FT_Nodes)
model.X = Param(model.FT_Nodes)
model.P_fx_Load = Param(model.Nodes, model.T)
model.Q_fx_Load = Param(model.Nodes, model.T)
model.P_PV = Param(model.Nodes, model.T)
# Parameters for Battery Energy Storage
model.P_battmax = Param(model.Nodes)
model.battsize = Param(model.Nodes)
model.E_min = Param(model.Nodes)
model.E_max = Param(model.Nodes)
model.eff_in = Param(mutable=True, initialize=0.95)
model.eff_out = Param(mutable=True, initialize=0.95)
model.dt = Param(mutable=True, initialize=1)
model.eps = Param(mutable=True, initialize=0.001)
model.v_min = Param(mutable=True, initialize = 3.952)
model.v_max = Param(mutable = True, initialize = 4.368)
# Parameters for Shapeable Load
model.P_shpmax = Param(model.Nodes)
model.P_shpmin = Param(model.Nodes)
model.E_dem = Param(model.Nodes)
model.selectShp = Param(model.Nodes, model.T)
# Parameters for Generator
model.alpha = Param(model.Nodes)
model.beta = Param(model.Nodes, model.T)
model.Pg_min = Param(model.Nodes, model.T)
model.Pg_max = Param(model.Nodes, model.T)

data = DataPortal(model=model)
data.load(filename="zone2_linear_opt_model.xlsx", range="T", format='set', set='T')
data.load(filename="zone2_linear_opt_model.xlsx", range="Nodes", format='set', set='Nodes')
data.load(filename="zone2_linear_opt_model.xlsx", range="FT_Nodes", format='set', set='FT_Nodes')
data.load(filename="zone2_linear_opt_model.xlsx", range="RX", index='FT_Nodes', param=('R','X'))
data.load(filename="zone2_linear_opt_model.xlsx", range="P_Load", param='P_fx_Load', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="Q_Load", param='Q_fx_Load', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="P_PV", param='P_PV', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="BESS", index='Nodes', param=('P_battmax','battsize', 'E_min', 'E_max'))
data.load(filename="zone2_linear_opt_model.xlsx", range="P_Shp_Load", index='Nodes', param=('P_shpmax','P_shpmin', 'E_dem'))
data.load(filename="zone2_linear_opt_model.xlsx", range="selectShp", param='selectShp', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="alpha", index='Nodes', param=('alpha'))
data.load(filename="zone2_linear_opt_model.xlsx", range="beta", param='beta', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="Pg_min", param='Pg_min', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="Pg_max", param='Pg_max', format='array')
data.load(filename="zone2_linear_opt_model.xlsx", range="eff_in", format='param', param='eff_in')
data.load(filename="zone2_linear_opt_model.xlsx", range="eff_out", format='param', param='eff_out')
data.load(filename="zone2_linear_opt_model.xlsx", range="dt", format='param', param='dt')
data.load(filename="zone2_linear_opt_model.xlsx", range="eps", format='param', param='eps')
data.load(filename="zone2_linear_opt_model.xlsx", range="v_min", format='param', param='v_min')
data.load(filename="zone2_linear_opt_model.xlsx", range="v_max", format='param', param='v_max')

## Declaring Variables
# Declaring Network Variables
model.P = Var(model.Nodes, model.T)
model.Q = Var(model.Nodes, model.T)
model.P_FT = Var(model.FT_Nodes, model.T, initialize=0)
model.Q_FT = Var(model.FT_Nodes, model.T, initialize=0)
model.v = Var(model.Nodes, model.T, initialize=4.16)
model.l = Var(model.Nodes, model.T)
model.p = Var(model.Nodes, model.T)
model.q = Var(model.Nodes, model.T)
model.v_parent = Var(model.FT_Nodes, model.T, initialize=4.16)

# Declaring Variables for Generator
model.Pg = Var(model.Nodes, model.T, initialize=160)
model.Qg = Var(model.Nodes, model.T, initialize=160)

# Declaring Variables for Battery Energy Storage System
model.P_bc = Var(model.Nodes, model.T)
model.P_bd = Var(model.Nodes, model.T)
model.P_batt = Var(model.Nodes, model.T)
model.E_batt = Var(model.Nodes, model.T)

# Declaring Variables for Shapeable Load
model.P_shp = Var(model.Nodes, model.T)
##

# Define Objective Function
def cost(model):
    return (sum(model.alpha[k] * (model.Pg[k, t]**2) + model.beta[k, t] * model.Pg[k, t] for k in model.Nodes for t in model.T))
model.cost = Objective(rule = cost, sense = minimize)

## Declaring Constraints for OPF model
# Active Power Balance On a Particular Node
def ap_balance(model, k, t):
    return (model.p[k, t] == model.Pg[k, t] + model.P_PV[k, t] + model.P_bd[k, t] - model.P_bc[k, t] - model.P_fx_Load[k, t] - model.P_shp[k, t])
model.ap_balance = Constraint(model.Nodes, model.T, rule = ap_balance)

# Reactive Power Balance on a Particular Node
def rq_balance(model, k, t):
    return (model.q[k, t] == model.Qg[k, t] - model.Q_fx_Load[k, t])
model.rq_balance = Constraint(model.Nodes, model.T, rule = rq_balance)

## Generator Constraints
# Generator Minimum
def gc_min(model, k, t):
    return (model.Pg_min[k, t] <= model.Pg[k, t])
model.gc_min = Constraint(model.Nodes, model.T, rule = gc_min)

# Generator Maximum
def gc_max(model, k, t):
    return (model.Pg[k, t] <= model.Pg_max[k, t])
model.gc_max = Constraint(model.Nodes, model.T, rule = gc_max)

## Energy Storage
def pbdch_min(model, k ,t):
    return (0 <= model.P_bd[k, t])
model.pbdch_min = Constraint(model.Nodes, model.T, rule = pbdch_min)

def pbdch_max(model, k ,t):
    return (model.P_bd[k, t] <= model.P_battmax[k])
model.pbdch_max = Constraint(model.Nodes, model.T, rule = pbdch_max)

def pbch_min(model, k ,t):
    return (0 <= model.P_bc[k, t])
model.pbch_min = Constraint(model.Nodes, model.T, rule = pbch_min)

def pbch_max(model, k ,t):
    return (model.P_bc[k, t] <= model.P_battmax[k])
model.pbch_max = Constraint(model.Nodes, model.T, rule = pbch_max)

def battp_balance(model, k, t):
    return (model.P_bd[k, t] - model.P_bc[k, t] == model.P_batt[k, t])
model.battp_balance = Constraint(model.Nodes, model.T, rule = battp_balance)

def e_min(model, k, t):
    return (model.E_min[k] <= model.E_batt[k, t])
model.e_min = Constraint(model.Nodes, model.T, rule = e_min)

def e_max(model, k, t):
    return (model.E_batt[k, t] <= model.E_max[k])
model.e_max = Constraint(model.Nodes, model.T, rule = e_max)

def bess_balance(model, k, t):
    if t < 2:
        return (model.E_batt[k, t] == model.E_batt[k, t+23] + model.P_bc[k, t] * model.dt * model.eff_in - model.P_bd[k, t] * model.dt/model.eff_out)
    return (model.E_batt[k, t] == model.E_batt[k, t-1] + model.P_bc[k, t] * model.dt * model.eff_in - model.P_bd[k, t] * model.dt/model.eff_out)
model.bess_balance = Constraint(model.Nodes, model.T, rule = bess_balance)

def bess_eps1(model, k, t):
    if t < 2:
        return ((1-model.eps) * model.E_batt[k, t] <= model.E_batt[k, t+23])
    else:
        return Constraint.Skip
model.eps1 = Constraint(model.Nodes, model.T, rule = bess_eps1)

def bess_eps2(model, k, t):
    if t < 2:
        return (model.E_batt[k, t+23] <= (1+model.eps) * model.E_batt[k, t])
    else:
        return Constraint.Skip
model.eps2 = Constraint(model.Nodes, model.T, rule = bess_eps2)

# Shapeable Load
def pshp_min(model, k, t):
    return (model.P_shpmin[k] <= model.P_shp[k, t])
model.pshp_min = Constraint(model.Nodes, model.T, rule = pshp_min)

def pshp_max(model, k, t):
    return (model.P_shp[k, t] <= model.P_shpmax[k])
model.pshp_max = Constraint(model.Nodes, model.T, rule = pshp_max)

# Shapeable Load Total Energy Consumption
def pshp_dem(model, k):
    return sum(model.P_shp[k, t] for t in model.T) == model.E_dem[k]
model.pshp_dem = Constraint(model.Nodes, rule = pshp_dem)

# Shapeable Load Start and End Time
def pshp_SEtime(model, k, t):
    if model.selectShp[k, t] == 1:
        return (model.P_shp[k, t] == 0)
    else:
        return Constraint.Skip
model.pshp_SEtime = Constraint(model.Nodes, model.T, rule = pshp_SEtime)

# Shapeable Load Should be Greater than or Equal to Zero
def pshp_load(model, k, t):
    return model.P_shp[k, t] >= 0
model.pshp_load = Constraint(model.Nodes, model.T, rule = pshp_load)

## Network Constraints
# Root Node Constraint
def Proot(model, k, t):
    if k == 1:
        return (model.P[k, t] == 0)
    else:
        return Constraint.Skip
model.Proot = Constraint(model.Nodes, model.T, rule = Proot)

def Qroot(model, k, t):
    if k == 1:
        return (model.P[k, t] == 0)
    else:
        return Constraint.Skip
model.Qroot = Constraint(model.Nodes, model.T, rule = Qroot)

# Power Flow Equations
def Pflow(model, k, t):
    return (model.p[k, t] == model.P[k, t] - sum(model.P_FT[i, j, t] for i,j in model.FT_Nodes if k==i))
model.Pflow = Constraint(model.Nodes, model.T, rule = Pflow)

def Qflow(model, k, t):
    return (model.q[k, t] == model.Q[k, t] - sum(model.Q_FT[i, j, t] for i,j in model.FT_Nodes if k==i))
model.Qflow = Constraint(model.Nodes, model.T, rule = Qflow)

def Peq(model, i, j, t):
        return (model.P_FT[i, j, t] == model.P[j, t])
model.Peq = Constraint(model.FT_Nodes, model.T, rule = Peq)

def Qeq(model, i, j, t):
        return (model.Q_FT[i, j, t] == model.Q[j, t])
model.Qeq = Constraint(model.FT_Nodes, model.T, rule = Qeq)

def zvolt(model, k, i, j, t):
    if k == j:
        return (model.v[k, t] == model.v[i, t] + 2*(model.R[i, j] * model.P[k, t]) + 2*(model.X[i, j] * model.Q[k, t]))
    else:
        return Constraint.Skip
model.zvolt = Constraint(model.Nodes, model.FT_Nodes, model.T, rule = zvolt)

#def veq(model, i, j, t):
        #return (model.v_parent[i, j, t] == model.v[i, t])
#model.veq = Constraint(model.FT_Nodes, model.T, rule = veq)

def vroot(model, k, t):
    if k == 1:
        return (model.v[k, t] == 4.16)
    else:
        return Constraint.Skip
model.vroot = Constraint(model.Nodes, model.T, rule = vroot)

def v_min(model, k, t):
    return(model.v[k, t] >= 3.952)
model.v_min = Constraint(model.Nodes, model.T, rule = v_min)

def v_max(model, k , t):
    return(model.v[k, t] <= 4.368)
model.v_max = Constraint(model.Nodes, model.T, rule = v_max)

instance = model.create_instance(data)
instance.pprint()

results = zone2_opt.solve(instance)

results.write()

# Generate Results
P_g = np.zeros(Time)
Psol = np.zeros(Time)
Pbatt = np.zeros(Time)
Ebatt = np.zeros(Time)
PshpLoad = np.zeros(Time)
PfxLoad = np.zeros(Time)

for t in range(1, Time+1):
    for i in range(1, cutNode+2):
        if i == 1:
            P_g[t-1] = -instance.Pg[i,t].value
            Psol[t-1] = -instance.P_PV[i,t]
            Pbatt[t-1] = -instance.P_batt[i,t].value
            Ebatt[t-1] = instance.E_batt[i,t].value
            PshpLoad[t-1] = instance.P_shp[i,t].value
            PfxLoad[t-1] = instance.P_fx_Load[i,t]
        else:
            P_g[t-1] = P_g[t-1] - instance.Pg[i,t].value
            Psol[t-1] = Psol[t-1] - instance.P_PV[i,t]
            Pbatt[t-1] = Pbatt[t-1] - instance.P_batt[i,t].value
            Ebatt[t-1] = Ebatt[t-1] + instance.E_batt[i,t].value
            PshpLoad[t-1] = PshpLoad[t-1] + instance.P_shp[i,t].value
            PfxLoad[t-1] = PfxLoad[t-1] + instance.P_fx_Load[i,t]

opt_data = pd.DataFrame({'P_Gen': P_g,
		                'P_Bat': Pbatt,
                        'E_Bat': Ebatt,
                         'P_Solar': Psol,
                         'P_shp_Load': PshpLoad,
                         'P_fixed_Load': PfxLoad
})

filepath = 'zone2_linear_opt_data.xlsx'
opt_data.to_excel(filepath)

voltage = np.zeros((cutNode+1, Time))
for t in range(1, Time+1):
    for i in range(1, cutNode+2):
        voltage[i-1,t-1] = instance.v[i,t].value

print(voltage)
