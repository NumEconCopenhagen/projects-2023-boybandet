from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import sympy as sm
from sympy import *
from IPython.display import display


class OLGModelClass():

    def __init__(self,do_print=True): # initialize the model
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 1 # CRRA coefficient
        par.beta = 1/1.40 # discount factor

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.theta = 0.0 # substitution parameter
        par.delta = 0.50 # depreciation rate

        # c. government
        par.tau_w = 0 # labor income tax
        par.tau_r = 0 # capital income tax
        par.bal_budget = True # balanced budget

        # d. misc
        par.K_lag_ini = 0 + 1e-08 # initial capital stock
        par.B_lag_ini = 0.0 # initial government debt
        par.A_lag_ini = 1.0 # initial productivity
        
        # e. simulation length
        par.simT = 50 # length of simulation
        
        #f. population and productivity growth
        par.n = 0.02 # population growth rate
        par.L_lag_ini = 1.0 # initial labor supply
        par.g = 0.03 # productivity growth rate

        #g steady state
        par.k_ss = ((1-par.alpha)/((1+par.n)*(1+1.0/par.beta)))**(1/(1-par.alpha)) # steady state capital stock


    def allocate(self): # allocate arrays for simulation
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables to store results from simulation
        household = ['C1','C2'] # consumption
        firm = ['K','Y','K_lag','k','k_lag'] # capital stock, output, capital stock at time t and t-1, capital per worker at time t and t-1
        prices = ['w','rk','rb','r','rt'] # wages, rental rates, interest rates
        government = ['G','T','B','balanced_budget','B_lag'] # government debt at time t and t-1
        population = ['L','L_lag'] # labor supply at time t and t-1
        productivity = ['A','A_lag'] # productivity at time t and t-1

        # b. empty arrays to store simulation results each named according to the variable names
        allvarnames = household + firm + prices + government + population + productivity 
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def annakss(self): # steady state capital per worker for plotting
        par = self.par # call parameters
        sim = self.sim  # call parameters

        k_ss = ((1-par.alpha)/((1+par.g)*(1+par.n)*(1+1.0/par.beta)))**(1/(1-par.alpha)) # steady state capital stock

        return k_ss # return steady state capital per worker


    def simulate(self,do_print=True): # simulate model
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values for simulation
        sim.K_lag[0] = par.K_lag_ini # initial capital stock
        sim.B_lag[0] = par.B_lag_ini # initial government debt
        sim.L_lag[0] = par.L_lag_ini # initial labor supply
        sim.k_lag[0] = par.K_lag_ini /(par.L_lag_ini * par.A_lag_ini) # initial capital per worker
        sim.A_lag[0] = par.A_lag_ini # initial productivity

        # b. iterate  over time and simulate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue   # skip last period       

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t) # find bracket for s to search in

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t) # define objective function
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect') 
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s) 

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs') # print time elapsed

def find_s_bracket(par,sim,t,maxiter=10000,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t) # euler error
    sign_max = np.sign(value) # sign of euler error
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}') # print result

    # c. find bracket      
    lower = s_min # lower bound
    upper = s_max # upper bound

    it = 0 # iteration counter
    while it < maxiter: # iterate until maxiter reached
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t) 

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}') # print result

        # ii. check conditions
        valid = not np.isnan(value) # Checks if result is NaN
        correct_sign = np.sign(value)*sign_max < 0 # Checks if signs are different
        
        # iii. next step
        if valid and correct_sign: # found!
            s_min = s # lower bound
            s_max = upper # upper bound
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t): # calculate euler error
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s) # current period
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma) # left-hand side of the Euler equation
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma) # right-hand side of the Euler equation

    return LHS-RHS  # return euler error

def simulate_before_s(par,sim,t): # simulations done before finding s
    """ simulate forward """

    if t > 0:
        sim.K_lag[t] = sim.K[t-1] # capital stock at time t-1
        sim.B_lag[t] = sim.B[t-1] # government debt at time t-1
        sim.L_lag[t] = sim.L_lag[0]*(1+par.n)**t   # labor supply grows at rate n
        sim.k_lag[t] = sim.K_lag[t]/(sim.L_lag[t] * sim.A_lag[t]) # capital per effektive worker at time t-1
        sim.A_lag[t] = sim.A_lag[0]*(1+par.g)**t # productivity at time t-1

    # a. production and factor prices
    if par.production_function == 'ces': # ces production function

        # i. production
        sim.Y[t] = ( par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(sim.A_lag[t]*sim.L_lag[t])**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta) # return on capital
        sim.w[t] = (1-par.alpha)*(sim.A_lag[t]*sim.L_lag[t])**(-par.theta-1) * sim.Y[t]**(1.0+par.theta) # wage

    elif par.production_function == 'cobb-douglas': # cobb-douglas production function

        # i. production
        sim.Y[t] = sim.K_lag[t]**par.alpha * (sim.A_lag[t]*sim.L_lag[t])**(1-par.alpha) 

        # ii. factor prices
        sim.rk[t] = par.alpha * sim.K_lag[t]**(par.alpha-1) * (sim.A_lag[t]*sim.L_lag[t])**(1-par.alpha) # return on capital
        sim.w[t] = (1-par.alpha) * sim.K_lag[t]**(par.alpha) * (sim.A_lag[t]*sim.L_lag[t])**(-par.alpha) # wage

    else:

        raise NotImplementedError('unknown type of production function') # raise error if production function is not specified correctly

    # b. no-arbitrage and after-tax return
    sim.r[t] = sim.rk[t]-par.delta # after-depreciation return
    sim.rb[t] = sim.r[t] # same return on bonds
    sim.rt[t] = (1-par.tau_r)*sim.r[t] # after-tax return

    # c. consumption
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t]) # consumption of old

    # d. government
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t] *sim.L_lag[t]*sim.A_lag[t] # taxes

    if par.bal_budget == True: # if government run's a balanced budget
        sim.balanced_budget[:] = True 

    if sim.balanced_budget[t]: 
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t]

def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*sim.L_lag[t]*sim.A_lag[t]*(1.0-s) 

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t] # investment
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I # capital stock


def plotKLA(x,y,z):
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    # First subplot for 'k'
    axs[0].plot(x, label=r'$K_{t-1}$')
    axs[0].legend(frameon=True)
    axs[0].set_ylabel('k')

    # Second subplot for 'l'
    axs[1].plot(y, label=r"$L_{t-1}$")
    axs[1].legend(frameon=True)
    axs[1].set_ylabel('l')

    # Third subplot for 'A'
    axs[2].plot(z, label='$A_{t-1}$')
    axs[2].legend(frameon=True)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('A')

    plt.tight_layout()
    plt.show()


def plotSS(x,y):
    # Creating the figure and axes
    fig, ax = plt.subplots()

    # Plotting the data
    ax.plot(x, label='$k_{t-1}$')
    ax.axhline(y,ls='--',color='black',label='analytical steady state w/o taxes')

    # Adding labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of k and k_ss')

    # Adding legend
    ax.legend()

    # Displaying the plot
    plt.show()




def steady_state(v_alpha = 0.3,v_beta = 1.0/1.4,v_n = 0.02, v_g=0.03): 

    """Solve for steady_state level of per-worker capital
    given Log period utility and Cobb-Douglas production
    
    Args:
        
        v_alpha (float): value of output elasticity to factors change; int [0,1]
        v_beta (float): value of the consumption discount factor; > -1
        v_n (float): population growth rate, > -1
        v_g (float): technological growth rate, > -1

    """
    # a. household
    k = sm.symbols("k")
    n = sm.symbols("n") # population growth rate
    g = sm.symbols("g") # technological growth rate
    beta = sm.symbols("beta") # future consumption discount rate
    alpha = sm.symbols("alpha")


    ss = sm.Eq(k, ((1 - alpha)/((1 + n)*(1 + g)*(2 + (1 - beta)/beta)))*k**alpha) # Makes an equation for the OLG savings locus
    kss = sm.solve(ss,k)[0] # Isolates the expression for k

    print("the analytical steady state will be derived from the capital accumulation formula seen below:")
    display(ss) # displays the savings locus
    print("From this the steady state formula of capital, k_ss, is derived using sympy")

    display(kss) # displays the steady state formula for capital


    # b. characterizes the solution with value of beta, alpha and n
    solve_steady = sm.lambdify(args=(alpha,beta,n,g),expr=kss) # trasforms simpy to function
    ss_k = solve_steady(v_alpha,v_beta,v_n,v_g) #solves function for values of beta, n and aplha
    print(f'For alpha = {v_alpha};\nbeta = {v_beta};\nn = {v_n};\ng = {v_g};\nSteady state value of capital, k_ss =')
    display(ss_k)