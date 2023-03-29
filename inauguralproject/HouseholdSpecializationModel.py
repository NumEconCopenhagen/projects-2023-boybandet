
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec =  np.linspace(0.8,1.2,5)
        
        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan


    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production for different values of sigma
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        
        elif par.sigma == 0:
            H = np.minimum(HM, HF)
        
        else:
            HM = np.fmax(HM, 1e-07)
            HF = np.fmax(HF, 1e-07)
            H = ((1-par.alpha) * HM**((par.sigma - 1) / par.sigma) + par.alpha * HF**((par.sigma - 1) / par.sigma))**(par.sigma/(par.sigma - 1))

    
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self, do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        #if do_print == True:
         #   for k,v in opt.__dict__.items():
         #       print(f'{k} = {v:6.4f}')

        return opt


    def plot_ratios_alpha(self):
        """ plots the ratio for different alphas """
        par = self.par
        alpha_vec = (0.25, 0.50 , 0.75) # Values for alpha

        alpha_ratios = [] #initialize empty list
        # a. loop over the different values for alpha
        for par.alpha in alpha_vec:
            result = self.solve_discrete(par.alpha)
            ratio = result.HF / result.HM 
            alpha_ratios.append(ratio)
        
        # b. plot figure for alpha
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(alpha_vec, alpha_ratios)
        ax.set_ylabel("Female work from home relative to male")
        ax.set_xlabel("Alpha")
        ax.set_title("Ratio between working from home dependent on alpha")
        ax.set_ylim()
        ax.set_xlim([0.2,0.8])

        # c. Returns alpha to its original
        par.alpha = 0.5

        
    def plot_ratios_sigma(self):
        """ plots the ratio for different alphas """
        par = self.par
        sigma_vec = (0.5, 1 , 1.5 )
        


        sigma_ratios = [] #initialize empty list
        # a. loop over the different values for sigma
        for par.sigma in sigma_vec:
            result = self.solve_discrete(par.sigma)
            ratio = result.HF / result.HM 
            sigma_ratios.append(ratio)
        
        # b. plot figure for sigma
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(sigma_vec, sigma_ratios)
        ax.set_ylabel("Female work from home relative to male")
        ax.set_xlabel("Sigma")
        ax.set_title("Ratio between working from home dependent on sigma")
        ax.set_ylim()
        ax.set_xlim([0.4,1.6])

        # c. Returns sigma to its original
        par.sigma = 1.0
    
    
    def plot_logratios_discrete(self):
        """ plots the ratio for different wFs """
        par = self.par
     

        log_workratios = [] #initialize empty list
        log_wageratios = [] #initialize empty list

        # a. loop over the different values for wF
        for par.wF in par.wF_vec:
            result = self.solve_discrete(par.wF)
            log_workratiosCalc = np.log(result.HF / result.HM)
            log_workratios.append(log_workratiosCalc)
            log_wageratiosCalc = np.log(par.wF / par.wM)
            log_wageratios.append(log_wageratiosCalc)
        
        # b. plot figure for different values of wF
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(log_wageratios, log_workratios )
        ax.set_ylabel("log_workratios")
        ax.set_xlabel("log_wageratios")
        ax.set_title("Log workratios and log wage ratios when varying female wages")
        ax.set_ylim()
        ax.set_xlim([-0.25,0.25])


    def solve_cont(self,do_print=False):
        """ solve model continously """       
        par = self.par
        sol = self.sol 

        opt = SimpleNamespace()

        # The objective is defined
        objective = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])
        
        # We define bounds
        bounds = ((1e-8,24-1e-8), (1e-8,24-1e-8), (1e-8,24-1e-8), (1e-8,24-1e-8))

        initial_guess = [24/par.wM/2, 24/par.wM/2, 24/par.wF/2, 24/par.wF/2] # initial guess is made 

        sol_case = optimize.minimize(
            objective, initial_guess, method="Nelder-Mead", bounds = bounds)
        
        # unpack solution
        opt.LM = sol_case.x[0]
        opt.HM = sol_case.x[1]
        opt.LF = sol_case.x[2]
        opt.HF = sol_case.x[3]
        
        return opt

    #Define how the continuous solution changes with wF
    def changes_wF(self, doprint = True):
        """ plots the the log relations for different values of wF"""
        par = self.par
        sol = self.sol

        log_workratios = [] #initialize empty list
        log_wageratios = [] #initialize empty list

        # a. loop over the different values for wF
        for par.wF in par.wF_vec:
            result = self.solve_cont(par.wF)
            log_workratiosCalc = np.log(result.HF / result.HM)
            log_workratios.append(log_workratiosCalc)
            log_wageratiosCalc = np.log(par.wF / par.wM)
            log_wageratios.append(log_wageratiosCalc)
        
        if doprint:
            #b. plot figure for different values of wF
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(log_wageratios, log_workratios )
            ax.set_ylabel("log of HF/HM")
            ax.set_xlabel("log of wF/wM")
            ax.set_title("Log work ratios and log wage ratios when varying female wages")
            ax.set_ylim()
            ax.set_xlim()

        
    
    def solve_wF_vec(self):  #MIGHT BE DELETED!
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        for i, wF in enumerate(par.wF_vec):
            #Set wF for this iteration
            par.wF = wF
            #Solve for optimal choices
            opt = self.solve_cont()
            #Store results in solution arrays
            sol.HM_vec[i] = opt.HM
            sol.HF_vec[i] = opt.HF
            
        pass


    
    def run_regression(self): 
        """ run regression """

        par = self.par
        sol = self.sol

        # Taking logs of used vectors
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)

        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        pass


    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective(x):

            par = self.par
            sol = self.sol

            par.alpha, par.sigma = x 

            target_beta0 = par.beta0_target
            target_beta1 = par.beta1_target
            
            self.solve_wF_vec()
            self.run_regression()

            objective_loss = (target_beta0 - sol.beta0)**2 + (target_beta1 - sol.beta1)**2

            return objective_loss
        
        bounds = ((1e-8,1), (1e-8,5))

        initial_guess = [0.5, 1]
        
        result = optimize.minimize(objective, initial_guess, bounds = bounds, method = "Nelder-Mead")

        alpha = result.x[0]
        sigma = result.x[1]

        print("Optimal values for alpha and sigma are " +str(alpha)  + " and "+ str(sigma) +". Down below a plot that illustrates how the model fits the data is made")

        #Creating 3D-plot:
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection = "3d")

        alpha_grid, sigma_grid = np.meshgrid(np.linspace(alpha-0.2, alpha+0.2, 10), np.linspace(sigma-0.2, sigma+0.2, 10))
        aladdin = [alpha_grid, sigma_grid]
        errors = np.zeros_like(alpha_grid)

        for i in range(len(alpha_grid)):
            for j in range(len(sigma_grid)):
                errors[i,j] = objective([alpha_grid[i,j], sigma_grid[i,j]])

        ax.plot_surface(alpha_grid, sigma_grid, errors, cmap="viridis")
        ax.set_xlabel("alpha")
        ax.set_ylabel("sigma")
        ax.set_zlabel("errors")
        ax.set_title("How the std. error varies with different values for alpha and sigma")
        plt.show
