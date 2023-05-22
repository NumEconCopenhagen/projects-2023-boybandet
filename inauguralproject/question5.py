from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Model2:
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
        par.nu_new = 0.0005

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


    def calc_utility_2(self,LM,HM,LF,HF):
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
            disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_) + par.nu_new * HM



            return utility - disutility

    def solve_cont2(self,do_print=False):
            """ solve model continously """       
            par = self.par
            sol = self.sol 

            opt = SimpleNamespace()

            # The objective is defined
            objective = lambda x: -self.calc_utility_2(x[0], x[1], x[2], x[3])

            # We define bounds
            bounds = ((0,24), (0,24), (0,24), (0,24))

            initial_guess = [24/par.wM/2, 24/par.wM/2, 24/par.wF/2, 24/par.wF/2] # initial guess is made 

            # We make constraints to prevent workrates > 24
            constraints = ({"type": "ineq", "fun": lambda x: 24 - (x[0] + x[1])}, {"type": "ineq", "fun": lambda x: 24 - (x[2] + x[3])})

            sol_case = optimize.minimize(
                objective, initial_guess, method="Nelder-Mead", bounds = bounds, constraints = constraints)

            # unpack solution
            opt.LM = sol_case.x[0]
            opt.HM = sol_case.x[1]
            opt.LF = sol_case.x[2]
            opt.HF = sol_case.x[3]

            return opt
    
    def solve_wF_vec2(self):  #MIGHT BE DELETED!
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        for i, wF in enumerate(par.wF_vec):
            #Set wF for this iteration
            par.wF = wF
            #Solve for optimal choices
            opt = self.solve_cont2()
            #Store results in solution arrays
            sol.HM_vec[i] = opt.HM
            sol.HF_vec[i] = opt.HF


    def run_regression2(self): 
        """ run regression """

        par = self.par
        sol = self.sol

        # Taking logs of used vectors
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)

        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0, sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def estimate2(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective2(x):

            par = self.par
            sol = self.sol

            par.alpha, par.sigma = x 

            target_beta0 = par.beta0_target
            target_beta1 = par.beta1_target
            
            self.solve_wF_vec2()
        
            self.run_regression2()

            objective_loss = (target_beta0 - sol.beta0)**2 + (target_beta1 - sol.beta1)**2

            return objective_loss
        
        bounds = ((0.001,0.99), (0.001,0.99))

        initial_guess = [0.8, 0.05]
        
        result = optimize.minimize(objective2, initial_guess, bounds = bounds, method = "Nelder-Mead", tol= 1e-09)

        alpha = result.x[0]
        sigma = result.x[1]

        print("Optimal values for alpha and sigma are " + str(alpha)  + " and "+ str(sigma) +". Down below a plot that illustrates how the model fits the data is made")

        #Creating 3D-plot:
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection = "3d")

        alpha_grid, sigma_grid = np.meshgrid(np.linspace(0.9, 0.99, 10), np.linspace(0.07, 0.15, 10))
       
        errors = np.zeros_like(alpha_grid)

        for i in range(len(alpha_grid)):
            for j in range(len(sigma_grid)):
                errors[i,j] = objective2([alpha_grid[i,j], sigma_grid[i,j]])

        ax.plot_surface(sigma_grid, alpha_grid, errors, cmap="viridis")
        ax.set_ylabel("alpha")
        ax.set_xlabel("sigma")
        ax.set_zlabel("errors")
        ax.set_title("How the std. error varies with different values for alpha and sigma")
        plt.show
