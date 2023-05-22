# Include all the required packages
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from matplotlib import cm

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        global par
        global sol
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences (parameters)
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production (parameters)
        par.alpha = 0.5
        par.sigma = 1

        # d. wages for male and female
        par.wM = 1.0 
        par.wF = 1.0 
        par.wF_vec = np.linspace(0.8,1.2,5) # varying female wages

        # e. targets for beta
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. unpack the (empty) solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility function """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**(par.alpha)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        # e. Total utility
        return utility - disutility
    
    def plot_line(self, a, b):
        # D. Create a Figure
        # D.1 Plot the optimal ratios
        plt.plot(a, b, 'o--', label='Data Points')

        # D.2 Create heading, axis labels and legend
        plt.title('Exploring the Link Between Household Work and Gender Wage Disparity', fontsize=16)
        plt.xlabel('log(wF/wM)', fontsize=12)
        plt.ylabel('log(HF/HM)', fontsize=12)
        plt.legend()

        # D.3 Remove black box around the plot
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # D.4 Add data point values to the plot
        for i, (x, y) in enumerate(zip(a, b)):
            wf_value = par.wF_vec[i]
            plt.text(x, y, f'wF = {wf_value}', ha='left', va='bottom')

        # D.5 Show the plot
        plt.show()
    

    # Question 2 - discrete solver
    def solve_discrete(self,do_print=False):
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
        if do_print:
            for k,v in opt._dict_.items():
                print(f'{k} = {v:6.4f}')

        return opt

       
    # Question 3 - continuous solver
    # create the objective function for the solver
    def objective(self, x):
        """ objective function to minimize """
        LM, LF, HM, HF = x
        return -self.calc_utility(LM, HM, LF, HF) # the minus is necessary to maximize within the minimize solver

    def solve_continous(self):
        """ solve for optimal values of LM, LF, HM, HF """
        par = self.par
        sol = self.sol

        # initial guess of results
        x0 = [10 ,10, 10, 10]

        # predefine the bounds for the solutions
        bounds = ((0, 24), (0, 24), (0, 24), (0, 24))

        # creating the optimization function
        res = optimize.minimize(self.objective, x0, bounds=bounds, method = 'Nelder-Mead' )

        # store the optimal results
        sol.LM = res.x[0] 
        sol.LF = res.x[1] 
        sol.HM = res.x[2] 
        sol.HF = res.x[3]


    # Question 4
    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """


        par = self.par
        sol = self.sol


        #loop over the vector of female wage and change the value of wF to whereever we are in the vector
        for i, wF in enumerate(par.wF_vec):
            self.par.wF = wF
           
            self.solve_continous()


            #store the resulting values
            sol.LM_vec[i] = sol.LM
            sol.HM_vec[i] = sol.HM
            sol.LF_vec[i] = sol.LF
            sol.HF_vec[i] = sol.HF


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        #placeholder lists used later for graphic interpretation
        global placeholder_a
        global placeholder_s
        global placeholder_b
        placeholder_a = []
        placeholder_s = []
        placeholder_b = []

        # Objective Function to optimize alpha and sigma
        def objective_function(x):
            alpha, sigma = x
            par.alpha = alpha
            par.sigma = sigma
            placeholder_HF = []
            placeholder_HM = []
            for wF in par.wF_vec:
                par.wF = wF
                self.solve_continous() 
                placeholder_HF.append(sol.HF)
                placeholder_HM.append(sol.HM)
                placeholder_a.append(alpha)
                placeholder_s.append(sigma)
                placeholder_b.append(sol.beta0)
            k = np.array(placeholder_HF)
            l = np.array(placeholder_HM)
            sol.HF_vec = k
            sol.HM_vec = l
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        # Set narrow bounds for alpha and sigma
        bounds = ((0.9, 1), (0, 0.1))

        # Run the optimization
        res = optimize.minimize(objective_function, x0=[0, 0], bounds=bounds, options={'eps':0.0001})

        #store results
        par.alpha = res.x[0]
        par.sigma = res.x[1]


    # Question 5 - Continous Solver for the Extension
    def objective_extension(self, x):
        """ objective function to minimize """
        LM, LF, HM, HF = x
        return -self.calc_utility(LM, HM, LF, HF) # the minus is necessary to maximize within the minimize solver

    def solve_continous_extension(self, bound_female):
        """ solve for optimal values of LM, LF, HM, HF """
        par = self.par
        sol = self.sol

        # initial guess of results
        x0 = [10 ,10, 10, 10]

        # predefine the bounds for the solutions
        bounds = ((0, 24), (0, 24), (0, 24), (0, bound_female))

        # creating the optimization function
        res_extension = optimize.minimize(self.objective_extension, x0, bounds=bounds, method = 'Nelder-Mead' )

        # store the optimal results
        sol.LM = res_extension.x[0] 
        sol.LF = res_extension.x[1] 
        sol.HM = res_extension.x[2] 
        sol.HF = res_extension.x[3] 