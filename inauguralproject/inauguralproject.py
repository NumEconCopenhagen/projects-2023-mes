# Import all required packages
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from matplotlib import cm


# Create HouseholdSpecializationModelClass
class HouseholdSpecializationModelClass:


    # A. Define all parameters of Inaugural Model

    def __init__(self):
        """ setup model """

        # A1. create namespaces
        global par
        global sol
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # A2. Set the preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # A3. Set the household production
        par.alpha = 0.5 # measures the importance of HF relative to HM
        par.sigma = 1 # measures the elasticity of substitution

        # A4. Set wages for male and female
        par.wM = 1.0 
        par.wF = 1.0 
        par.wF_vec = np.linspace(0.8,1.2,5) # varying female wages (instead of fixed wF)

        # A5. Set targets for beta
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # A6. Create empty vectors, where the model solutions are appended to
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        # A7. Create empty values for the optimal betas
        sol.beta0 = np.nan
        sol.beta1 = np.nan


    # B. Define the model (Insert the utility function and the constraints of Inaugural Model)

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility function """

        # B1. Refer to the initialised parameters 
        par = self.par
        sol = self.sol

        # B2. Constraint: Consumption of Market Goods
        C = par.wM*LM + par.wF*LF

        # B3. Constraint: Consumption of Home Production
        if par.sigma == 0:
            H = min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**(par.alpha)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # B4. Constraint: Total Consumption Utility
        Q = C**par.omega*H**(1-par.omega)

        # B5. First Part Utility Function: Utility of Consumption
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # B6. Second Part Utility Function: Disutility of Work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        # B7. Total utility: Combining both parts of Utility Function
        return utility - disutility
    

    # C. Function for Plotting log(HF/HM) against log(wF/wM) - Used for Question 2 and 3

    def plot_line(self, a, b):
        
        # C.1 Insert the variables (ratios log(HF/HM) and log(wF/wM)) into the graph
        plt.plot(a, b, 'o--', label='Data Points')

        # C.2 Create heading, axis labels and legend for the graph
        plt.title('Exploring the Relationship between log(HF/HM) and log(wF/wM)', fontsize=16)
        plt.xlabel('log(wF/wM)', fontsize=12)
        plt.ylabel('log(HF/HM)', fontsize=12)
        plt.legend()

        # C.3 Remove black box around the plot
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # C.4 Add data point values for wF to the plot
        for i, (x, y) in enumerate(zip(a, b)):
            wf_value = par.wF_vec[i]
            plt.text(x, y, f'wF = {wf_value}', ha='left', va='bottom')

        # C.5 Show the plot
        plt.show()
    

    # D. Creating the Solver for the discrete Solution - Used for Question 2 

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        # D.1 Refer to the predefined parameters
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # D.2 Create the possible (discrete) solutions for LM, HM, LF and HF
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x)
    
        # D.3 Reduce multidimensional Arrays into one dimensional array
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # D.4 Calculate utility (Refer to the predefined utility function)
        u = self.calc_utility(LM,HM,LF,HF)

        # D.5 Include a penalty: If constraints are broken, then set utility to -infinity
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # D.6 Define j as solution that maximizes u
        j = np.argmax(u)
        
        # D.7 Append the optimal solution for j to the individual optimal values LM, HM, LF, HF
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # D.8 Return the optimal solutions
        return opt

       
    # E. Creating the Solver for the continous Solution - Used for Question 3
    
    # E.1 Create the Objective function (Utility Function)
    def objective(self, x):
        """ objective function to minimize """
        LM, LF, HM, HF = x
        return -self.calc_utility(LM, HM, LF, HF) # minus necessary to maximize

    # E.2 Define the continous solver
    def solve_continous(self):
        """ solve for optimal values of LM, LF, HM, HF """

        # E.3 Refer to the predefined parameters
        par = self.par
        sol = self.sol

        # E.4 Initial guess of results
        x0 = [10 ,10, 10, 10]

        # E.5 Bounds for the solutions of LM, HM, LF, HF
        bounds = ((0, 24), (0, 24), (0, 24), (0, 24))

        # E.6 Creating optimization function
        res = optimize.minimize(self.objective, x0, bounds=bounds, method = 'Nelder-Mead' )

        # E.7 Store the optimal results
        sol.LM = res.x[0] 
        sol.LF = res.x[1] 
        sol.HM = res.x[2] 
        sol.HF = res.x[3]


    # F. Creating the Solver, Regression and Estimation - Used for Question 4

    # F.1 Define the solver for wF
    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        # F.2 Refer to the predefined parameters
        par = self.par
        sol = self.sol

        # F.3 Loop over the vector of female wage
        for i, wF in enumerate(par.wF_vec):
            self.par.wF = wF
           
            # F.4 Solve for the different wF
            self.solve_continous()

            # F.5 Store the resulting values
            sol.LM_vec[i] = sol.LM
            sol.HM_vec[i] = sol.HM
            sol.LF_vec[i] = sol.LF
            sol.HF_vec[i] = sol.HF

    # F.6 Run the Regression
    def run_regression(self):
        """ run regression """

        # F.7 Refer to the predefined parameters
        par = self.par
        sol = self.sol

        # F.8 Store the solutions from  the solver as variables for the regression
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T

        # F.9 Estimate beta0 and beta1
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    # F.10 Estimate alpha and sigma
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        # F.11 Refer to the predefined parameters
        par = self.par
        sol = self.sol

        # F.12 Placeholder lists used later for graphic interpretation
        global placeholder_a
        global placeholder_s
        global placeholder_b
        placeholder_a = []
        placeholder_s = []
        placeholder_b = []

        # F.13 Set Objective Function to optimize alpha and sigma
        def objective_function(x):
            alpha, sigma = x
            par.alpha = alpha
            par.sigma = sigma
            placeholder_HF = []
            placeholder_HM = []

            # F.14 Solve for varying wF
            for wF in par.wF_vec:
                par.wF = wF
                self.solve_continous() 

                # F.15 Append the solutions to the placeholders
                placeholder_HF.append(sol.HF)
                placeholder_HM.append(sol.HM)
                placeholder_a.append(alpha)
                placeholder_s.append(sigma)
                placeholder_b.append(sol.beta0)

            # F.16 Define the placeholders as arrays named k and l 
            k = np.array(placeholder_HF)
            l = np.array(placeholder_HM)

            # F.17 Set it to optimal solution for HF and HM
            sol.HF_vec = k
            sol.HM_vec = l
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        # F.18 Set narrow bounds for alpha and sigma
        bounds = ((0.9, 1), (0, 0.1))

        # F.19 Run the optimization
        res = optimize.minimize(objective_function, x0=[0, 0], bounds=bounds, options={'eps':0.0001})

        # F.20 Store results
        par.alpha = res.x[0]
        par.sigma = res.x[1]


    # G. Creating the Solver for the Model Extension - Used for Question 5

    # G.1 Set the objective function (maximize utility)
    def objective_extension(self, x):
        """ objective function to minimize """
        LM, LF, HM, HF = x
        return -self.calc_utility(LM, HM, LF, HF) # minus necessary to maximize

    # G.2 Solve the model extension
    def solve_continous_extension(self, bound_female):
        """ solve for optimal values of LM, LF, HM, HF """

        # G.3 Refer to the predefined parameters
        par = self.par
        sol = self.sol

        # G.4 Initial guess of results
        x0 = [10 ,10, 10, 10]

        # G.5 Predefine the bounds for the solutions
        bounds = ((0, 24), (0, 24), (0, 24), (0, bound_female)) # Here, we restrict that maximum bound of females

        # G.6 Creating the optimization function
        res_extension = optimize.minimize(self.objective_extension, x0, bounds=bounds, method = 'Nelder-Mead' )

        # G.7 Store the optimal results
        sol.LM = res_extension.x[0] 
        sol.LF = res_extension.x[1] 
        sol.HM = res_extension.x[2] 
        sol.HF = res_extension.x[3] 