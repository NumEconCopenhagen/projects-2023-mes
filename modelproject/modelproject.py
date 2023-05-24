from scipy.optimize import minimize
import numpy as np
import sympy as sm 
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import ipywidgets as widgets
from IPython.display import display


class RicardianModelClass:


    # Autarky: Plotting both PPFs in one Graph
    def ppf_plot(self, a1_d, a2_d, a1_g, a2_g):

        # A. points in German ppf
        x1 = [a1_g, 0]  
        y1 = [0, a2_g]  

        # B. points in Danish ppf
        x2 = [a1_d, 0]  
        y2 = [0, a2_d]  

        # C. Plotting the lines
        plt.plot(x1, y1, label='Germany')
        plt.plot(x2, y2, label='Denmark')

        # D. Adding labels and title
        plt.xlabel('Beer')
        plt.ylabel('Pharmaceuticals')
        plt.title('PPFs')
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # E. Adding a legend
        plt.legend()

        # F. Displaying the plot
        plt.show()


    # Autarky: Solver for the optimal consumption
    
    # A. Utility function solver
    def utility_func(self, c1, c2, alpha):
        return c1**alpha * c2**(1-alpha)

    # B. Utility function to calculate utility after optimal consumption is derived
    def utility(self, x, alpha):
        c1, c2 = x
        return c1**alpha * c2**(1-alpha)
    
    # C. PPF constraint
    def constraint_autarky(self, x, a1, a2):
        """Production Possibility Frontier constraint function"""
        c1, c2 = x
        return c2-a1+(a2/a1)*c1
    
    # D. Definition of the optimizer
    def optimize_autarky(self, alpha, a1, a2):

        # D.1 Initial guess for x
        x0 = [1, 1]

        # D.2 Bounds for x
        bounds = [(0, None), (0, None)]

        # D.3 Constraint definition
        cons = ({'type': 'eq', 'fun': self.constraint_autarky, 'args': (a1, a2)})

        # D.4 Optimization function
        res = minimize(lambda x: -self.utility(x, alpha), x0, bounds=bounds, constraints=cons)

        return res.x * (a2/a1)


    # Autarky: Plot PPF individually
    # A. Definition of plot
    def ppf_plot_individual(self, a1, a2, optimal_c1, optimal_c2):
        
        # B. Generate x1 values
        y1 = np.linspace(0, 10)

        # C. Calculate corresponding y-values according to PPF
        y2 = a2 - (a2/a1)*y1

        # D. Plot the PPF
        plt.plot(y1, y2)

        # E. Adding labels and title
        plt.xlabel('Beer')
        plt.ylabel('Pharmaceuticals')
        plt.title('PPFs')
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # F. Adding optimal consumption bundle
        plt.scatter(optimal_c1, optimal_c2, color='red')

        # G. Displaying the plot
        plt.show()



    # Plots both PPFs with optimal points in autarky and trade
    def ppf_plot_trade(self, a1_d, a2_d, a1_g, a2_g):
        # points in German ppf
        x1 = [a1_g, 0]  
        y1 = [0, a2_g]  

        # points in Danish ppf
        x2 = [a1_d, 0]  
        y2 = [0, a2_d]  

        # Plotting the lines
        plt.plot(x1, y1, label='Germany')
        plt.plot(x2, y2, label='Denmark')

        # Adding labels and title
        plt.xlabel('Beer')
        plt.ylabel('Pharmaceuticals')
        plt.title('PPFs')
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # add autarky equlibrium points Germany
        plt.scatter(5, 1.5, color='black')

        # add autarky equlibrium points Denmarl
        plt.scatter(2, 4, color='red')

        # add the trade equlibirum 
        plt.scatter(5, 4, color='green')

        # Adding a legend
        plt.legend()

        # Displaying the plot
        plt.show()


    # Solver for the optimal consumption in autarky 
    
    # utility function for the solver (maximization)
    def utility_func(self, c1, c2, alpha):
        return c1**alpha * c2**(1-alpha)

    # utility function to calculate utility after optimal consumption is derived
    def utility(self, x, alpha):
        c1, c2 = x
        return c1**alpha * c2**(1-alpha)
    
    # PPF constraint
    def constraint_autarky(self, x, a1, a2):
        """Production Possibility Frontier constraint function"""
        c1, c2 = x
        return c2-a1+(a2/a1)*c1
    
    # Definition of the optimizer
    def optimize_autarky(self, alpha, a1, a2):

        # Initial guess for x
        x0 = [1, 1]

        # Bounds for x
        bounds = [(0, None), (0, None)]

        # Constraint definition
        cons = ({'type': 'eq', 'fun': self.constraint_autarky, 'args': (a1, a2)})

        # Optimization function
        res = minimize(lambda x: -self.utility(x, alpha), x0, bounds=bounds, constraints=cons)

        return res.x * (a2/a1)