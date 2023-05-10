from scipy.optimize import minimize
import numpy as np
import sympy as sm 
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import ipywidgets as widgets
from IPython.display import display


class RicardianModelClass:
    

    def __init__(self):

        #default inputs of productivity for Denmark and Germany
        a1_d = 4
        a2_d = 8
        a1_g = 10
        a2_g = 3

        #default labor shares in Denmark and Germany
        L_d = 1
        L_g = 1

        #default wages (guesses)
        w_d_initial = 1
        w_g_initial = 1
        w_d = 1
        w_g = 1

        #default consumption guesses
        c1_d = 1
        c2_d = 1
        c1_g = 1
        c2_g = 1

    
    def ricardian_ppf(self, a1, a2, L):
        def ppf(y1):
            y2 = a2*L - (a2/a1)*y1
            return y2
        return np.vectorize(ppf)


    def ppf_plot(self, a1_d, a2_d, a1_g, a2_g):
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

        # Adding a legend
        plt.legend()

        # Displaying the plot
        plt.show()


    #production function

    def production_function(self, a, L):
        return a * L
    
    #utility function

    def utility(self, x, alpha):
        c1, c2 = x
        return c1**alpha * c2**(1-alpha)
    
    #ppf constraint
    
    def constraint(self, x, a1, a2):
        """Production Possibility Frontier constraint function"""
        c1, c2 = x
        return c2-a1+(a2/a1)*c1
    
    def optimize_autarky(self, alpha, a1, a2):
        # Initial guess for x
        x0 = [1, 1]

        # Bounds for x
        bounds = [(0, None), (0, None)]

        # Constraint definition
        cons = ({'type': 'eq', 'fun': self.constraint, 'args': (a1, a2)})

        # Optimization function
        res = minimize(lambda x: -self.utility(x, alpha), x0, bounds=bounds, constraints=cons)

        return res.x * (a2/a1)
    

    #income contraints
    
    def income(self, w, L):
        return w * L

    def income_alt(self, p, y1, y2):
        return p * y1 + y2
    
    def income_cons(self, p, c1, c2):
        return p * c1 + c2

    

    def lagrangian(self, U, I):
        # Define symbols
        c1, c2, p, I, epsilon = sm.symbols('c1 c2 p I epsilon', positive=True)
        # Use the utility function
        U = self.utility(c1, c2, epsilon)
        # Use budeget constraint
        I = self.income_cons(p, c1, c2)
        # Define the Lagrangian function
        L = U + sm.Symbol('lambda')*I
        # Calculate the first-order conditions
        foc_c1 = sm.diff(L, c1)
        foc_c2 = sm.diff(L, c2)
        foc_lambda = sm.diff(L, sm.Symbol('lambda'))

        # Solve the first-order conditions for c1 and lambda
        sol = sm.solve([foc_c1, foc_lambda, foc_c2], [c1, c2, sm.Symbol('lambda')])
         # Extract the optimal values of c1 and lambda from the solution
        c1_opt = sol[c1]
        c2_opt = sol[c2]
        # Solve for the optimal price
        p_opt = sm.solve(sm.Eq(I, p*(c1_opt[c1] + c2_opt[c2])), p)
        # Return the optimal price and the optimal value of c1
        return p_opt, c1_opt, c2_opt
    

    


    
    
