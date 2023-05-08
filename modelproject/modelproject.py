from scipy import optimize
import numpy as np
import sympy as sm 
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve


class RicardianModelClass:
    

    def __init__(self):

        #default inputs of productivity for Denmark and Germany
        a1_d = 4
        a2_d = 8
        a1_g = 10
        a2_g = 3

        #default labor shares in Denmark and Germany
        L1_d = 0.5
        L2_d = 0.5
        L_d = 1
        L1_g = 0.7
        L2_g = 0.3
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



    def ricardian_ppf_Denmark(self, a1_d, a2_d, L_d):
        #production possibility set for Denmark
        def ppf_Denmark(y1):
            #calculates maximum y2 for a given y1, a1, a2, labor fixed at 1
            y2 = a2_d*L_d - (a2_d/a1_d)*y1
            return y2 if y2 >= 0 else 0.0  # ensure non-negative output level of the second sector

        return np.vectorize(ppf_Denmark)  # return a vectorized version of ppf
    
    def ricardian_ppf_Germany(self, a1_g, a2_g, L_d):
        #production possibility set for Germany
        def ppf_Germany(y1):
            #calculates maximum y2 for a given y1, a1, a2, labor fixed at 1
            y2 = a2_g*L_d - (a2_g/a1_g)*y1
            return y2 if y2 >= 0 else 0.0  # ensure non-negative output level of the second sector

        return np.vectorize(ppf_Germany)  # return a vectorized version of ppf

    def plot_ppf(self, a1_d, a2_d, a1_g, a2_g, L_d, L_g):

        ppf_d = self.ricardian_ppf_Denmark(a1_d, a2_d, L_d)
        ppf_g = self.ricardian_ppf_Germany(a1_g, a2_g, L_g)

        # Plot the PPF curve
        y1_values = np.linspace(-2, 2, num=101)
        y2_values_d = ppf_d(y1_values)
        y2_values_g = ppf_g(y1_values)
        plt.plot(y1_values, y2_values_d, label = "Denmark")
        plt.plot(y1_values, y2_values_g, label = "Germany")
        plt.xlabel('Output of sector 1')
        plt.ylabel('Output of sector 2')
        plt.title('Production Possibility Frontier')
        plt.legend()
        plt.show()


    #production function

    def production_function(self, a, L):
        return a * L
    
    #utility function

    def utility(self, c1, c2, epsilon):
        numerator = epsilon - 1
        denominator = epsilon
        term1 = c1 ** (numerator / denominator)
        term2 = c2 ** (numerator / denominator)
        inner_expression = term1 + term2
        utility_value = inner_expression ** (numerator / denominator)
        return utility_value
    

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
    

    def find_equilibrium_price(self, a, L, w, y1, y2, epsilon):
        p = symbols('p')
        
        c1 = self.production_function(a[0], L[0])
        c2 = self.production_function(a[1], L[1])
        
        demand_c1 = self.utility(c1, c2, epsilon)
        demand_c2 = self.utility(c2, c1, epsilon)
        
        supply_c1 = self.income_cons(p, c1, c2)
        supply_c2 = self.income_cons(p, c2, c1)
        
        eq1 = Eq(demand_c1, supply_c1)
        eq2 = Eq(demand_c2, supply_c2)
        
        equilibrium_price = solve((eq1, eq2), (p,))
        
        return equilibrium_price[0]


    
    
