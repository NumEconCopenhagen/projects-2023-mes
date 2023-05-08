from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


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

    



