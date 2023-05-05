from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class RicardianModelClass:
    

    def __init__(self):
        a1_d = 4
        a2_d = 8
        a1_g = 10
        a2_g = 3

    def ricardian_ppf_Denmark(self, a1_d, a2_d):
        """
        Returns a production possibility set (PPF) in the setting of a Ricardian model of international trade.

        Returns:
        - ppf: function of the form y2 = a2*L - (a2/a1)*y1, where y1 and y2 are the output levels of the first and second sectors, respectively, and L is the fixed amount of labor available in the economy.
        """
        def ppf_Denmark(y1):
            """
            Computes the maximum output level of the second sector (y2) given a certain output level of the first sector (y1), subject to the constraint that all available labor is used.

            Arguments:
            - y1: float, output level of the first sector.

            Returns:
            - y2: float, maximum output level of the second sector given y1.
            """
            L = 1.0  # fixed amount of labor
            y2 = a2_d*L - (a2_d/a1_d)*y1
            return y2 if y2 >= 0 else 0.0  # ensure non-negative output level of the second sector

        return np.vectorize(ppf_Denmark)  # return a vectorized version of ppf
    
    def ricardian_ppf_Germany(self, a1_g, a2_g):
        """
        Returns a production possibility set (PPF) in the setting of a Ricardian model of international trade.

        Returns:
        - ppf: function of the form y2 = a2*L - (a2/a1)*y1, where y1 and y2 are the output levels of the first and second sectors, respectively, and L is the fixed amount of labor available in the economy.
        """
        def ppf_Germany(y1):
            """
            Computes the maximum output level of the second sector (y2) given a certain output level of the first sector (y1), subject to the constraint that all available labor is used.

            Arguments:
            - y1: float, output level of the first sector.

            Returns:
            - y2: float, maximum output level of the second sector given y1.
            """
            L = 1.0  # fixed amount of labor
            y2 = a2_g*L - (a2_g/a1_g)*y1
            return y2 if y2 >= 0 else 0.0  # ensure non-negative output level of the second sector

        return np.vectorize(ppf_Germany)  # return a vectorized version of ppf

    def plot_ppf(self, a1_d, a2_d, a1_g, a2_g):

        ppf_d = self.ricardian_ppf_Denmark(a1_d, a2_d)
        ppf_g = self.ricardian_ppf_Germany(a1_g, a2_g)

        # Plot the PPF curve
        y1_values = np.linspace(-2, 2, num=101)
        y2_values_d = ppf_d(y1_values)
        y2_values_g = ppf_g(y1_values)
        plt.plot(y1_values, y2_values_d)
        plt.plot(y1_values, y2_values_g)
        plt.xlabel('Output of sector 1')
        plt.ylabel('Output of sector 2')
        plt.title('Production Possibility Frontier')
        plt.show()


