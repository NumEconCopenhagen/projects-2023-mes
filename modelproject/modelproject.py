from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class RicardianModelClass:

    def ricardian_ppf(a1, a2):
            """
            Returns a production possibility set (PPF) in the setting of a Ricardian model of international trade.

            Arguments:
            - a1: float, productivity of the first sector.
            - a2: float, productivity of the second sector.

            Returns:
            - ppf: function of the form y2 = a2*L - (a2/a1)*y1, where y1 and y2 are the output levels of the first and second sectors, respectively, and L is the fixed amount of labor available in the economy.
            """
            def ppf(y1):
                """
                Computes the maximum output level of the second sector (y2) given a certain output level of the first sector (y1), subject to the constraint that all available labor is used.

                Arguments:
                - y1: float, output level of the first sector.

                Returns:
                - y2: float, maximum output level of the second sector given y1.
                """
                L = 1.0  # fixed amount of labor
                y2 = a2*L - (a2/a1)*y1
                return y2 if y2 >= 0 else 0.0  # ensure non-negative output level of the second sector

            return np.vectorize(ppf)  # return a vectorized version of ppf

    
    #def plot_ppf(self):
         
        #ppf = ricardian_ppf(a1=2.0, a2=1.0)

        #Plot the PPF curve
        #y1_values = np.linspace(0, 2, num=101)
        #y2_values = ppf(y1_values)
        #plt.plot(y1_values, y2_values)
        #plt.xlabel('Output of sector 1')
        #plt.ylabel('Output of sector 2')
        #plt.title('Production Possibility Frontier')
        #plt.show()
