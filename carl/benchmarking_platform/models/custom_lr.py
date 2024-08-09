import numpy as np

class custom_LR: 

    def predict(self, X_test):
        # Assuming X_test is a DataFrame
        alpha = X_test['polarizability'].values
        alcohol = X_test['alcohol'].values
        carbonyl = X_test['carbonyl'].values
        amine = X_test['amine'].values
        carboxylic_acid = X_test['carboxylic_acid'].values
        nitro = X_test['nitro'].values
        nitrile = X_test['nitrile'].values

        # Initialize log_vps array with -100 for all indices
        log_vps = np.full(len(alpha), -20)

        # Identify indices where no values are NaN
        valid_indices = ~np.isnan(alpha) & ~np.isnan(alcohol) & ~np.isnan(carbonyl) & ~np.isnan(amine) & \
                        ~np.isnan(carboxylic_acid) & ~np.isnan(nitro) & ~np.isnan(nitrile)

        # Apply the linear regression equation only to valid rows
        log_vps[valid_indices] = (
            -0.432 * alpha[valid_indices]
            - 1.382 * alcohol[valid_indices]
            - 0.482 * carbonyl[valid_indices]
            - 0.416 * amine[valid_indices]
            - 2.197 * carboxylic_acid[valid_indices]
            - 1.382 * nitro[valid_indices]
            - 1.101 * nitrile[valid_indices]
            + 4.610
        )
        
        return log_vps
