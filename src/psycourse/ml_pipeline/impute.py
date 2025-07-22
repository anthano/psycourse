import numpy as np
from sklearn.impute import KNNImputer


class KNNMedianImputer(KNNImputer):
    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """Helper function to impute a single column.

        Args:
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns:
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        imputed_values = np.ma.median(donors, axis=1).data

        return imputed_values
