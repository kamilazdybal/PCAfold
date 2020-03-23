import numpy as np

def get_mixture_fraction_from_equivalence_ratio(equivalence_ratio, Z_stoich):
    """
    This function computes mixture fraction vector based on the equivalence
    ratio vector and the stoichiometric mixture fraction using:

    equivalence_ratio = Z/(1 - Z) * (1 - Z_stoich)/Z_stoich

    Input:
    ----------
    `equivalence_ratio`
               - vector of equivalence ratios.
    `Z_stoich` - stoichiometric mixture fraction.

    Output:
    ----------
    `Z`        - vector of mixture fractions. Each element corresponds to the
                 element in `equivalence_ratio` vector.
    """

    Z = np.empty([len(equivalence_ratio), 1])

    for i in range(0, len(equivalence_ratio)):

        A = equivalence_ratio[i] * Z_stoich / (1 - Z_stoich)
        Z[i] = A / (1+A)

    return Z

def get_equivalence_ratio_from_mixture_fraction(Z, Z_stoich):
    """
    This function computes equivalence ratio vector based on the mixture
    fraction vector and the stoichiometric mixture fraction using:

    equivalence_ratio = Z/(1 - Z) * (1 - Z_stoich)/Z_stoich

    Input:
    ----------
    `Z`        - vector of mixture fractions.
    `Z_stoich` - stoichiometric mixture fraction.

    Output:
    ----------
    `equivalence_ratio`
               - vector of equivalence ratios.
    """

    equivalence_ratio = np.empty([len(Z), 1])

    for i in range(0, len(Z)):

        equivalence_ratio[i] = Z[i]/(1 - Z[i]) * (1 - Z_stoich)/Z_stoich

    return equivalence_ratio
