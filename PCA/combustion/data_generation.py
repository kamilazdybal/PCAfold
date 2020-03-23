import numpy as np

# Spitfire:
from spitfire.chemistry import reactors
from spitfire.chemistry import mechanism
from spitfire.chemistry import flamelet

# This module contains functions to generate data sets using Spitfire:
#
# https://github.com/sandialabs/Spitfire

# Generic isobaric, adiabiatic, closed homogeneous reactor:
def homogeneous_reactor(chemical_mechanism, fuel_ratio, mixing_temperature, equivalence_ratio):
    """
    This function generates a state-space data set from a homogeneous, isobaric,
    adiabatic closed reactor. It uses `HomogeneousReactor` from Spitfire.

    Input:
    ----------
    `chemical_mechanism`
               - is a chemical mechanism, object of `ChemicalMechanismSpec`.
    `fuel_ratio`
               - string specifying the mass ratios of fuel components.
    `mixing_temperature`
               - a vector of initial mixing temperatures to loop over.
    `equivalence_ratio`
               - a vector of equivalence ratios to loop over.
    Output:
    ----------
    `state_space`
               - a matrix of state space: [temperature, species mass fractions].
    `time_list`
               - a vector of times corresponding to each row of the
                 `state_space` matrix.
    `number_of_steps`
               - list of numbers of time steps that were generated at each
                 iteration.
    `state_space_names`
               - list of strings labeling the variables in the `state_space`
                 matrix.
    `Z_stoich`
               - stoichiometric mixture fraction.
    """

    iteration = 0

    # Fuel and air streams:
    fuel = chemical_mechanism.stream('X', fuel_ratio)
    air = chemical_mechanism.stream(stp_air=True)

    for IC_T in mixing_temperature:

        for IC_ER in equivalence_ratio:

            # Mix fuel and oxidizer stream for the selected equivalence ratio:
            mix = chemical_mechanism.mix_for_equivalence_ratio(IC_ER, fuel, air)

            # Set the initial conditions for temperature and pressure:
            mix.TP = IC_T, 101325.

            # Extract the state-space variables names:
            number_of_species = mix.n_species
            species_names = mix.species_names
            state_space_names = np.array(['T'] + species_names)

            Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

            if iteration==0:

                print('Integrating the homogeneous reactor...\n')

                # Initialize matrices that will store the outputs:
                state_space = np.zeros_like(state_space_names, dtype=float)
                time_list = np.zeros((1,))
                number_of_steps = []

            # Intialize the homogeneous reactor and integrate:
            r = reactors.HomogeneousReactor(chemical_mechanism, mix, 'isobaric', 'adiabatic', 'closed')
            output = r.integrate_to_steady_after_ignition(steady_tolerance=1.e-8, first_time_step=1.e-9, transient_tolerance=1.e-12)

            # Get the time:
            time = output.time_values
            time_list = np.hstack((time_list, time))

            # Compute how many time steps were used at this iteration:
            (n_steps, ) = np.shape(time)
            number_of_steps.append(n_steps)

            # Intialize the state space matrix at this iteration:
            state_space_at_this_iteration = np.zeros([n_steps, len(state_space_names)])

            # Get the temperatures:
            temperature = output['temperature']
            state_space_at_this_iteration[:,0] = temperature

            # Get the density:
            # rho = output['density']

            # Get the species mass fractions:
            for i, species in enumerate(species_names):

                species_mass_fraction = output['mass fraction ' + species]
                state_space_at_this_iteration[:,i+1] = species_mass_fraction

            print('it.%.0f \tT: %.0f \tEquivalence ratio: %.2f \tNumber of time steps: %.0f' % (iteration, IC_T, IC_ER, n_steps))

            # Append the current iteration state space to the global state space:
            state_space = np.vstack((state_space, state_space_at_this_iteration))

            iteration += 1

    # Remove the zeros row:
    state_space = state_space[1:,:]
    time_list = time_list[1:]

    print('\nDone!')

    return (state_space, time_list, number_of_steps, state_space_names, Z_stoich)
