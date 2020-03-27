import numpy as np
from spitfire.chemistry import reactors
from spitfire.chemistry import mechanism
from spitfire.chemistry import flamelet
from spitfire.chemistry import tabulation
import spitfire.chemistry.analysis as sca
from cantera import gas_constant

# This module contains functions to generate data sets using Spitfire:
#
# https://github.com/sandialabs/Spitfire

# Generic isobaric, adiabiatic, closed homogeneous reactor:
def homogeneous_reactor(chemical_mechanism, fuel_ratio, mixing_temperature, equivalence_ratio):
    """
    This function generates a state-space data set from a homogeneous, isobaric,
    adiabatic closed reactor.
    It uses `HomogeneousReactor` class from Spitfire.

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
    `state_space_sources`
               - a matrix of state space sources:
                 [heat release rate, production rates].
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

    # Extract the state-space variables names:
    number_of_species = chemical_mechanism.n_species
    species_names = chemical_mechanism.species_names
    state_space_names = np.array(['T'] + species_names)

    Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

    # Initialize matrices that will store the outputs:
    state_space = np.zeros_like(state_space_names, dtype=float)
    state_space_sources = np.zeros_like(state_space_names, dtype=float)
    time_list = np.zeros((1,))
    number_of_steps = []

    print('Integrating the homogeneous reactor...\n')

    for IC_T in mixing_temperature:

        for IC_ER in equivalence_ratio:

            # Mix fuel and oxidizer stream for the selected equivalence ratio:
            mix = chemical_mechanism.mix_for_equivalence_ratio(IC_ER, fuel, air)

            # Set the initial conditions for temperature and pressure:
            mix.TP = IC_T, 101325.

            # Intialize the homogeneous reactor and integrate:
            r = reactors.HomogeneousReactor(chemical_mechanism, mix, 'isobaric', 'adiabatic', 'closed')
            output = r.integrate_to_steady_after_ignition(steady_tolerance=1.e-8, first_time_step=1.e-9, transient_tolerance=1.e-12)

            # Get the time:
            time = output.time_values
            time_list = np.hstack((time_list, time))

            # Compute how many time steps were used at this iteration:
            (n_steps, ) = np.shape(time)
            number_of_steps.append(n_steps)

            # Density:
            output = sca.compute_density(chemical_mechanism, output)
            density = output['density']

            # Isobaric heat capacity:
            output = sca.compute_isobaric_specific_heat(chemical_mechanism, output) # J/kg/K
            isobaric_heat_capacity = output['heat capacity cp']

            ct_sa, lib_shape = sca.get_ct_solution_array(chemical_mechanism, output)

            # Species mass fractions:
            mass_fractions = ct_sa.Y

            # Temperature:
            temperature = ct_sa.T[:,None] # K

            # Species production rates:
            production_rates = ct_sa.net_production_rates * ct_sa.molecular_weights
            production_rates_over_density = np.divide(production_rates, np.reshape(density, (n_steps, 1)))

            # Species enthalpies:
            species_enthalpies = ct_sa.standard_enthalpies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

            # Species energies:
            species_energies = ct_sa.standard_int_energies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

            # Heat release rate:
            heat_release_rate = - np.sum(production_rates * species_enthalpies, axis=1) / density.ravel() / isobaric_heat_capacity.ravel() # K/s

            print('it.%.0f \tT: %.0f \tEquivalence ratio: %.2f \tNumber of time steps: %.0f' % (iteration, IC_T, IC_ER, n_steps))

            # Intialize the state space matrix at this iteration:
            state_space_at_this_iteration = np.zeros([n_steps, len(state_space_names)])
            state_space_sources_at_this_iteration = np.zeros([n_steps, len(state_space_names)])

            # Populate the solution matrices:
            state_space_at_this_iteration = np.hstack((temperature, mass_fractions))
            state_space_sources_at_this_iteration = np.hstack((np.reshape(heat_release_rate, (n_steps,1)), production_rates_over_density))

            # Append the current iteration state space to the global state space:
            state_space = np.vstack((state_space, state_space_at_this_iteration))
            state_space_sources = np.vstack((state_space_sources, state_space_sources_at_this_iteration))

            iteration += 1

    # Remove the zeros row:
    state_space = state_space[1:,:]
    state_space_sources = state_space_sources[1:,:]
    time_list = time_list[1:]

    print('\nDone!')

    return (state_space, state_space_sources, time_list, number_of_steps, state_space_names, Z_stoich)

# Generic steady laminar flamelet:
def steady_laminar_flamelet(chemical_mechanism, fuel_ratio, initial_condition, dissipation_rates):
    """
    This function generates a state-space data set from laminar flamelet.
    It uses `Flamelet` class from Spitfire.

    Input:
    ----------
    `chemical_mechanism`
               - is a chemical mechanism, object of `ChemicalMechanismSpec`.
    `fuel_ratio`
               - string specifying the mass ratios of fuel components.
    `initial_condition`
               - initial condition for the flamelet library.
    `dissipation_rates`
               - a vector of dissipation rates.

    Output:
    ----------
    `state_space`
               - a matrix of state space: [temperature, species mass fractions].
    `state_space_sources`
               - a matrix of state space sources:
                 [heat release rate, production rates].
    `state_space_names`
               - list of strings labeling the variables in the `state_space`
                 matrix.
    `mixture_fraction`
               - a mixture fraction vector.
    `Z_stoich`
               - stoichiometric mixture fraction.
    """

    # Fuel and air streams:
    fuel = chemical_mechanism.stream('X', fuel_ratio)
    air = chemical_mechanism.stream(stp_air=True)

    Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

    # Extract the state-space variables names:
    number_of_species = chemical_mechanism.n_species
    species_names = chemical_mechanism.species_names
    state_space_names = np.array(['T'] + species_names)

    pressure = 101325.

    flamelet_specs = {'mech_spec': chemical_mechanism,
                      'pressure': pressure,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'grid_points': 100,
                      'grid_type': 'uniform',
                      'include_enthalpy_flux': True,
                      'include_variable_cp': True}

    output = tabulation.build_adiabatic_slfm_library(flamelet_specs,
                                        diss_rate_values=dissipation_rates,
                                        diss_rate_ref='stoichiometric',
                                        verbose=False)

    # Density:
    output = sca.compute_density(chemical_mechanism, output)
    density = output['density']

    # Isobaric heat capacity:
    output = sca.compute_isobaric_specific_heat(chemical_mechanism, output) # J/kg/K
    isobaric_heat_capacity = output['heat capacity cp']

    ct_sa, lib_shape = sca.get_ct_solution_array(chemical_mechanism, output)

    # Species mass fractions:
    mass_fractions = ct_sa.Y

    # Temperature:
    temperature = ct_sa.T[:,None] # K

    n_obs = len(temperature)

    # Species production rates:
    production_rates = ct_sa.net_production_rates * ct_sa.molecular_weights
    production_rates_over_density = np.divide(production_rates, np.reshape(density, (n_obs, 1)))

    # Species enthalpies:
    species_enthalpies = ct_sa.standard_enthalpies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

    # Species energies:
    species_energies = ct_sa.standard_int_energies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

    # Heat release rate:
    heat_release_rate = - np.sum(production_rates * species_enthalpies, axis=1) / density.ravel() / isobaric_heat_capacity.ravel() # K/s

    # Mixture fraction:
    mixture_fraction = np.reshape(output.mixture_fraction_grid, (n_obs,1))

    # Populate the solution matrices:
    state_space = np.hstack((temperature, mass_fractions))
    state_space_sources = np.hstack((np.reshape(heat_release_rate, (n_obs,1)), production_rates_over_density))

    print('\nDone!')

    return (state_space, state_space_sources, state_space_names, mixture_fraction, Z_stoich)
