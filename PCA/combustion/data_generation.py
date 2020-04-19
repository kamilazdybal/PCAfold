import numpy as np
from spitfire.chemistry import reactors
from spitfire.chemistry import mechanism
from spitfire.chemistry import flamelet
from spitfire.chemistry import tabulation
import spitfire.chemistry.analysis as sca
from cantera import gas_constant
from PCA.combustion import transformations

# This module contains functions to generate data sets using Spitfire:
#
# https://github.com/sandialabs/Spitfire

# Generic isobaric, adiabiatic, closed homogeneous reactor:
def batch_reactor(chemical_mechanism, fuel_ratio, mixing_temperature, equivalence_ratio, verbose=False):
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
    `Z_stoich`
               - stoichiometric mixture fraction.
    """

    iteration = 0

    # Fuel and air streams:
    fuel = chemical_mechanism.stream('X', fuel_ratio)
    air = chemical_mechanism.stream(stp_air=True)

    Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

    species_names = chemical_mechanism.species_names
    state_space_names = np.array(['T'] + species_names)

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

            if verbose == True:
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

    return (state_space, state_space_sources, time_list, number_of_steps, Z_stoich)

# Generic isobaric, adiabiatic, open reactor (PSR):
def PSR_reactor(chemical_mechanism, fuel_ratio, mixing_temperature, equivalence_ratio, residence_times, verbose=False):
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
    `residence_times
               - a vector of residence times to loop over.

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
    `mixture_fraction`
               - vector of mixture fractions.
    `Z_stoich`
               - stoichiometric mixture fraction.
    """

    iteration = 0

    # Fuel and air streams:
    fuel = chemical_mechanism.stream('X', fuel_ratio)
    air = chemical_mechanism.stream(stp_air=True)

    Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

    # Initialize matrices that will store the outputs:
    state_space = np.zeros_like(state_space_names, dtype=float)
    state_space_sources = np.zeros_like(state_space_names, dtype=float)
    time_list = np.zeros((1,))
    mixture_fraction = np.zeros((1,))
    number_of_steps = []

    print('Integrating the homogeneous reactor...\n')

    for IC_T in mixing_temperature:

        for IC_ER in equivalence_ratio:

            for tau in residence_times:

                # Mix fuel and oxidizer stream for the selected equivalence ratio:
                mix = chemical_mechanism.mix_for_equivalence_ratio(IC_ER, fuel, air)

                # Set the initial conditions for temperature and pressure:
                mix.TP = IC_T, 101325.

                # Intialize the homogeneous reactor and integrate:

                r = reactors.HomogeneousReactor(chemical_mechanism,
                                       mix,
                                       configuration='isobaric',
                                       heat_transfer='adiabatic',
                                       mass_transfer='open',
                                       mixing_tau=tau,
                                       feed_temperature=300,
                                       feed_mass_fractions=mix.Y)

                output = r.integrate_to_steady()

                # Get the time:
                time = output.time_values
                time_list = np.hstack((time_list, time))

                # Compute how many time steps were used at this iteration:
                (n_steps, ) = np.shape(time)
                number_of_steps.append(n_steps)

                # Assuming constant mixture fraction within PSR (initial composition = feed composition):
                single_equivalence_ratio_vector = np.repeat(IC_ER, n_steps).tolist()
                mixture_fraction_at_this_iteration = transformations.get_mixture_fraction_from_equivalence_ratio(single_equivalence_ratio_vector, Z_stoich)
                mixture_fraction = np.hstack((mixture_fraction, mixture_fraction_at_this_iteration.ravel()))

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

                if verbose == True:
                    print('it.%.0f \tT: %.0f \tEquivalence ratio: %.2f \tTau: %.0e \tNumber of time steps: %.0f' % (iteration, IC_T, IC_ER, tau, n_steps))

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
    mixture_fraction = mixture_fraction[1:]

    print('\nDone!')

    return (state_space, state_space_sources, time_list, number_of_steps, mixture_fraction, Z_stoich)

# Generic steady laminar flamelet:
def steady_laminar_flamelet(chemical_mechanism, fuel_ratio, initial_condition, dissipation_rates, n_of_mf=100):
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
    `n_of_mf`
               - number of grid points in the mixture fraction space.

    Output:
    ----------
    `state_space`
               - a matrix of state space: [temperature, species mass fractions].
    `state_space_sources`
               - a matrix of state space sources:
                 [heat release rate, production rates].
    `mixture_fraction`
               - a mixture fraction vector.
    `chi`      - a dissipation rates vector.
    `Z_stoich`
               - stoichiometric mixture fraction.
    """

    # Fuel and air streams:
    fuel = chemical_mechanism.stream('X', fuel_ratio)
    air = chemical_mechanism.stream(stp_air=True)

    Z_stoich = chemical_mechanism.stoich_mixture_fraction(fuel, air)

    pressure = 101325.

    flamelet_specs = {'mech_spec': chemical_mechanism,
                      'pressure': pressure,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'grid_points': n_of_mf,
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

    # Dissipation rates:
    chi = output.dissipation_rate_stoich_grid
    (n_mf, n_chi) = np.shape(chi)
    chi = np.reshape(chi, (n_mf*n_chi, ))

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

    return (state_space, state_space_sources, mixture_fraction, chi, Z_stoich)

# Generic isobaric, adiabiatic, open reactor (PSR):
def PSR_reactors_in_series(chemical_mechanism, fuel_ratio, equivalence_ratio, residence_times, verbose=False):
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
    `equivalence_ratio`
               - a vector of equivalence ratios to loop over.
    `residence_times
               - a vector of residence times to loop over.

    Output:
    ----------
    `state_space`
               - a matrix of state space: [temperature, species mass fractions].
    `state_space_sources`
               - a matrix of state space sources:
                 [heat release rate, production rates].
    `mixture_fraction`
               - vector of mixture fractions.
    `residence_times_vector`
               - vector of residence times.
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
    mixture_fraction = np.zeros((1,))
    residence_times_vector = np.zeros((1,))
    number_of_steps = []

    print('Integrating the homogeneous reactor...\n')

    for IC_ER in equivalence_ratio:

        Z = transformations.get_mixture_fraction_from_equivalence_ratio(IC_ER, Z_stoich)

        # Mix fuel and oxidizer stream for the selected equivalence ratio:
        initial_mix = chemical_mechanism.mix_streams([(air, 1 - Z), (fuel, Z)], basis='mass', constant='HP')
        initial_feed = chemical_mechanism.copy_stream(initial_mix)

        initial_mix.equilibrate('HP')

        for tau in residence_times:

            # This PSR solution will be an initial mix in the PSR:
            r = reactors.HomogeneousReactor(chemical_mechanism,
                                   initial_mix,
                                   configuration='isobaric',
                                   heat_transfer='adiabatic',
                                   mass_transfer='open',
                                   mixing_tau=tau,
                                   feed_temperature=initial_feed.T,
                                   feed_mass_fractions=initial_feed.Y)

            output = r.integrate_to_steady()

            # Initialize temperature and composition:
            ct_sa, lib_shape = sca.get_ct_solution_array(chemical_mechanism, output)

            mass_fractions = ct_sa.Y
            temperature = ct_sa.T[:,None]

            EQ_mass_fractions = mass_fractions[-1,:]
            EQ_temperature = int(temperature[-1][0])

            # Assuming constant mixture fraction within PSR (initial composition = feed composition):
            mixture_fraction = np.hstack((mixture_fraction, Z))

            # Append the current residence time:
            residence_times_vector = np.hstack((residence_times_vector, tau))

            # Density:
            output = sca.compute_density(chemical_mechanism, output)
            density = output['density']

            # Get the time:
            time = output.time_values

            # Compute how many time steps were used at this iteration:
            (n_steps, ) = np.shape(time)
            number_of_steps.append(n_steps)

            # Isobaric heat capacity:
            output = sca.compute_isobaric_specific_heat(chemical_mechanism, output) # J/kg/K
            isobaric_heat_capacity = output['heat capacity cp']

            # Species production rates:
            production_rates = ct_sa.net_production_rates * ct_sa.molecular_weights
            production_rates_over_density = np.divide(production_rates, np.reshape(density, (n_steps, 1)))

            # Species enthalpies:
            species_enthalpies = ct_sa.standard_enthalpies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

            # Species energies:
            species_energies = ct_sa.standard_int_energies_RT * temperature * gas_constant / ct_sa.molecular_weights[None,:] # J/kg

            # Heat release rate:
            heat_release_rate = - np.sum(production_rates * species_enthalpies, axis=1) / density.ravel() / isobaric_heat_capacity.ravel() # K/s

            # Populate the solution matrices:
            state_space_at_this_iteration = np.hstack((EQ_temperature, EQ_mass_fractions))
            state_space_sources_at_this_iteration = np.hstack((heat_release_rate[-1], production_rates_over_density[-1,:]))

            # Append the current iteration state space to the global state space:
            state_space = np.vstack((state_space, state_space_at_this_iteration))
            state_space_sources = np.vstack((state_space_sources, state_space_sources_at_this_iteration))

            if verbose == True:
                print('it.%.0f \tMix T: %.0f \tFeed T: %.0f \tEQ T: %.0f \tEquivalence ratio: %.2f \tTau: %.0e' % (iteration, initial_mix.T, initial_feed.T, EQ_temperature, IC_ER, tau))

            # Set the new initial mixture in the reactor to be used for the next residence time:
            initial_mix.Y = EQ_mass_fractions
            initial_mix.TP = EQ_temperature, 101325.

            iteration += 1

    # Remove the zeros row:
    state_space = state_space[1:,:]
    state_space_sources = state_space_sources[1:,:]
    mixture_fraction = mixture_fraction[1:]
    residence_times_vector = residence_times_vector[1:]

    print('\nDone!')

    return (state_space, state_space_sources, mixture_fraction, residence_times_vector, Z_stoich)
