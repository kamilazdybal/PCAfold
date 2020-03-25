from spitfire.time.governor import FinalTime, Steady, Governor, SaveAllDataToList, CustomTermination
from spitfire.time.methods import ESDIRK64, BackwardEulerWithError, ForwardEuler, AdaptiveERK54CashKarp
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController

def transport_PCs(eta_start, RHS_function):
    """
    This function performs time integration of the PCs using a Right Hand Side function.
    """

    governor = Governor()

    governor.termination_criteria = Steady(1.e-4)
    governor.log_rate = 100
    governor.maximum_time_step_count = 2e4
    governor.projector_setup_rate = 1

    data = SaveAllDataToList(initial_solution=eta_start, initial_time=0.)

    governor.custom_post_process_step = data.save_data

    governor.integrate(right_hand_side=lambda t, y: RHS_function(y),
                       initial_condition=eta_start,
                       controller=PIController(first_step=1.e-2, max_step=1.e0, target_error=1.e-8),
                       method=AdaptiveERK54CashKarp())

    return data.solution_list
