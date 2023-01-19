import numpy as np
from scipy.special import erf
from .Exceptions import ConvergenceFailure
from scipy.optimize import fmin_powell
def get_smearing_function(scheme_name):
    """
    This function returns the requested smearing function.

    Parameters:
        - scheme_name (string): The name of the desired smearing function.

    Returns:
        - function: A smearing function

    Available smearing functions:
        - 'fermi': Fermi-Dirac smearing method
        - 'gaussian': Gaussian smearing method
    """

    # Dictionary containing all available smearing schemes
    schemes = {'fermi': fermi,
               'gaussian': gaussian}

    # Confirm that the smearing scheme is known form
    assert scheme_name in schemes, f'Unknown smearing scheme provided "{scheme}"'

    # Select and return the requested smearing function
    return schemes[scheme_name]

def get_occupancies(eigenvalues, fermi_level, scheme, sigma):
    """
    Will calculate the occupancies of a list of orbitals.

    Parameters
        - eigenvalues (list/array): Eigen-energies, i.e energy levels, in H
        - fermi_level (float): Fermi-energy in Ha
        - scheme (string): Specifies smearing method 'gaussian' or 'fermi'
        - sigma (float): Smearing width (kT) in Ha

    Returns:
        - array (float): Occupancies of the orbitals
    """

    # Retire the requested smearing function
    smearing_function = get_smearing_function(scheme)

    # Calculate the occupancies
    occupancies = np.array([smearing_function(e, fermi_level, sigma)
                            for e in eigenvalues])
    # Return the occupancies
    return occupancies

def get_entropy_term(eigenvalues, fermi_level, scheme, sigma):
    """
    Calculated entropy term for fractional occupations temperate kT (sigma).

    Parameters:
        - eigenvalues (list/array): Eigen-energies, i.e energy levels, in Ha
        - fermi_level (float): Fermi-energy in Ha
        - scheme (string): Specifies smearing method 'gaussian' or 'fermi'
        - sigma (float): Smearing width (kT) in Ha

    Returns:
        - float: Entropy term in Ha

    Notes:
        - Ensure that scheme selected is the same function that was
          used to calculate the Fermi-level

    Forms:
        For the Fermi-Dirac method:
                2 * -SUM_i[fo_i * ln(fo_i) + (1 - fo_i) * ln(1 - f_i)]
            Where 'fo_i' is the fractional occupancy of orbital 'i'. The
            factor of 2 is required for non-spin-polarised systems.

        For the Gaussian method:
                0.5 * SUM_i(exp(-(Ei - Ef) / sigma)^2) / sqrt(pi)
            Where Ef is the Fermi-energy & Ei is the eigen-energy of orbital i.

    Notes:
        - The Mermin free energy can be obtained via E_free = E - sigma * S
    """

    # If the Fermi-Dirac method was used
    if scheme == 'fermi':

        # Get fractional orbital occupancies, i.e values in the range of [0, 1]
        fo = [fermi(e, fermi_level, sigma) / 2 for e in eigenvalues]

        # Calculate and return the entropy term. Normally 1's an 0's would
        # be removed to avoid math errors. This can be avoided by rearranging
        # the equation, its also faster.
        return 2 * -sum([np.log(f ** f * (1 - f) ** (1 - f)) for f in fo])

    # If the Gaussian method was used
    elif scheme == 'gaussian':

        # First calculate the xi values
        xi = [(e - fermi_level) / sigma for e in eigenvalues]

        # Now calculate and return the entropy term
        return sum([np.exp(- x ** 2) for x in xi]) * 0.5 / np.sqrt(np.pi)

    # If any other or an unknown method was requested
    else:
        # Raise an exception
        raise Exception(f'Unknown smearing scheme provided "{scheme}"')


def fermi(e, ef, sigma):
    """
    Fermi-Dirac smearing function.

    Parameters:
        - e (float): Eigenenergy of orbital in Ha
        - ef (float): Fermi-energy in Ha
        - sigma (float): Smearing width (kT) in Ha

    Returns:
        - float: Number of electrons present in the target orbital

    Notes:
        - Occupancy is the total number of electrons present in the
          orbital, i.e any real value within the range [0, 2] can be
          returned. As opposed to a fractional value of range [0, 1].
    """

    # The numpy exp(x) function is capable of handling large x values
    # gracefully, i.e. returns 'inf' rather than crashing. Therefore,
    # there is no need to manually cap the exponential function.
    return 2.0 / (1.0 + np.exp((e - ef) / sigma))


def gaussian(e, ef, sigma):
    """
    Gaussian smearing function.

    Parameters:
        - e (float): Eigen-energy of orbital in Ha
        - ef (float): Fermi-energy in Ha
        - sigma (float): Smearing width (kT) in Ha

    Returns:
        - float: Number of electrons present in the target orbital

    Notes:
        - Occupancy is the total number of electrons present in the
          orbital, i.e any real value within the range [0, 2] can be
          returned. As opposed to a fractional value of range [0, 1].
    """

    return 1.0 - erf((e - ef) / sigma)


def determine_fermi_level(eigenvalues, nelectrons, sigma=0.0, scheme='fermi',
                          threshold=1.E-10, max_steps=600, powell=False):
    """
    Determines the Fermi-level (aka chemical potential) of a given system.

    Parameters:
        - eigenvalues (list/array): Eigen-energies, i.e energy levels, in Ha
        - nelectrons (int): Number of valence electrons
        - scheme (string): Specifies smearing method 'gaussian' or 'fermi'
        - sigma (float): Smearing width (kT) in Ha
        - threshold (float): Sets convergence criteria
        - max_steps (int): Sets maximum permitted number of search iterations
        - powell (bool): If True then an alternative scipy powell solver is used
            if False then a simple line search is employed.

    Returns:
        - float: Calculated Fermi-level, in Ha

    Notes:
        - Currently this function only supports non-spin-polarised systems.
          That is to say, systems with 2 electrons per orbital.
        - Method used is a top down line search method.
    """
    def count_elec(ef):
        """
        Counts the number of electrons at a given Fermi-level

        Parameters:
            - ef (float): Fermi-energy in Ha

        Returns:
            - float: Total number of electrons
        """

        # Calculate occupancies for each eigenvalue, and return the sum
        return sum([smearing_function(e, ef, sigma) for e in eigenvalues])

    # Select the smearing scheme
    smearing_function = get_smearing_function(scheme)

    # Set the initial fermi_level; to an artificially high value
    fermi_level = eigenvalues[-1] + 10 * sigma



    # If using the powell method
    if powell:

        def electron_delta(fermi_level):
            return (count_elec(fermi_level) - nelectrons) ** 2

        fermi_level, *_, warnings = fmin_powell(electron_delta,
                                                fermi_level,
                                                xtol=threshold,
                                                ftol=threshold,
                                                disp=False,
                                                maxiter=max_steps,
                                                full_output=True,
                                                direc=np.array([-1]))

        # As fmin_powell returns a 0D array we must extract the value before continuing
        fermi_level = fermi_level.item()

        # If the maximum number of iterations was reached then raise an exception
        if warnings != 0:
            raise ConvergenceFailure('Fermi level search failed to converge within'
                                     ' the maximum number of allowed iterations')

    # If using the simple line search method
    else:
        # Get the current electron count
        nelec_current = count_elec(fermi_level)

        # step counter to track the number of iterations
        nsteps = 0

        # Loop until convergence or maximum iteration count is reached
        while((nelec_current - nelectrons) ** 2 > threshold):

            # If maximum permitted number of iterations has been reached
            if nsteps > max_steps:
                # Then raise and exception
                raise ConvergenceFailure('Fermi level search failed to converge within'
                                ' the maximum number of allowed iterations')

            # Determine the step direction. While, not required for a top
            # down line search method, this is kept in-case we somehow end
            # up below the Fermi-level.
            direc = np.sign(nelectrons - nelec_current)

            # Set an initial step size of 0.50
            step = 0.50

            # If the step overshoots
            while((count_elec(fermi_level + direc * step) - nelectrons) < 0.0):
                # Then keep decreasing the step size until it doesn't
                step *= 0.50

            # Update the Fermi-level
            fermi_level += step * direc

            # Update the current electron count
            nelec_current = count_elec(fermi_level)

            # Iterate the step counter
            nsteps += 1

    # Finally, return the Fermi-level
    return fermi_level
