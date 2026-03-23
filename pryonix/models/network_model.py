from abc import ABC, abstractmethod
from ..connectomes._connectomes import Connectome, laplacian_matrix
import jax.numpy as jnp 
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve, PIDController

class NetworkModel(ABC): 
    """Abstract Base Class for implementing dynamical network models.

       Parameters:
       -----------
       c : Connectome
           A Connectome object containing the network structure on which the model will be implemented.

    """
    def __init__(self, c: Connectome):
        """initialise class with network and model 
        args:
        c : Connectome
            A Connectome object containing the network structure on which the model will be implemented.
        """
        self._c = c
        self._nv = len(c.parc)
        self.L = jnp.array(laplacian_matrix(c))
        self.term = ODETerm(self.f)
    #     self._stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

    # @property
    # def stepsize_controller(self):
    #     """Get the stepsize controller"""
    #     return self._stepsize_controller
    
    # @stepsize_controller.setter
    # def stepsize_controller(self, controller):
    #     """Set the stepsize controller
    #     args:
    #     controller : StepsizeController
    #         A stepsize controller (PIDController from Diffrax)
    #     """
    #     if controller is not None and not isinstance(controller, PIDController):
    #         raise TypeError(f"Controller must be a PIDController from diffrax, got {type(controller)}")
    #     self._stepsize_controller = controller

    @abstractmethod
    def f(self):
        """Differential equation function to be implemented

        Parameters:
        -----------
        t (float)
            time variable
        u0 (array)
            array containing state variable
        params (array)
            array containing parameter values

        Example:
        --------
        def f(self, t, u, args):
            transport_rate = args[0]
            du = -transport_rate * jnp.dot(self._L, u)
            return du
        """
        raise NotImplementedError('This should be implemented')

    def simulate(self, y0, args, t0, t1, ts, 
                dt0=1e-2, 
                max_steps=10000, 
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
                **kwargs):
        """Simulate the model over a given time interval.

        Parameters
        ----------
        y0 : array
            Initial state vector.
        args : array
            Parameters passed to the ODE function `f`.
        t0 : float
            Start time.
        t1 : float
            End time.
        ts : array
            Timepoints at which to save the solution.
        dt0 : float, optional
            Initial step size (default: 1e-2). Can be set to None for solvers
            that determine their own initial step size.
        max_steps : int, optional
            Maximum number of solver steps before raising a RuntimeError (default: 10000).
        **kwargs
            Additional keyword arguments forwarded to `diffeqsolve`. These take
            precedence over any defaults set by this method. Commonly used options:

            - ``solver`` : AbstractSolver
                ODE solver to use (default: Tsit5()).
            - ``saveat`` : SaveAt
                Override the default SaveAt(ts=ts) with a custom save configuration.
            - ``stepsize_controller`` : AbstractStepSizeController
                Step size controller (default: PIDController(rtol=1e-3, atol=1e-3)).
            - ``adjoint`` : AbstractAdjoint
                Method for differentiating through the solver.

            See the `diffrax documentation <https://docs.kidger.site/diffrax/api/diffeqsolve/>`_
            for the full list of available options.

        Returns
        -------
        Solution
            A diffrax ``Solution`` object with attributes:

            - ``ys`` : array of shape ``(len(ts), *y0.shape)`` — saved states.
            - ``ts`` : array of shape ``(len(ts),)`` — saved timepoints.

        Raises
        ------
        RuntimeError
            If the solver exceeds ``max_steps`` before reaching ``t1``.

        Examples
        --------
        Basic usage:

        sol = model.simulate(y0, params, t0=0, t1=100, ts=jnp.linspace(0, 100, 10))

        With a custom solver and tighter tolerances:

        from diffrax import Dopri5, PIDController
        sol = model.simulate(y0, params, t0=0, t1=100, ts=jnp.linspace(0, 100, 10),
                            solver=Tsit5(),
                            stepsize_controller=PIDController(rtol=1e-6, atol=1e-6))
        """
        base_kwargs = dict(
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
        )
        base_kwargs.update(kwargs)

        solver = base_kwargs.pop('solver', Tsit5())
        return diffeqsolve(
            self.term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=SaveAt(ts=ts),
            args=args,
            **base_kwargs,
        )