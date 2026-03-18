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
        self._stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

    @property
    def stepsize_controller(self):
        """Get the stepsize controller"""
        return self._stepsize_controller
    
    @stepsize_controller.setter
    def stepsize_controller(self, controller):
        """Set the stepsize controller
        args:
        controller : StepsizeController
            A stepsize controller (PIDController from Diffrax)
        """
        if controller is not None and not isinstance(controller, PIDController):
            raise TypeError(f"Controller must be a PIDController from diffrax, got {type(controller)}")
        self._stepsize_controller = controller

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

    def simulate(self, y0, args, t0, t1, dt0, ts, max_steps=10000):
        saveat = SaveAt(ts=ts)
        return diffeqsolve(
                self.term, 
                Tsit5(), 
                t0, 
                t1, 
                dt0, 
                y0, 
                args=args, 
                saveat=saveat,
                max_steps=max_steps,
                stepsize_controller=self._stepsize_controller
            )
