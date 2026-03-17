from abc import ABC, abstractmethod
from .connectomes._connectomes import Connectome, laplacian_matrix
import jax.numpy as jnp 
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve, PIDController

class NetworkModel(ABC): 
    """Abstract Base Class for implementing dynamical network models.

       Parameters:
       -----------
       network_path (str)
            path to a network array, .csv, or .graphml

    """
    def __init__(self, c: Connectome):
        """initialise class with network and model 
        args:
        c           : Connectome
                     string to the location of a csv containing an adjacency matrix
        model_name  : str 
                     string containing the model user wishes to initialised 
                     options: 
                        - 'network_diffusion' 
                           du = k*(L @ p)
        """
        self._c = c
        self._nv = len(c.parc)
        # self._A = adjacency_matrix(network_path)
        # self._D = degree_matrix(self.__A)
        self._L = jnp.array(laplacian_matrix(c))
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
        u0 (array)
            array containing intial conditions
        t (array)
            numpy array containing time steps at which to evaluate model
            e.g. t = jnp.linespace(0, 1, 100)
        params (array)
            array containing parameter values
            e.g. params = jnp.array([k])
                k = diffusion coefficient
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
