
""" script containing class and functions for modelling, integrating etc
""" 
from abc import ABC, abstractmethod
from .connectomes._connectomes import Connectome, laplacian_matrix
import jax.numpy as jnp 
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve

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

    @abstractmethod
    def f(self):
        """Differential equation function to be implemented

        Parameters:
        -----------
        u0 (array)
            array containing intial conditions
        t (array)
            numpy array containing time steps at which to evaluate model
            e.g. t = np.linespace(0, 1, 100)
        params (array)
            array containing parameter values
            e.g. params = [L, k]
                L = Graph Laplacian
                k = diffusion coefficient
        """
        raise NotImplementedError('This should be implemented')

    def simulate(self, y0, args, t0, t1, dt0, ts):
        term = ODETerm(self.f)
        saveat = SaveAt(ts=ts)

        return diffeqsolve(term, Tsit5(), t0, t1, dt0, y0, args=args, saveat=saveat)