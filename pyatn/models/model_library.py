import jax.numpy as jnp 
from .._NetworkModels import NetworkModel

class NetworkDiffusion(NetworkModel): 
    def f(self, t, u, args):
        transport_rate = args 
        du = -transport_rate * jnp.dot(self._L, u) 
        return du


class NetworkFKPP(NetworkModel): 
    def f(self, t, u, args):
        transport_rate, production_rate = args 
        du = -transport_rate * jnp.dot(self._L, u) + production_rate * u * ( 1 - u )
        return du

class ScaledNetworkFKPP(NetworkModel):
    def __init__(self, L, u0, ui):
        self.u0 = u0
        self.ui = ui

        super().__init__(L)

    def f(self, t, u, args):
        transport_rate, production_rate = args
        du = -transport_rate * jnp.dot(self._L, u - self.u0) + production_rate * (u - self.u0) * ( (self.ui - self.u0) - (u - self.u0) )
        return du

class NetworkATN(NetworkModel):
    def f(self, t, u, args):
        x, y, z = u
        amyloid_production_rate, tau_transport_rate, tau_production_rate, baseline, coupling, atrophy_rate = args

        d_x = amyloid_production_rate * x * ( 1 - x )
        d_y = -tau_transport_rate * jnp.dot(self._L, y) + tau_production_rate * y * ( (baseline + coupling * x ) - y )
        d_z = atrophy_rate * y * ( 1 - z )
        Du = jnp.vstack([d_x, d_y, d_z])
        return Du
    
class ScaledNetworkATN(NetworkModel):
    def __init__(self, L, ui, part): 
        self.ui = ui
        self.part = part
        
        super().__init__(L)

    def f(self, t, u, args):
        x, y, z = u
        amyloid_production_rate, tau_transport_rate, tau_production_rate, coupling, atrophy_rate = args

        d_x = amyloid_production_rate * self.ui * x * ( 1 - x )
        d_y = -tau_transport_rate * jnp.dot(self._L, y) + tau_production_rate * (self.part + (coupling * x)) * y * ( 1 - y )
        d_z = atrophy_rate * y * ( 1 - z )
        Du = jnp.vstack([d_x, d_y, d_z])
        return Du
