import jax.numpy as jnp 
from .._NetworkModels import NetworkModel

class NetworkFKPP(NetworkModel): 
    def f(self, t, u, args):
        transport_rate, production_rate = args 
        du = -transport_rate * jnp.dot(self._L, u) + production_rate * u * ( 1 - u )
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


class ScaledNetworkATNLinearPD(NetworkModel):
    def __init__(self, L, ui, part): 
        self.ui = ui
        self.part = part
        
        super().__init__(L)

    def f(self, t, u, args):
        x, y, z = u
        amyloid_production_rate, tau_transport_rate, tau_production_rate, coupling, atrophy_rate, drug_effect = args

        d_x = amyloid_production_rate * self.ui * x * ( 1 - x ) - drug_effect * x
        d_y = -tau_transport_rate * jnp.dot(self._L, y) + tau_production_rate * (self.part + (coupling * x)) * y * ( 1 - y )
        d_z = atrophy_rate * y * ( 1 - z )
        Du = jnp.vstack([d_x, d_y, d_z])
        return Du


class ScaledNetworkATNNonLinearPD(NetworkModel):
    def __init__(self, L, ui, part): 
        self.ui = ui
        self.part = part
        
        super().__init__(L)

    def f(self, t, u, args):
        x, y, z = u
        amyloid_production_rate, tau_transport_rate, tau_production_rate, coupling, atrophy_rate, emax, ehalf = args

    
        h = (emax) / (ehalf + 1)
        d_x = amyloid_production_rate * self.ui * x * ( 1 - x ) - h * x
        d_y = -tau_transport_rate * jnp.dot(self._L, y) + tau_production_rate * (self.part + (coupling * x)) * y * ( 1 - y )
        d_z = atrophy_rate * y * ( 1 - z )
        Du = jnp.vstack([d_x, d_y, d_z])
        return Du