import pytest
import jax.numpy as jnp
import jax
from diffrax import PIDController
from pryonix import NetworkModel
from pryonix.models.model_library import NetworkDiffusion, NetworkFKPP, NetworkATN
from pryonix.connectomes import Parcellation, Connectome, connectome_path, get_node_id

@pytest.fixture
def params():
    key = jax.random.key(1234)
    inits =  0.5 + jax.random.normal(key, 83) * 0.01
    params = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return inits, params

@pytest.fixture
def connectome():
    c = Connectome.from_graph_path(connectome_path())
    return c

class TestModel:
    """Tests for NetworkModel"""
    def test_network_diffusion(self, connectome, params):
        m = NetworkDiffusion(connectome)
        u0, p = params
        m.stepsize_controller = PIDController(1e-5, 1e-7)
        sol = m.simulate(u0, p[0], 0, 100, 1e-2, jnp.linspace(0, 100, 10))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 0.5, atol=1e-3)
        assert m.stepsize_controller.rtol == 1e-5
        assert m.stepsize_controller.atol == 1e-7
        assert isinstance(m.stepsize_controller, PIDController)

    def test_fkpp(self, connectome, params):
        m = NetworkFKPP(connectome)
        u0, p = params
        m.stepsize_controller = PIDController(1e-5, 1e-7)
        sol = m.simulate(u0, p[0:2], 0, 100, 1e-2, jnp.linspace(0, 100, 10))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 1.0, atol=1e-3)
        assert m.stepsize_controller.rtol == 1e-5
        assert m.stepsize_controller.atol == 1e-7
        assert isinstance(m.stepsize_controller, PIDController)

    def test_atn(self, connectome, params):
        m = NetworkATN(connectome)
        u0, p = params
        m.stepsize_controller = PIDController(1e-5, 1e-7)
        sol = m.simulate(jnp.array([u0, u0, u0]), p, 0, 100, 1e-2, jnp.linspace(0, 100, 10))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 3, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1,0, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,1, :]), 2.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,2, :]), 1.0, atol=1e-3)
        assert m.stepsize_controller.rtol == 1e-5
        assert m.stepsize_controller.atol == 1e-7
        assert isinstance(m.stepsize_controller, PIDController)