import pytest
import jax.numpy as jnp
import jax
from diffrax import PIDController, Dopri5
from pryonix.models import NetworkModel
from pryonix.models.model_library import NetworkDiffusion, NetworkFKPP, ScaledNetworkATN, ScaledNetworkFKPP, NetworkATN, ScaledNetworkATN
from pryonix.connectomes import Connectome, connectome_path

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
        sol = m.simulate(u0, p[0], 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 0.5, atol=1e-3)

    def test_network_diffusion_dopri5(self, connectome, params):
        m = NetworkDiffusion(connectome)
        u0, p = params
        sol = m.simulate(u0, p[0], 0, 100, jnp.linspace(0, 100, 10), solver=Dopri5(), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 0.5, atol=1e-3)    
    
    def test_network_diffusion_single_step(self, connectome, params):
        m = NetworkDiffusion(connectome)
        u0, p = params
        with pytest.raises(RuntimeError, match="max_steps"):
            m.simulate(u0, p[0], 0, 100, jnp.linspace(0, 100, 10), dt0=1e-2,
                    max_steps=1, stepsize_controller=PIDController(1e-5, 1e-7))

    def test_fkpp(self, connectome, params):
        m = NetworkFKPP(connectome)
        u0, p = params
        sol = m.simulate(u0, p[0:2], 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 1.0, atol=1e-3)

    def test_scaled_fkpp(self, connectome, params):
        m = ScaledNetworkFKPP(connectome, u0=0.5, ui=2.0)
        _, p = params
        u0 = jnp.ones(83) * 0.5
        sol = m.simulate(u0, p[0:2], 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1]), 0.5, atol=1e-3)
        
        u0_2 = jnp.ones(83) * 0.6
        sol_2 = m.simulate(u0_2, p[0:2], 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert jnp.isclose(jnp.mean(sol_2.ys[-1]), 2.0, atol=1e-3)

    def test_atn(self, connectome, params):
        m = NetworkATN(connectome)
        u0, p = params
        sol = m.simulate(jnp.array([u0, u0, u0]), p, 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 3, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1,0, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,1, :]), 2.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,2, :]), 1.0, atol=1e-3)

    def test_scaled_atn(self, connectome, params):
        m = ScaledNetworkATN(connectome, ui=1.0, part=1.0)
        u0, p = params
        sol = m.simulate(jnp.array([u0, u0, u0]), p[0:5], 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7))
        
        assert isinstance(m, NetworkModel)
        assert sol.ys.shape == (10, 3, 83)
        assert sol.ts.shape == (10,)
        assert jnp.isclose(jnp.mean(sol.ys[-1,0, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,1, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,2, :]), 1.0, atol=1e-3)

        sol = m.simulate(jnp.array([u0, u0, u0]),  jnp.array([1.0, 1.0, 1.0, 0.0, 1.0]), 0, 100, jnp.linspace(0, 100, 10), stepsize_controller=PIDController(1e-5, 1e-7)        )
        assert jnp.isclose(jnp.mean(sol.ys[-1,0, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,1, :]), 1.0, atol=1e-3)
        assert jnp.isclose(jnp.mean(sol.ys[-1,2, :]), 1.0, atol=1e-3)
