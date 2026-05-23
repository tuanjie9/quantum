"""量子系统仿真 - 时间演化/Lindblad/主方程/随机薛定谔方程"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
import math


@dataclass
class SimulationResult:
    """仿真结果"""
    times: np.ndarray
    states: List[np.ndarray]
    expectations: Dict[str, np.ndarray] = field(default_factory=dict)
    num_steps: int = 0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def final_state(self) -> np.ndarray:
        """获取终态"""
        return self.states[-1] if self.states else np.array([])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_steps': self.num_steps,
            'total_time': self.total_time,
            'final_purity': float(np.real(np.trace(
                np.outer(self.final_state(), self.final_state().conj())
            ))) if len(self.final_state()) > 0 else 0.0
        }


@dataclass
class EvolutionConfig:
    """演化配置"""
    dt: float = 0.01
    total_time: float = 1.0
    method: str = "rk4"
    observable_interval: int = 1
    seed: Optional[int] = None


class SchrodingerEvolver:
    """薛定谔方程时间演化器"""

    def __init__(self, hamiltonian: np.ndarray, config: Optional[EvolutionConfig] = None):
        self.hamiltonian = hamiltonian
        self.config = config or EvolutionConfig()
        self.dim = hamiltonian.shape[0]

    def evolve(self, initial_state: np.ndarray, observables: Optional[List[np.ndarray]] = None) -> SimulationResult:
        """演化纯态"""
        dt = self.config.dt
        num_steps = int(self.config.total_time / dt)
        times = np.linspace(0, self.config.total_time, num_steps + 1)
        state = initial_state.copy().astype(complex)
        states = [state.copy()]
        expectations: Dict[str, np.ndarray] = {}
        if observables:
            for i, obs in enumerate(observables):
                expectations[f'obs_{i}'] = np.zeros(num_steps + 1)
                expectations[f'obs_{i}'][0] = float(np.real(state.conj() @ obs @ state))
        U = self._compute_propagator(dt)
        for step in range(1, num_steps + 1):
            state = U @ state
            state /= np.linalg.norm(state)
            states.append(state.copy())
            if observables and step % self.config.observable_interval == 0:
                for i, obs in enumerate(observables):
                    expectations[f'obs_{i}'][step] = float(np.real(state.conj() @ obs @ state))
        return SimulationResult(
            times=times, states=states, expectations=expectations,
            num_steps=num_steps, total_time=self.config.total_time
        )

    def _compute_propagator(self, dt: float) -> np.ndarray:
        """计算时间演化算符 U = exp(-iHdt)"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        phases = np.exp(-1j * eigenvalues * dt)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T

    def energy_expectation(self, state: np.ndarray) -> float:
        """能量期望值"""
        return float(np.real(state.conj() @ self.hamiltonian @ state))

    def time_evolution_operator(self, t: float) -> np.ndarray:
        """任意时刻的演化算符"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        phases = np.exp(-1j * eigenvalues * t)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


class LindbladEvolver:
    """Lindblad 主方程演化器 (密度矩阵)"""

    def __init__(self, hamiltonian: np.ndarray, lindblad_ops: List[np.ndarray],
                 rates: Optional[List[float]] = None, config: Optional[EvolutionConfig] = None):
        self.hamiltonian = hamiltonian
        self.lindblad_ops = lindblad_ops
        self.rates = rates or [1.0] * len(lindblad_ops)
        self.config = config or EvolutionConfig()
        self.dim = hamiltonian.shape[0]

    def evolve(self, initial_rho: np.ndarray, observables: Optional[List[np.ndarray]] = None) -> SimulationResult:
        """Lindblad 演化密度矩阵"""
        dt = self.config.dt
        num_steps = int(self.config.total_time / dt)
        times = np.linspace(0, self.config.total_time, num_steps + 1)
        rho = initial_rho.copy().astype(complex)
        states = [rho.copy()]
        expectations: Dict[str, np.ndarray] = {}
        if observables:
            for i, obs in enumerate(observables):
                expectations[f'obs_{i}'] = np.zeros(num_steps + 1)
                expectations[f'obs_{i}'][0] = float(np.real(np.trace(obs @ rho)))
        for step in range(1, num_steps + 1):
            rho = self._rk4_step(rho, dt)
            states.append(rho.copy())
            if observables and step % self.config.observable_interval == 0:
                for i, obs in enumerate(observables):
                    expectations[f'obs_{i}'][step] = float(np.real(np.trace(obs @ rho)))
        return SimulationResult(
            times=times, states=states, expectations=expectations,
            num_steps=num_steps, total_time=self.config.total_time
        )

    def _lindblad_rhs(self, rho: np.ndarray) -> np.ndarray:
        """Lindblad 方程右端: dρ/dt = -i[H,ρ] + Σ γₖ(LₖρLₖ† - ½{Lₖ†Lₖ, ρ})"""
        commutator = -1j * (self.hamiltonian @ rho - rho @ self.hamiltonian)
        dissipation = np.zeros_like(rho)
        for L, gamma in zip(self.lindblad_ops, self.rates):
            L_dag = L.conj().T
            dissipation += gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
        return commutator + dissipation

    def _rk4_step(self, rho: np.ndarray, dt: float) -> np.ndarray:
        """四阶 Runge-Kutta 积分"""
        k1 = self._lindblad_rhs(rho)
        k2 = self._lindblad_rhs(rho + dt / 2 * k1)
        k3 = self._lindblad_rhs(rho + dt / 2 * k2)
        k4 = self._lindblad_rhs(rho + dt * k3)
        return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class MonteCarloWaveFunction:
    """Monte Carlo 波函数方法 (量子跳跃)"""

    def __init__(self, hamiltonian: np.ndarray, jump_ops: List[np.ndarray],
                 rates: Optional[List[float]] = None, seed: Optional[int] = None):
        self.hamiltonian = hamiltonian
        self.jump_ops = jump_ops
        self.rates = rates or [1.0] * len(jump_ops)
        self.rng = np.random.RandomState(seed)
        self.dim = hamiltonian.shape[0]

    def simulate_single(self, initial_state: np.ndarray, dt: float, num_steps: int) -> List[np.ndarray]:
        """单次 Monte Carlo 轨迹"""
        state = initial_state.copy().astype(complex)
        trajectory = [state.copy()]
        H_eff = self._effective_hamiltonian()
        for _ in range(num_steps):
            # 非厄米演化
            state = self._non_hermitian_evolve(state, H_eff, dt)
            norm_sq = float(np.real(state.conj() @ state))
            # 量子跳跃判定
            if self.rng.random() < 1 - norm_sq:
                state = self._apply_jump(state)
                state /= np.linalg.norm(state)
            else:
                state /= np.sqrt(norm_sq)
            trajectory.append(state.copy())
        return trajectory

    def simulate_ensemble(self, initial_state: np.ndarray, dt: float,
                          num_steps: int, num_trajectories: int = 100) -> np.ndarray:
        """多轨迹系综平均"""
        rho_avg = np.zeros((self.dim, self.dim), dtype=complex)
        for _ in range(num_trajectories):
            traj = self.simulate_single(initial_state, dt, num_steps)
            final = traj[-1]
            rho_avg += np.outer(final, final.conj())
        return rho_avg / num_trajectories

    def _effective_hamiltonian(self) -> np.ndarray:
        """有效非厄米哈密顿量"""
        H_eff = self.hamiltonian.copy().astype(complex)
        for L, gamma in zip(self.jump_ops, self.rates):
            H_eff -= 1j * gamma / 2 * L.conj().T @ L
        return H_eff

    def _non_hermitian_evolve(self, state: np.ndarray, H_eff: np.ndarray, dt: float) -> np.ndarray:
        """非厄米一阶演化"""
        return state - 1j * dt * H_eff @ state

    def _apply_jump(self, state: np.ndarray) -> np.ndarray:
        """应用量子跳跃"""
        probabilities = []
        new_states = []
        for L, gamma in zip(self.jump_ops, self.rates):
            new_s = np.sqrt(gamma) * L @ state
            prob = float(np.real(new_s.conj() @ new_s))
            probabilities.append(prob)
            new_states.append(new_s)
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        idx = self.rng.choice(len(probabilities), p=probabilities)
        return new_states[idx]


class AdiabaticEvolution:
    """绝热量子演化"""

    def __init__(self, h_initial: np.ndarray, h_final: np.ndarray, config: Optional[EvolutionConfig] = None):
        self.h_initial = h_initial
        self.h_final = h_final
        self.config = config or EvolutionConfig()
        self.dim = h_initial.shape[0]

    def evolve(self, initial_state: np.ndarray) -> SimulationResult:
        """绝热演化"""
        dt = self.config.dt
        num_steps = int(self.config.total_time / dt)
        times = np.linspace(0, self.config.total_time, num_steps + 1)
        state = initial_state.copy().astype(complex)
        states = [state.copy()]
        energies = np.zeros(num_steps + 1)
        energies[0] = self._energy(state, 0.0)
        for step in range(1, num_steps + 1):
            t = step * dt
            s = t / self.config.total_time
            H = self._schedule_hamiltonian(s)
            U = self._compute_propagator(H, dt)
            state = U @ state
            state /= np.linalg.norm(state)
            states.append(state.copy())
            energies[step] = self._energy(state, s)
        return SimulationResult(
            times=times, states=states,
            expectations={'energy': energies},
            num_steps=num_steps, total_time=self.config.total_time
        )

    def _schedule_hamiltonian(self, s: float) -> np.ndarray:
        """线性调度: H(s) = (1-s)H_i + sH_f"""
        return (1 - s) * self.h_initial + s * self.h_final

    def _compute_propagator(self, H: np.ndarray, dt: float) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        phases = np.exp(-1j * eigenvalues * dt)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T

    def _energy(self, state: np.ndarray, s: float) -> float:
        H = self._schedule_hamiltonian(s)
        return float(np.real(state.conj() @ H @ state))

    def instantaneous_ground_state(self, s: float) -> np.ndarray:
        """瞬时基态"""
        H = self._schedule_hamiltonian(s)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvectors[:, 0]


class SimulationEngine:
    """仿真引擎 - 统一接口"""

    def __init__(self):
        self.results: List[SimulationResult] = []

    def schrodinger(self, hamiltonian: np.ndarray, initial_state: np.ndarray,
                    total_time: float = 1.0, dt: float = 0.01,
                    observables: Optional[List[np.ndarray]] = None) -> SimulationResult:
        """薛定谔方程仿真"""
        config = EvolutionConfig(dt=dt, total_time=total_time)
        evolver = SchrodingerEvolver(hamiltonian, config)
        result = evolver.evolve(initial_state, observables)
        self.results.append(result)
        return result

    def lindblad(self, hamiltonian: np.ndarray, initial_rho: np.ndarray,
                 lindblad_ops: List[np.ndarray], rates: Optional[List[float]] = None,
                 total_time: float = 1.0, dt: float = 0.01) -> SimulationResult:
        """Lindblad 主方程仿真"""
        config = EvolutionConfig(dt=dt, total_time=total_time)
        evolver = LindbladEvolver(hamiltonian, lindblad_ops, rates, config)
        result = evolver.evolve(initial_rho)
        self.results.append(result)
        return result

    def adiabatic(self, h_initial: np.ndarray, h_final: np.ndarray,
                  initial_state: np.ndarray, total_time: float = 10.0,
                  dt: float = 0.01) -> SimulationResult:
        """绝热演化仿真"""
        config = EvolutionConfig(dt=dt, total_time=total_time)
        evolver = AdiabaticEvolution(h_initial, h_final, config)
        result = evolver.evolve(initial_state)
        self.results.append(result)
        return result

    def get_history(self) -> List[Dict]:
        return [r.to_dict() for r in self.results]
