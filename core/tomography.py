"""量子态层析 - 最大似然估计/贝叶斯重建/压缩感知"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math


@dataclass
class TomographyResult:
    """层析重建结果"""
    rho: np.ndarray
    fidelity: float = 0.0
    purity: float = 0.0
    iterations: int = 0
    convergence_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fidelity': self.fidelity,
            'purity': self.purity,
            'iterations': self.iterations,
            'convergence_error': self.convergence_error,
            'rho_trace': float(np.real(np.trace(self.rho)))
        }


@dataclass
class MeasurementOutcome:
    """单次测量结果"""
    basis: str
    outcome: int
    probability: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)

    def expectation(self) -> float:
        """计算期望值"""
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        return sum((-1)**int(k) * v for k, v in self.counts.items()) / total


class PauliMeasurementBasis:
    """Pauli 测量基生成器"""

    PAULI_I = np.eye(2, dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    PAULIS = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
    PAULI_NAMES = ['I', 'X', 'Y', 'Z']

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

    def generate_pauli_strings(self) -> List[Tuple[str, np.ndarray]]:
        """生成所有 Pauli 字符串"""
        if self.num_qubits == 1:
            return [(n, p) for n, p in zip(self.PAULI_NAMES[1:], self.PAULIS[1:])]
        result = []
        for idx in range(4 ** self.num_qubits):
            name_parts = []
            op = np.array([[1.0]], dtype=complex)
            temp = idx
            for _ in range(self.num_qubits):
                p_idx = temp % 4
                temp //= 4
                name_parts.append(self.PAULI_NAMES[p_idx])
                op = np.kron(op, self.PAULIS[p_idx])
            name = ''.join(reversed(name_parts))
            if not all(c == 'I' for c in name):
                result.append((name, op))
        return result

    def eigenvectors(self, pauli_name: str) -> np.ndarray:
        """获取 Pauli 算子的本征矢"""
        eigvecs = {
            'X': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            'Y': np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2),
            'Z': np.eye(2, dtype=complex)
        }
        result = np.array([[1.0]], dtype=complex)
        for ch in pauli_name:
            result = np.kron(result, eigvecs.get(ch, np.eye(2, dtype=complex)))
        return result


class MaximumLikelihoodEstimation:
    """最大似然估计层析"""

    def __init__(self, num_qubits: int, max_iter: int = 500, tol: float = 1e-8):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.max_iter = max_iter
        self.tol = tol

    def reconstruct(self, measurements: Dict[str, Dict[str, int]]) -> TomographyResult:
        """从测量数据重建密度矩阵"""
        # 初始化为最大混合态
        rho = np.eye(self.dim, dtype=complex) / self.dim
        log_likelihood_old = self._log_likelihood(rho, measurements)
        for iteration in range(self.max_iter):
            grad = self._gradient(rho, measurements)
            # 梯度上升 + 重投影
            rho_new = rho + 0.01 * grad
            rho_new = self._project_to_valid(rho_new)
            log_likelihood_new = self._log_likelihood(rho_new, measurements)
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                rho = rho_new
                return TomographyResult(
                    rho=rho, purity=float(np.real(np.trace(rho @ rho))),
                    iterations=iteration + 1,
                    convergence_error=abs(log_likelihood_new - log_likelihood_old)
                )
            rho = rho_new
            log_likelihood_old = log_likelihood_new
        return TomographyResult(
            rho=rho, purity=float(np.real(np.trace(rho @ rho))),
            iterations=self.max_iter, convergence_error=self.tol * 10
        )

    def _log_likelihood(self, rho: np.ndarray, measurements: Dict[str, Dict[str, int]]) -> float:
        """计算对数似然"""
        ll = 0.0
        for basis_name, counts in measurements.items():
            total = sum(counts.values())
            if total == 0:
                continue
            for outcome_str, count in counts.items():
                idx = int(outcome_str, 2) if len(outcome_str) > 1 else int(outcome_str)
                if idx < self.dim:
                    p = max(np.real(rho[idx, idx]), 1e-15)
                    ll += count * np.log(p)
        return float(ll)

    def _gradient(self, rho: np.ndarray, measurements: Dict[str, Dict[str, int]]) -> np.ndarray:
        """计算似然梯度"""
        grad = np.zeros_like(rho)
        for basis_name, counts in measurements.items():
            total = sum(counts.values())
            if total == 0:
                continue
            for outcome_str, count in counts.items():
                idx = int(outcome_str, 2) if len(outcome_str) > 1 else int(outcome_str)
                if idx < self.dim:
                    p = max(np.real(rho[idx, idx]), 1e-15)
                    grad[idx, idx] += count / p
        return grad

    def _project_to_valid(self, rho: np.ndarray) -> np.ndarray:
        """投影到有效密度矩阵空间"""
        rho = (rho + rho.conj().T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)
        total = np.sum(eigenvalues)
        if total > 0:
            eigenvalues /= total
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        return rho


class BayesianTomography:
    """贝叶斯量子态层析"""

    def __init__(self, num_qubits: int, num_samples: int = 1000, seed: Optional[int] = None):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)

    def reconstruct(self, measurements: Dict[str, Dict[str, int]]) -> TomographyResult:
        """贝叶斯后验采样重建"""
        samples = []
        log_weights = []
        for _ in range(self.num_samples):
            rho = self._sample_random_density_matrix()
            ll = self._log_likelihood(rho, measurements)
            samples.append(rho)
            log_weights.append(ll)
        log_weights = np.array(log_weights)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        rho_mean = np.zeros((self.dim, self.dim), dtype=complex)
        for w, rho in zip(weights, samples):
            rho_mean += w * rho
        purity = float(np.real(np.trace(rho_mean @ rho_mean)))
        return TomographyResult(
            rho=rho_mean, purity=purity,
            iterations=self.num_samples,
            metadata={'effective_samples': float(1.0 / np.sum(weights**2))}
        )

    def _sample_random_density_matrix(self) -> np.ndarray:
        """随机采样密度矩阵 (Hilbert-Schmidt 均匀)"""
        A = self.rng.randn(self.dim, self.dim) + 1j * self.rng.randn(self.dim, self.dim)
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        return rho

    def _log_likelihood(self, rho: np.ndarray, measurements: Dict[str, Dict[str, int]]) -> float:
        """对数似然"""
        ll = 0.0
        for _, counts in measurements.items():
            for outcome_str, count in counts.items():
                idx = int(outcome_str, 2) if len(outcome_str) > 1 else int(outcome_str)
                if idx < self.dim:
                    p = max(np.real(rho[idx, idx]), 1e-15)
                    ll += count * np.log(p)
        return float(ll)


class CompressedSensingTomography:
    """压缩感知量子态层析"""

    def __init__(self, num_qubits: int, rank: int = 2, max_iter: int = 200):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.rank = rank
        self.max_iter = max_iter

    def reconstruct(self, observable_expvals: List[Tuple[np.ndarray, float]]) -> TomographyResult:
        """从算符期望值重建低秩密度矩阵"""
        # 初始化因子矩阵
        M = self._initialize_factor()
        for iteration in range(self.max_iter):
            grad = self._compute_gradient(M, observable_expvals)
            lr = 0.001 / (1 + iteration * 0.01)
            M -= lr * grad
        rho = M @ M.conj().T
        rho /= max(np.real(np.trace(rho)), 1e-15)
        err = self._reconstruction_error(rho, observable_expvals)
        return TomographyResult(
            rho=rho, purity=float(np.real(np.trace(rho @ rho))),
            iterations=self.max_iter, convergence_error=err
        )

    def _initialize_factor(self) -> np.ndarray:
        """初始化因子矩阵"""
        rng = np.random.RandomState(42)
        return rng.randn(self.dim, self.rank) + 1j * rng.randn(self.dim, self.rank)

    def _compute_gradient(self, M: np.ndarray, obs_expvals: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """计算因子矩阵梯度"""
        grad = np.zeros_like(M)
        rho = M @ M.conj().T
        for obs, exp_val in obs_expvals:
            diff = np.real(np.trace(obs @ rho)) - exp_val
            grad += 2 * diff * (obs @ M + obs.conj().T @ M)
        return grad

    def _reconstruction_error(self, rho: np.ndarray, obs_expvals: List[Tuple[np.ndarray, float]]) -> float:
        """重建误差"""
        total_err = 0.0
        for obs, exp_val in obs_expvals:
            total_err += (np.real(np.trace(obs @ rho)) - exp_val) ** 2
        return float(np.sqrt(total_err / max(len(obs_expvals), 1)))


class TomographyAnalyzer:
    """层析分析引擎"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.basis_gen = PauliMeasurementBasis(num_qubits)

    def simulate_tomography(self, true_rho: np.ndarray, shots: int = 4096, seed: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """模拟层析测量过程"""
        rng = np.random.RandomState(seed)
        measurements = {}
        pauli_strings = self.basis_gen.generate_pauli_strings()
        for name, op in pauli_strings[:min(6, len(pauli_strings))]:
            eigvals, eigvecs = np.linalg.eigh(op)
            probs = np.zeros(len(eigvals))
            for i, (ev, vec) in enumerate(zip(eigvals, eigvecs.T)):
                probs[i] = max(np.real(vec.conj() @ true_rho @ vec), 0)
            probs /= max(np.sum(probs), 1e-15)
            outcomes = rng.choice(len(probs), size=shots, p=probs)
            counts = {}
            for o in outcomes:
                key = format(o, f'0{self.num_qubits}b')
                counts[key] = counts.get(key, 0) + 1
            measurements[name] = counts
        return measurements

    def compute_tomographic_fidelity(self, rho_est: np.ndarray, rho_true: np.ndarray) -> float:
        """计算层析保真度"""
        sqrt_rho = self._matrix_sqrt(rho_true)
        inner = sqrt_rho @ rho_est @ sqrt_rho
        return float(np.real(np.trace(self._matrix_sqrt(inner))) ** 2)

    def process_tomography(self, channel_fn, input_states: List[np.ndarray], shots: int = 1024) -> np.ndarray:
        """量子过程层析"""
        chi_dim = self.dim ** 2
        chi = np.zeros((chi_dim, chi_dim), dtype=complex)
        for rho_in in input_states:
            rho_out = channel_fn(rho_in)
            chi += np.outer(rho_out.flatten(), rho_in.flatten().conj())
        chi /= len(input_states)
        return chi

    def _matrix_sqrt(self, m: np.ndarray) -> np.ndarray:
        """矩阵平方根"""
        eigenvalues, eigenvectors = np.linalg.eigh(m)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
