"""纠缠度量与分析"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math


@dataclass
class EntanglementMeasure:
    """纠缠度量结果"""
    concurrence: float = 0.0
    negativity: float = 0.0
    entanglement_entropy: float = 0.0
    bell_fidelity: float = 0.0
    mutual_information: float = 0.0
    log_negativity: float = 0.0
    formation_entropy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'concurrence': self.concurrence,
            'negativity': self.negativity,
            'entanglement_entropy': self.entanglement_entropy,
            'bell_fidelity': self.bell_fidelity,
            'mutual_information': self.mutual_information,
            'log_negativity': self.log_negativity
        }

    def is_entangled(self, threshold: float = 0.01) -> bool:
        return self.concurrence > threshold or self.negativity > threshold

    def entanglement_class(self) -> str:
        if self.concurrence > 0.99:
            return "maximally_entangled"
        elif self.concurrence > 0.5:
            return "highly_entangled"
        elif self.concurrence > 0.01:
            return "partially_entangled"
        return "separable"


class ConcurrenceCalculator:
    """并发度计算器"""

    def calculate_2qubit(self, state_vector: np.ndarray) -> float:
        """两量子比特态的并发度"""
        if len(state_vector) != 4:
            raise ValueError("并发度仅适用于两量子比特态")
        sy = np.array([[0, -1j], [1j, 0]])
        sigma_yy = np.kron(sy, sy)
        psi_tilde = sigma_yy @ state_vector.conj()
        overlap = np.abs(np.vdot(state_vector, psi_tilde))
        return float(max(0, 2 * overlap - 1))

    def calculate_density_matrix(self, rho: np.ndarray) -> float:
        """密度矩阵的并发度"""
        n = int(np.log2(rho.shape[0]))
        if n != 2:
            raise ValueError("仅支持两量子比特")
        sy = np.array([[0, -1j], [1j, 0]])
        sigma_yy = np.kron(sy, sy)
        R = rho @ sigma_yy @ rho.conj() @ sigma_yy
        eigenvalues = np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0))
        eigenvalues = sorted(eigenvalues, reverse=True)
        return float(max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3]))


class NegativityCalculator:
    """负性计算器"""

    def calculate(self, rho: np.ndarray, dim_a: int, dim_b: int) -> float:
        """计算负性"""
        rho_pt = self._partial_transpose(rho, dim_a, dim_b)
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        return float(max(0, -np.sum(eigenvalues[eigenvalues < 0])))

    def log_negativity(self, rho: np.ndarray, dim_a: int, dim_b: int) -> float:
        """对数负性"""
        neg = self.calculate(rho, dim_a, dim_b)
        return float(np.log2(2 * neg + 1)) if neg > 0 else 0.0

    def _partial_transpose(self, rho: np.ndarray, dim_a: int, dim_b: int) -> np.ndarray:
        """部分转置"""
        rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
        pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
        return pt.reshape(dim_a * dim_b, dim_a * dim_b)


class EntanglementEntropyCalculator:
    """纠缠熵计算器"""

    def von_neumann(self, rho: np.ndarray) -> float:
        """冯诺依曼熵"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def renyi(self, rho: np.ndarray, alpha: float = 2.0) -> float:
        """Rényi 熵"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if alpha == 1.0:
            return self.von_neumann(rho)
        return float(1 / (1 - alpha) * np.log2(np.sum(eigenvalues**alpha)))

    def tsallis(self, rho: np.ndarray, q: float = 2.0) -> float:
        """Tsallis 熵"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if q == 1.0:
            return self.von_neumann(rho)
        return float((1 - np.sum(eigenvalues**q)) / (q - 1))

    def mutual_information(self, rho_ab: np.ndarray, rho_a: np.ndarray, rho_b: np.ndarray) -> float:
        """互信息"""
        s_ab = self.von_neumann(rho_ab)
        s_a = self.von_neumann(rho_a)
        s_b = self.von_neumann(rho_b)
        return float(s_a + s_b - s_ab)


class BellStateAnalyzer:
    """Bell 态分析器"""

    BELL_STATES = {
        'phi_plus': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        'phi_minus': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        'psi_plus': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        'psi_minus': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }

    def fidelity_to_bell(self, state: np.ndarray, bell_name: str = 'phi_plus') -> float:
        """计算与 Bell 态的保真度"""
        bell = self.BELL_STATES[bell_name]
        return float(np.abs(np.vdot(state, bell))**2)

    def best_bell_match(self, state: np.ndarray) -> Tuple[str, float]:
        """找到最佳匹配的 Bell 态"""
        best_name = None
        best_fidelity = 0.0
        for name, bell in self.BELL_STATES.items():
            f = float(np.abs(np.vdot(state, bell))**2)
            if f > best_fidelity:
                best_fidelity = f
                best_name = name
        return best_name, best_fidelity

    def bell_inequality(self, correlations: Dict[str, float]) -> float:
        """CHSH 不等式: S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|"""
        E_ab = correlations.get('ab', 0)
        E_ab_prime = correlations.get('ab_prime', 0)
        E_a_prime_b = correlations.get('a_prime_b', 0)
        E_a_prime_b_prime = correlations.get('a_prime_b_prime', 0)
        return abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)


class SchmidtDecomposition:
    """Schmidt 分解"""

    def decompose(self, state: np.ndarray, dim_a: int, dim_b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Schmidt 分解: |ψ⟩ = Σ λᵢ |aᵢ⟩|bᵢ⟩"""
        matrix = state.reshape(dim_a, dim_b)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        # 归一化 Schmidt 系数
        S = S / np.linalg.norm(S)
        return U, S, Vh

    def schmidt_rank(self, state: np.ndarray, dim_a: int, dim_b: int) -> int:
        """Schmidt 秩"""
        _, S, _ = self.decompose(state, dim_a, dim_b)
        return int(np.sum(S > 1e-10))

    def schmidt_coefficients(self, state: np.ndarray, dim_a: int, dim_b: int) -> np.ndarray:
        """Schmidt 系数"""
        _, S, _ = self.decompose(state, dim_a, dim_b)
        return S


class EntanglementWitness:
    """纠缠见证算子"""

    def __init__(self, witness_matrix: np.ndarray):
        self.W = witness_matrix

    def expectation_value(self, rho: np.ndarray) -> float:
        """期望值 ⟨W⟩"""
        return float(np.real(np.trace(self.W @ rho)))

    def is_entangled(self, rho: np.ndarray) -> bool:
        """是否纠缠"""
        return self.expectation_value(rho) < 0

    @classmethod
    def for_bell_state(cls, bell_type: str = 'phi_plus') -> 'EntanglementWitness':
        """为 Bell 态构造见证算子"""
        bell = BellStateAnalyzer.BELL_STATES[bell_type]
        rho_bell = np.outer(bell, bell.conj())
        W = 0.5 * np.eye(4, dtype=complex) - rho_bell
        return cls(W)


class EntanglementAnalyzer:
    """纠缠分析引擎 - 统一接口"""

    def __init__(self):
        self.concurrence_calc = ConcurrenceCalculator()
        self.negativity_calc = NegativityCalculator()
        self.entropy_calc = EntanglementEntropyCalculator()
        self.bell_analyzer = BellStateAnalyzer()
        self.schmidt = SchmidtDecomposition()

    def analyze_state_vector(self, state: np.ndarray, num_qubits: int, partition: Optional[List[int]] = None) -> EntanglementMeasure:
        """分析态矢量的纠缠"""
        measure = EntanglementMeasure()
        if num_qubits == 2:
            measure.concurrence = self.concurrence_calc.calculate_2qubit(state)
            rho = np.outer(state, state.conj())
            measure.negativity = self.negativity_calc.calculate(rho, 2, 2)
            measure.log_negativity = self.negativity_calc.log_negativity(rho, 2, 2)
            rho_a = self._partial_trace(rho, 2, 2, 'A')
            measure.entanglement_entropy = self.entropy_calc.von_neumann(rho_a)
            name, f = self.bell_analyzer.best_bell_match(state)
            measure.bell_fidelity = f
            measure.mutual_information = self._compute_mutual_info(state, num_qubits)
        else:
            rho = np.outer(state, state.conj())
            if partition:
                dim_a = 2**len(partition)
                dim_b = 2**(num_qubits - len(partition))
                rho_a = self._partial_trace_partition(rho, num_qubits, partition)
                measure.entanglement_entropy = self.entropy_calc.von_neumann(rho_a)
                measure.negativity = self.negativity_calc.calculate(rho, dim_a, dim_b)
            else:
                dim_half = num_qubits // 2
                rho_a = self._partial_trace_partition(rho, num_qubits, list(range(dim_half)))
                measure.entanglement_entropy = self.entropy_calc.von_neumann(rho_a)
        return measure

    def analyze_density_matrix(self, rho: np.ndarray) -> EntanglementMeasure:
        """分析密度矩阵的纠缠"""
        measure = EntanglementMeasure()
        n = int(np.log2(rho.shape[0]))
        if n == 2:
            measure.concurrence = self.concurrence_calc.calculate_density_matrix(rho)
            measure.negativity = self.negativity_calc.calculate(rho, 2, 2)
        measure.entanglement_entropy = self.entropy_calc.von_neumann(rho)
        return measure

    def _partial_trace(self, rho: np.ndarray, dim_a: int, dim_b: int, trace_over: str) -> np.ndarray:
        """部分迹"""
        rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
        if trace_over == 'A':
            return np.trace(rho_reshaped, axis1=0, axis2=2)
        else:
            return np.trace(rho_reshaped, axis1=1, axis2=3)

    def _partial_trace_partition(self, rho: np.ndarray, num_qubits: int, keep: List[int]) -> np.ndarray:
        """按分区部分迹"""
        dim_keep = 2**len(keep)
        dim_trace = 2**(num_qubits - len(keep))
        rho_reshaped = rho.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
        return np.trace(rho_reshaped, axis1=1, axis2=3)

    def _compute_mutual_info(self, state: np.ndarray, num_qubits: int) -> float:
        """计算互信息"""
        rho = np.outer(state, state.conj())
        rho_a = self._partial_trace(rho, 2, 2, 'A')
        rho_b = self._partial_trace(rho, 2, 2, 'B')
        return self.entropy_calc.mutual_information(rho, rho_a, rho_b)

    def classify_entanglement(self, measure: EntanglementMeasure) -> str:
        """分类纠缠程度"""
        return measure.entanglement_class()
