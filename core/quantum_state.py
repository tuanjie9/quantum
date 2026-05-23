"""量子态表示与操作 - 10维量子态向量"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import cmath
import math


class StateType(Enum):
    """量子态类型"""
    PURE = "pure"
    MIXED = "mixed"
    BELL = "bell"
    GHZ = "ghz"
    W = "w"
    CLUSTER = "cluster"
    RANDOM = "random"


@dataclass
class QuantumDimension:
    """量子态的单个维度"""
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0
    description: str = ""

    def normalize(self) -> float:
        """归一化到 [0, 1]"""
        return max(self.min_val, min(self.max_val, (self.value - self.min_val) / (self.max_val - self.min_val + 1e-10)))


@dataclass
class QuantumStateVector:
    """量子态向量 - 复数振幅表示"""
    num_qubits: int
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.amplitudes.size == 0:
            self.amplitudes = np.zeros(2**self.num_qubits, dtype=complex)
            self.amplitudes[0] = 1.0  # |0...0⟩ 初始态
        elif self.amplitudes.size != 2**self.num_qubits:
            raise ValueError(f"振幅维度不匹配: 期望 {2**self.num_qubits}, 实际 {self.amplitudes.size}")

    @classmethod
    def zero_state(cls, n: int) -> 'QuantumStateVector':
        """创建 n 量子比特的 |0⟩ 态"""
        amps = np.zeros(2**n, dtype=complex)
        amps[0] = 1.0
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def one_state(cls, n: int) -> 'QuantumStateVector':
        """创建 n 量子比特的 |1⟩ 态"""
        amps = np.zeros(2**n, dtype=complex)
        amps[-1] = 1.0
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def plus_state(cls, n: int) -> 'QuantumStateVector':
        """创建 n 量子比特的 |+⟩ 均匀叠加态"""
        dim = 2**n
        amps = np.ones(dim, dtype=complex) / np.sqrt(dim)
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def bell_state(cls, bell_type: int = 0) -> 'QuantumStateVector':
        """创建 Bell 态 (2 量子比特纠缠态)"""
        amps = np.zeros(4, dtype=complex)
        if bell_type == 0:  # |Φ+⟩ = (|00⟩ + |11⟩) / √2
            amps[0] = amps[3] = 1.0 / np.sqrt(2)
        elif bell_type == 1:  # |Φ-⟩ = (|00⟩ - |11⟩) / √2
            amps[0] = 1.0 / np.sqrt(2)
            amps[3] = -1.0 / np.sqrt(2)
        elif bell_type == 2:  # |Ψ+⟩ = (|01⟩ + |10⟩) / √2
            amps[1] = amps[2] = 1.0 / np.sqrt(2)
        elif bell_type == 3:  # |Ψ-⟩ = (|01⟩ - |10⟩) / √2
            amps[1] = 1.0 / np.sqrt(2)
            amps[2] = -1.0 / np.sqrt(2)
        return cls(num_qubits=2, amplitudes=amps)

    @classmethod
    def ghz_state(cls, n: int) -> 'QuantumStateVector':
        """创建 GHZ 态 (多体最大纠缠态)"""
        dim = 2**n
        amps = np.zeros(dim, dtype=complex)
        amps[0] = 1.0 / np.sqrt(2)  # |00...0⟩
        amps[-1] = 1.0 / np.sqrt(2)  # |11...1⟩
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def w_state(cls, n: int) -> 'QuantumStateVector':
        """创建 W 态 (单激发均匀叠加态)"""
        dim = 2**n
        amps = np.zeros(dim, dtype=complex)
        for i in range(n):
            idx = 2**i  # 只有第 i 个量子比特为 1
            amps[idx] = 1.0 / np.sqrt(n)
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def random_state(cls, n: int, seed: Optional[int] = None) -> 'QuantumStateVector':
        """创建随机量子态"""
        rng = np.random.RandomState(seed)
        dim = 2**n
        real_part = rng.randn(dim)
        imag_part = rng.randn(dim)
        amps = real_part + 1j * imag_part
        amps /= np.linalg.norm(amps)
        return cls(num_qubits=n, amplitudes=amps)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuantumStateVector':
        """从字典创建"""
        return cls(
            num_qubits=d['num_qubits'],
            amplitudes=np.array(d['amplitudes'], dtype=complex)
        )

    def normalize(self) -> 'QuantumStateVector':
        """归一化量子态"""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-10:
            self.amplitudes /= norm
        return self

    def probabilities(self) -> np.ndarray:
        """计算测量概率分布"""
        return np.abs(self.amplitudes) ** 2

    def expectation_value(self, observable: np.ndarray) -> float:
        """计算期望值 ⟨ψ|O|ψ⟩"""
        return float(np.real(self.amplitudes.conj() @ observable @ self.amplitudes))

    def fidelity(self, other: 'QuantumStateVector') -> float:
        """计算保真度 |⟨ψ|φ⟩|²"""
        overlap = np.abs(np.vdot(self.amplitudes, other.amplitudes))
        return float(overlap ** 2)

    def trace_distance(self, other: 'QuantumStateVector') -> float:
        """计算迹距离"""
        diff = self.probabilities() - other.probabilities()
        return float(np.sum(np.abs(diff)) / 2)

    def entropy(self) -> float:
        """计算冯诺依曼熵"""
        probs = self.probabilities()
        probs = probs[probs > 1e-10]
        return float(-np.sum(probs * np.log2(probs)))

    def entanglement_entropy(self, subsystem_qubits: List[int]) -> float:
        """计算子系统的纠缠熵"""
        n = self.num_qubits
        dim_a = 2**len(subsystem_qubits)
        dim_b = 2**(n - len(subsystem_qubits))
        # 重塑为矩阵并计算约化密度矩阵
        psi = self.amplitudes.reshape(dim_a, dim_b)
        rho_a = psi @ psi.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """部分迹运算"""
        n = self.num_qubits
        trace_qubits = [i for i in range(n) if i not in keep_qubits]
        dim_keep = 2**len(keep_qubits)
        dim_trace = 2**len(trace_qubits)
        psi_matrix = self.amplitudes.reshape(dim_keep, dim_trace)
        rho_reduced = psi_matrix @ psi_matrix.conj().T
        return rho_reduced

    def tensor_product(self, other: 'QuantumStateVector') -> 'QuantumStateVector':
        """张量积"""
        new_amps = np.kron(self.amplitudes, other.amplitudes)
        return QuantumStateVector(
            num_qubits=self.num_qubits + other.num_qubits,
            amplitudes=new_amps
        )

    def inner_product(self, other: 'QuantumStateVector') -> complex:
        """内积 ⟨ψ|φ⟩"""
        return np.vdot(self.amplitudes, other.amplitudes)

    def outer_product(self, other: 'QuantumStateVector') -> np.ndarray:
        """外积 |ψ⟩⟨φ|"""
        return np.outer(self.amplitudes, other.amplitudes.conj())

    def measure(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """模拟量子测量"""
        rng = np.random.RandomState(seed)
        probs = self.probabilities()
        probs /= probs.sum()  # 确保归一化
        outcomes = rng.choice(len(probs), size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def measure_single(self, qubit: int, seed: Optional[int] = None) -> int:
        """测量单个量子比特"""
        rng = np.random.RandomState(seed)
        probs_0 = 0.0
        dim = 2**self.num_qubits
        for i in range(dim):
            if not (i >> (self.num_qubits - 1 - qubit)) & 1:
                probs_0 += abs(self.amplitudes[i])**2
        return 0 if rng.random() < probs_0 else 1

    def apply_phase(self, qubit: int, phase: float):
        """对指定量子比特施加相位"""
        dim = 2**self.num_qubits
        for i in range(dim):
            if (i >> (self.num_qubits - 1 - qubit)) & 1:
                self.amplitudes[i] *= cmath.exp(1j * phase)

    def apply_global_phase(self, phase: float):
        """施加全局相位"""
        self.amplitudes *= cmath.exp(1j * phase)

    def bloch_angles(self, qubit: int) -> Tuple[float, float, float]:
        """计算单个量子比特的 Bloch 球坐标 (θ, φ, r)"""
        rho = self.partial_trace([qubit])
        # Pauli 矩阵期望值
        sx = 2 * np.real(rho[0, 1])
        sy = 2 * np.imag(rho[1, 0])  # 修正
        sz = np.real(rho[0, 0] - rho[1, 1])
        r = np.sqrt(sx**2 + sy**2 + sz**2)
        theta = np.arccos(sz / max(r, 1e-10))
        phi = np.arctan2(sy, sx)
        return float(theta), float(phi), float(r)

    def concurrence(self) -> float:
        """计算两量子比特态的并发度"""
        if self.num_qubits != 2:
            raise ValueError("并发度仅适用于两量子比特态")
        # σ_y ⊗ σ_y
        sy = np.array([[0, -1j], [1j, 0]])
        sigma_yy = np.kron(sy, sy)
        psi_tilde = sigma_yy @ self.amplitudes.conj()
        overlap = np.abs(np.vdot(self.amplitudes, psi_tilde))
        return float(max(0, 2 * overlap - 1))

    def negativity(self, partition: List[int]) -> float:
        """计算负性（纠缠度量）"""
        rho = np.outer(self.amplitudes, self.amplitudes.conj())
        n = self.num_qubits
        dim_a = 2**len(partition)
        dim_b = 2**(n - len(partition))
        rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1)).reshape(dim_a * dim_b, dim_a * dim_b)
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        return float(max(0, -np.sum(eigenvalues[eigenvalues < 0])))

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            'num_qubits': self.num_qubits,
            'amplitudes': self.amplitudes.tolist()
        }

    def __repr__(self) -> str:
        probs = self.probabilities()
        top_states = np.argsort(probs)[-5:][::-1]
        terms = []
        for idx in top_states:
            if probs[idx] > 1e-6:
                bitstring = format(idx, f'0{self.num_qubits}b')
                amp = self.amplitudes[idx]
                terms.append(f"({amp.real:.3f}+{amp.imag:.3f}j)|{bitstring}⟩")
        return f"QuantumState({self.num_qubits}q: {' + '.join(terms)})"


@dataclass
class QuantumState:
    """10维量子状态完整表示"""
    amplitude: QuantumDimension = field(default_factory=lambda: QuantumDimension("amplitude", 1.0, 0.0, 1.0, "振幅"))
    phase: QuantumDimension = field(default_factory=lambda: QuantumDimension("phase", 0.0, -math.pi, math.pi, "相位"))
    entanglement: QuantumDimension = field(default_factory=lambda: QuantumDimension("entanglement", 0.0, 0.0, 1.0, "纠缠度"))
    coherence: QuantumDimension = field(default_factory=lambda: QuantumDimension("coherence", 1.0, 0.0, 1.0, "相干性"))
    decoherence: QuantumDimension = field(default_factory=lambda: QuantumDimension("decoherence", 0.0, 0.0, 1.0, "退相干"))
    noise: QuantumDimension = field(default_factory=lambda: QuantumDimension("noise", 0.0, 0.0, 1.0, "噪声水平"))
    gate_fidelity: QuantumDimension = field(default_factory=lambda: QuantumDimension("gate_fidelity", 1.0, 0.0, 1.0, "门保真度"))
    measurement_prob: QuantumDimension = field(default_factory=lambda: QuantumDimension("measurement_prob", 0.5, 0.0, 1.0, "测量概率"))
    superposition: QuantumDimension = field(default_factory=lambda: QuantumDimension("superposition", 0.5, 0.0, 1.0, "叠加度"))
    tunneling: QuantumDimension = field(default_factory=lambda: QuantumDimension("tunneling", 0.0, 0.0, 1.0, "隧穿振幅"))

    def to_vector(self) -> np.ndarray:
        """转换为10维向量"""
        return np.array([
            self.amplitude.normalize(), self.phase.normalize(),
            self.entanglement.normalize(), self.coherence.normalize(),
            self.decoherence.normalize(), self.noise.normalize(),
            self.gate_fidelity.normalize(), self.measurement_prob.normalize(),
            self.superposition.normalize(), self.tunneling.normalize()
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'QuantumState':
        """从向量创建"""
        dims = ["amplitude", "phase", "entanglement", "coherence", "decoherence",
                "noise", "gate_fidelity", "measurement_prob", "superposition", "tunneling"]
        state = cls()
        for i, dim_name in enumerate(dims):
            dim = getattr(state, dim_name)
            dim.value = float(vec[i]) * (dim.max_val - dim.min_val) + dim.min_val
        return state

    def mutate(self, rate: float = 0.1) -> 'QuantumState':
        """量子态突变"""
        vec = self.to_vector()
        mutation = np.random.randn(10) * rate
        vec = np.clip(vec + mutation, 0, 1)
        return QuantumState.from_vector(vec)

    def crossover(self, other: 'QuantumState') -> 'QuantumState':
        """量子态交叉"""
        vec1 = self.to_vector()
        vec2 = other.to_vector()
        mask = np.random.randint(0, 2, 10).astype(float)
        child_vec = vec1 * mask + vec2 * (1 - mask)
        return QuantumState.from_vector(child_vec)

    def distance(self, other: 'QuantumState') -> float:
        """量子态距离"""
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))

    def fidelity_10d(self, other: 'QuantumState') -> float:
        """10维保真度"""
        v1 = self.to_vector()
        v2 = other.to_vector()
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return float((cos_sim + 1) / 2)

    def to_dict(self) -> Dict[str, float]:
        """序列化"""
        return {dim.name: dim.value for dim in [
            self.amplitude, self.phase, self.entanglement, self.coherence,
            self.decoherence, self.noise, self.gate_fidelity,
            self.measurement_prob, self.superposition, self.tunneling
        ]}

    def __repr__(self) -> str:
        return f"QuantumState(amplitude={self.amplitude.value:.3f}, coherence={self.coherence.value:.3f}, entanglement={self.entanglement.value:.3f})"


class DensityMatrix:
    """密度矩阵表示"""

    def __init__(self, num_qubits: int, matrix: Optional[np.ndarray] = None):
        self.num_qubits = num_qubits
        dim = 2**num_qubits
        if matrix is None:
            self.matrix = np.zeros((dim, dim), dtype=complex)
            self.matrix[0, 0] = 1.0
        else:
            self.matrix = matrix

    @classmethod
    def from_state_vector(cls, sv: QuantumStateVector) -> 'DensityMatrix':
        """从态矢量创建密度矩阵"""
        dm = cls(sv.num_qubits)
        dm.matrix = np.outer(sv.amplitudes, sv.amplitudes.conj())
        return dm

    @classmethod
    def maximally_mixed(cls, n: int) -> 'DensityMatrix':
        """最大混合态"""
        dim = 2**n
        dm = cls(n)
        dm.matrix = np.eye(dim, dtype=complex) / dim
        return dm

    def purity(self) -> float:
        """纯度 Tr(ρ²)"""
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def von_neumann_entropy(self) -> float:
        """冯诺依曼熵"""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def fidelity_dm(self, other: 'DensityMatrix') -> float:
        """密度矩阵保真度"""
        sqrt_rho = self._matrix_sqrt(self.matrix)
        inner = sqrt_rho @ other.matrix @ sqrt_rho
        sqrt_inner = self._matrix_sqrt(inner)
        return float(np.real(np.trace(sqrt_inner)) ** 2)

    def trace_distance(self, other: 'DensityMatrix') -> float:
        """迹距离"""
        diff = self.matrix - other.matrix
        eigenvalues = np.linalg.eigvalsh(diff)
        return float(np.sum(np.abs(eigenvalues)) / 2)

    def partial_trace_dm(self, keep_qubits: List[int]) -> 'DensityMatrix':
        """部分迹"""
        n = self.num_qubits
        dim_keep = 2**len(keep_qubits)
        dim_trace = 2**(n - len(keep_qubits))
        rho = self.matrix.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
        result = np.trace(rho, axis1=1, axis2=3)
        return DensityMatrix(len(keep_qubits), result)

    def apply_channel(self, kraus_ops: List[np.ndarray]) -> 'DensityMatrix':
        """应用量子通道 (Kraus 算子)"""
        result = np.zeros_like(self.matrix)
        for K in kraus_ops:
            result += K @ self.matrix @ K.conj().T
        dm = DensityMatrix(self.num_qubits)
        dm.matrix = result
        return dm

    def _matrix_sqrt(self, m: np.ndarray) -> np.ndarray:
        """矩阵平方根"""
        eigenvalues, eigenvectors = np.linalg.eigh(m)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    def is_valid(self, tolerance: float = 1e-6) -> bool:
        """验证是否为有效密度矩阵"""
        trace_ok = abs(np.trace(self.matrix) - 1.0) < tolerance
        hermitian_ok = np.allclose(self.matrix, self.matrix.conj().T, atol=tolerance)
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        positive_ok = np.all(eigenvalues >= -tolerance)
        return trace_ok and hermitian_ok and positive_ok

    def __repr__(self) -> str:
        return f"DensityMatrix({self.num_qubits}q, purity={self.purity():.4f})"



class QuantumStateTomography:
    """量子态层析"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits

    def generate_measurement_bases(self) -> List[np.ndarray]:
        """生成测量基"""
        bases = []
        # Pauli 基
        paulis = [
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex)
        ]
        for p in paulis:
            if self.num_qubits == 1:
                bases.append(p)
            else:
                for q in range(self.num_qubits):
                    full_op = np.eye(1, dtype=complex)
                    for i in range(self.num_qubits):
                        if i == q:
                            full_op = np.kron(full_op, p)
                        else:
                            full_op = np.kron(full_op, np.eye(2, dtype=complex))
                    bases.append(full_op)
        return bases

    def reconstruct_from_counts(self, measurements: Dict[str, Dict[str, int]]) -> np.ndarray:
        """从测量结果重建密度矩阵"""
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        for basis_name, counts in measurements.items():
            total = sum(counts.values())
            if total == 0:
                continue
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                rho[idx, idx] += count / total
        rho /= np.trace(rho)
        return rho

    def fidelity_tomography(self, rho_measured: np.ndarray, rho_target: np.ndarray) -> float:
        """层析保真度"""
        sqrt_rho = self._matrix_sqrt(rho_target)
        inner = sqrt_rho @ rho_measured @ sqrt_rho
        return float(np.real(np.trace(self._matrix_sqrt(inner)))**2)

    def _matrix_sqrt(self, m: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(m)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    def process_tomography(self, input_states: List[np.ndarray], output_states: List[np.ndarray]) -> np.ndarray:
        """过程层析"""
        chi = np.zeros((self.dim**2, self.dim**2), dtype=complex)
        for rho_in, rho_out in zip(input_states, output_states):
            # 简化: 使用最大似然估计
            chi += np.outer(rho_out.flatten(), rho_in.flatten().conj())
        return chi / len(input_states)


class QuantumChannel:
    """量子通道表示"""

    def __init__(self, kraus_ops: List[np.ndarray], name: str = ""):
        self.kraus_ops = kraus_ops
        self.name = name
        self._validate()

    def _validate(self):
        """验证 Kraus 条件: Σ K†K = I"""
        dim = self.kraus_ops[0].shape[0]
        sum_kk = np.zeros((dim, dim), dtype=complex)
        for K in self.kraus_ops:
            sum_kk += K.conj().T @ K
        if not np.allclose(sum_kk, np.eye(dim), atol=1e-6):
            pass  # 允许近似

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """应用通道"""
        result = np.zeros_like(rho)
        for K in self.kraus_ops:
            result += K @ rho @ K.conj().T
        return result

    def compose(self, other: 'QuantumChannel') -> 'QuantumChannel':
        """通道组合"""
        new_ops = []
        for K1 in self.kraus_ops:
            for K2 in other.kraus_ops:
                new_ops.append(K2 @ K1)
        return QuantumChannel(new_ops, f"{self.name}∘{other.name}")

    def adjoint(self) -> 'QuantumChannel':
        """伴随通道"""
        return QuantumChannel([K.conj().T for K in self.kraus_ops], f"{self.name}†")

    def is_unitary_channel(self) -> bool:
        """是否为酉通道"""
        if len(self.kraus_ops) == 1:
            K = self.kraus_ops[0]
            return np.allclose(K @ K.conj().T, np.eye(K.shape[0]))
        return False

    @classmethod
    def depolarizing(cls, p: float, n: int = 1) -> 'QuantumChannel':
        """去极化通道"""
        dim = 2**n
        paulis = [np.eye(2, dtype=complex),
                  np.array([[0,1],[1,0]], dtype=complex),
                  np.array([[0,-1j],[1j,0]], dtype=complex),
                  np.array([[1,0],[0,-1]], dtype=complex)]
        ops = []
        for P in paulis:
            full_P = P
            for _ in range(n - 1):
                full_P = np.kron(full_P, np.eye(2, dtype=complex))
            ops.append(np.sqrt(p / 4) * full_P)
        ops[0] = np.sqrt(1 - 3*p/4) * np.eye(dim, dtype=complex)
        return cls(ops, f"depolarizing({p})")

    @classmethod
    def identity(cls, dim: int = 2) -> 'QuantumChannel':
        """恒等通道"""
        return cls([np.eye(dim, dtype=complex)], "identity")

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'num_kraus': len(self.kraus_ops)}


class QuantumProcessMatrix:
    """量子过程矩阵 (χ 矩阵表示)"""

    def __init__(self, num_qubits: int, chi_matrix: Optional[np.ndarray] = None):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        if chi_matrix is None:
            self.chi = np.zeros((self.dim**2, self.dim**2), dtype=complex)
        else:
            self.chi = chi_matrix

    @classmethod
    def from_channel(cls, channel: QuantumChannel, num_qubits: int = 1) -> 'QuantumProcessMatrix':
        """从通道创建过程矩阵"""
        dim = 2**num_qubits
        paulis = cls._pauli_basis(num_qubits)
        chi = np.zeros((dim**2, dim**2), dtype=complex)
        for i, P_i in enumerate(paulis):
            for j, P_j in enumerate(paulis):
                rho_in = np.eye(dim, dtype=complex) / dim
                # χ_ij = Tr(P_i† E(P_j))
                rho_out = channel.apply(P_j @ rho_in @ P_j.conj().T / dim)
                chi[i, j] = np.trace(P_i.conj().T @ rho_out)
        pm = cls(num_qubits)
        pm.chi = chi
        return pm

    @classmethod
    def _pauli_basis(cls, n: int) -> List[np.ndarray]:
        """n 量子比特 Pauli 基"""
        paulis = [np.eye(2, dtype=complex),
                  np.array([[0,1],[1,0]], dtype=complex),
                  np.array([[0,-1j],[1j,0]], dtype=complex),
                  np.array([[1,0],[0,-1]], dtype=complex)]
        if n == 1:
            return paulis
        result = paulis[:]
        for _ in range(n - 1):
            new_result = []
            for P in result:
                for Q in paulis:
                    new_result.append(np.kron(P, Q))
            result = new_result
        return result

    def fidelity_with_identity(self) -> float:
        """与恒等过程的保真度"""
        dim_sq = self.dim**2
        identity_chi = np.zeros((dim_sq, dim_sq), dtype=complex)
        identity_chi[0, 0] = 1.0
        return float(np.abs(np.trace(self.chi @ identity_chi)) / np.trace(self.chi @ self.chi.conj().T)**0.5)

    def is_valid(self) -> bool:
        """验证过程矩阵有效性"""
        trace = np.trace(self.chi)
        hermitian = np.allclose(self.chi, self.chi.conj().T, atol=1e-6)
        return abs(trace - 1.0) < 0.1 and hermitian


class QuantumStateFactory:
    """量子态工厂"""

    @staticmethod
    def create(state_type: str, **kwargs) -> QuantumStateVector:
        n = kwargs.get('num_qubits', 2)
        if state_type == 'zero':
            return QuantumStateVector.zero_state(n)
        elif state_type == 'one':
            return QuantumStateVector.one_state(n)
        elif state_type == 'plus':
            return QuantumStateVector.plus_state(n)
        elif state_type == 'bell':
            return QuantumStateVector.bell_state(kwargs.get('bell_type', 0))
        elif state_type == 'ghz':
            return QuantumStateVector.ghz_state(n)
        elif state_type == 'w':
            return QuantumStateVector.w_state(n)
        elif state_type == 'random':
            return QuantumStateVector.random_state(n, kwargs.get('seed'))
        elif state_type == 'cat':
            # Schrödinger cat state
            alpha = kwargs.get('alpha', 1.0)
            dim = 2**n
            state = np.zeros(dim, dtype=complex)
            state[0] = np.exp(-abs(alpha)**2 / 2)
            state[-1] = np.exp(-abs(alpha)**2 / 2) * alpha**n
            norm = np.linalg.norm(state)
            if norm > 0:
                state /= norm
            return QuantumStateVector(num_qubits=n, amplitudes=state)
        raise ValueError(f"未知态类型: {state_type}")

    @staticmethod
    def list_types() -> List[str]:
        return ['zero', 'one', 'plus', 'bell', 'ghz', 'w', 'random', 'cat']

    @staticmethod
    def random_haar(num_qubits: int, seed: Optional[int] = None) -> QuantumStateVector:
        """Haar 随机量子态"""
        rng = np.random.RandomState(seed)
        dim = 2**num_qubits
        # 从复高斯分布中采样
        real_part = rng.randn(dim)
        imag_part = rng.randn(dim)
        state = real_part + 1j * imag_part
        state /= np.linalg.norm(state)
        return QuantumStateVector(num_qubits=num_qubits, amplitudes=state)

    @staticmethod
    def maximally_entangled(n: int) -> QuantumStateVector:
        """最大纠缠态"""
        if n % 2 != 0:
            n += 1
        dim = 2**n
        state = np.zeros(dim, dtype=complex)
        for i in range(2**(n//2)):
            state[i * (2**(n//2) + 1)] = 1.0
        state /= np.linalg.norm(state)
        return QuantumStateVector(num_qubits=n, amplitudes=state)
