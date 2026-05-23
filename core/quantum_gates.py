"""量子门库 - 20+ 量子门实现"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import cmath


class GateType(Enum):
    """量子门类型"""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    PARAMETERIZED = "parameterized"
    MEASUREMENT = "measurement"


@dataclass
class QuantumGate:
    """量子门基类"""
    name: str
    gate_type: GateType
    matrix: np.ndarray
    num_qubits: int
    parameters: Optional[List[float]] = None
    description: str = ""

    def __post_init__(self):
        expected_dim = 2 ** self.num_qubits
        if self.matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"矩阵维度不匹配: 期望 ({expected_dim}, {expected_dim}), 实际 {self.matrix.shape}")

    def is_unitary(self, tolerance: float = 1e-10) -> bool:
        """验证是否为酉矩阵"""
        product = self.matrix @ self.matrix.conj().T
        return np.allclose(product, np.eye(2**self.num_qubits), atol=tolerance)

    def inverse(self) -> 'QuantumGate':
        """逆门"""
        return QuantumGate(
            name=f"{self.name}_inv",
            gate_type=self.gate_type,
            matrix=self.matrix.conj().T,
            num_qubits=self.num_qubits,
            description=f"逆 {self.name}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.gate_type.value,
            'num_qubits': self.num_qubits,
            'matrix': self.matrix.tolist(),
            'parameters': self.parameters
        }

    def __repr__(self) -> str:
        return f"QuantumGate({self.name}, {self.num_qubits}q)"


# === 单量子比特门 ===

class PauliX(QuantumGate):
    """Pauli-X 门 (量子非门 / NOT 门)"""
    def __init__(self):
        super().__init__(
            name="X", gate_type=GateType.SINGLE,
            matrix=np.array([[0, 1], [1, 0]], dtype=complex),
            num_qubits=1,
            description="量子非门: |0⟩→|1⟩, |1⟩→|0⟩"
        )


class PauliY(QuantumGate):
    """Pauli-Y 门"""
    def __init__(self):
        super().__init__(
            name="Y", gate_type=GateType.SINGLE,
            matrix=np.array([[0, -1j], [1j, 0]], dtype=complex),
            num_qubits=1,
            description="Pauli-Y 门"
        )


class PauliZ(QuantumGate):
    """Pauli-Z 门 (相位翻转)"""
    def __init__(self):
        super().__init__(
            name="Z", gate_type=GateType.SINGLE,
            matrix=np.array([[1, 0], [0, -1]], dtype=complex),
            num_qubits=1,
            description="相位翻转门: |1⟩→-|1⟩"
        )


class Hadamard(QuantumGate):
    """Hadamard 门 (创建叠加态)"""
    def __init__(self):
        super().__init__(
            name="H", gate_type=GateType.SINGLE,
            matrix=np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            num_qubits=1,
            description="创建叠加态: |0⟩→(|0⟩+|1⟩)/√2"
        )


class SGate(QuantumGate):
    """S 门 (√Z 门)"""
    def __init__(self):
        super().__init__(
            name="S", gate_type=GateType.SINGLE,
            matrix=np.array([[1, 0], [0, 1j]], dtype=complex),
            num_qubits=1,
            description="S 门: π/2 相位旋转"
        )


class TGate(QuantumGate):
    """T 门 (√S 门)"""
    def __init__(self):
        super().__init__(
            name="T", gate_type=GateType.SINGLE,
            matrix=np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]], dtype=complex),
            num_qubits=1,
            description="T 门: π/4 相位旋转"
        )


class PhaseGate(QuantumGate):
    """参数化相位门"""
    def __init__(self, phi: float):
        self.phi = phi
        super().__init__(
            name=f"P({phi:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=complex),
            num_qubits=1,
            parameters=[phi],
            description=f"相位门: 相位={phi:.3f}"
        )


class RxGate(QuantumGate):
    """X 轴旋转门"""
    def __init__(self, theta: float):
        self.theta = theta
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        super().__init__(
            name=f"Rx({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([[c, -1j*s], [-1j*s, c]], dtype=complex),
            num_qubits=1,
            parameters=[theta],
            description=f"X轴旋转: θ={theta:.3f}"
        )


class RyGate(QuantumGate):
    """Y 轴旋转门"""
    def __init__(self, theta: float):
        self.theta = theta
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        super().__init__(
            name=f"Ry({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([[c, -s], [s, c]], dtype=complex),
            num_qubits=1,
            parameters=[theta],
            description=f"Y轴旋转: θ={theta:.3f}"
        )


class RzGate(QuantumGate):
    """Z 轴旋转门"""
    def __init__(self, theta: float):
        self.theta = theta
        super().__init__(
            name=f"Rz({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([[cmath.exp(-1j*theta/2), 0], [0, cmath.exp(1j*theta/2)]], dtype=complex),
            num_qubits=1,
            parameters=[theta],
            description=f"Z轴旋转: θ={theta:.3f}"
        )


class U3Gate(QuantumGate):
    """通用单量子比特酉门 U3(θ, φ, λ)"""
    def __init__(self, theta: float, phi: float, lam: float):
        self.theta = theta
        self.phi = phi
        self.lam = lam
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        super().__init__(
            name=f"U3({theta:.2f},{phi:.2f},{lam:.2f})",
            gate_type=GateType.PARAMETERIZED,
            matrix=np.array([
                [c, -s * cmath.exp(1j * lam)],
                [s * cmath.exp(1j * phi), c * cmath.exp(1j * (phi + lam))]
            ], dtype=complex),
            num_qubits=1,
            parameters=[theta, phi, lam],
            description=f"通用酉门 U3"
        )


# === 双量子比特门 ===

class CNOT(QuantumGate):
    """受控非门 (CX 门)"""
    def __init__(self):
        super().__init__(
            name="CNOT", gate_type=GateType.DOUBLE,
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex),
            num_qubits=2,
            description="受控非门: 控制比特翻转目标比特"
        )


class CZ(QuantumGate):
    """受控 Z 门"""
    def __init__(self):
        super().__init__(
            name="CZ", gate_type=GateType.DOUBLE,
            matrix=np.diag([1, 1, 1, -1]).astype(complex),
            num_qubits=2,
            description="受控相位门"
        )


class CY(QuantumGate):
    """受控 Y 门"""
    def __init__(self):
        m = np.eye(4, dtype=complex)
        m[2, 2] = 0; m[2, 3] = -1j; m[3, 2] = 1j; m[3, 3] = 0
        super().__init__(
            name="CY", gate_type=GateType.DOUBLE,
            matrix=m, num_qubits=2,
            description="受控 Y 门"
        )


class SWAP(QuantumGate):
    """SWAP 门"""
    def __init__(self):
        super().__init__(
            name="SWAP", gate_type=GateType.DOUBLE,
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            num_qubits=2,
            description="交换两个量子比特的状态"
        )


class ISWAP(QuantumGate):
    """iSWAP 门"""
    def __init__(self):
        super().__init__(
            name="iSWAP", gate_type=GateType.DOUBLE,
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            num_qubits=2,
            description="iSWAP 门"
        )


class DCNOT(QuantumGate):
    """双重受控非门"""
    def __init__(self):
        super().__init__(
            name="DCNOT", gate_type=GateType.DOUBLE,
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex),
            num_qubits=2,
            description="双重受控非门"
        )


class CPhaseGate(QuantumGate):
    """受控相位门"""
    def __init__(self, phi: float):
        m = np.eye(4, dtype=complex)
        m[3, 3] = cmath.exp(1j * phi)
        super().__init__(
            name=f"CPhase({phi:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=m, num_qubits=2, parameters=[phi],
            description=f"受控相位门: φ={phi:.3f}"
        )


class XXGate(QuantumGate):
    """XX 相互作用门"""
    def __init__(self, theta: float):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        super().__init__(
            name=f"XX({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([
                [c, 0, 0, -1j*s],
                [0, c, -1j*s, 0],
                [0, -1j*s, c, 0],
                [-1j*s, 0, 0, c]
            ], dtype=complex),
            num_qubits=2, parameters=[theta],
            description=f"XX 相互作用门: θ={theta:.3f}"
        )


class YYGate(QuantumGate):
    """YY 相互作用门"""
    def __init__(self, theta: float):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        super().__init__(
            name=f"YY({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.array([
                [c, 0, 0, 1j*s],
                [0, c, -1j*s, 0],
                [0, -1j*s, c, 0],
                [1j*s, 0, 0, c]
            ], dtype=complex),
            num_qubits=2, parameters=[theta],
            description=f"YY 相互作用门: θ={theta:.3f}"
        )


class ZZGate(QuantumGate):
    """ZZ 相互作用门"""
    def __init__(self, theta: float):
        p = cmath.exp(1j * theta / 2)
        m = cmath.exp(-1j * theta / 2)
        super().__init__(
            name=f"ZZ({theta:.3f})", gate_type=GateType.PARAMETERIZED,
            matrix=np.diag([m, p, p, m]).astype(complex),
            num_qubits=2, parameters=[theta],
            description=f"ZZ 相互作用门: θ={theta:.3f}"
        )


# === 三量子比特门 ===

class Toffoli(QuantumGate):
    """Toffoli 门 (CCX 门)"""
    def __init__(self):
        m = np.eye(8, dtype=complex)
        m[6, 6] = 0; m[6, 7] = 1; m[7, 6] = 1; m[7, 7] = 0
        super().__init__(
            name="Toffoli", gate_type=GateType.TRIPLE,
            matrix=m, num_qubits=3,
            description="双控非门: 两个控制比特同时为1时翻转目标"
        )


class Fredkin(QuantumGate):
    """Fredkin 门 (受控 SWAP)"""
    def __init__(self):
        m = np.eye(8, dtype=complex)
        m[5, 5] = 0; m[5, 6] = 1
        m[6, 5] = 1; m[6, 6] = 0
        super().__init__(
            name="Fredkin", gate_type=GateType.TRIPLE,
            matrix=m, num_qubits=3,
            description="受控 SWAP 门"
        )


class CCZ(QuantumGate):
    """CCZ 门 (双控 Z 门)"""
    def __init__(self):
        m = np.eye(8, dtype=complex)
        m[7, 7] = -1
        super().__init__(
            name="CCZ", gate_type=GateType.TRIPLE,
            matrix=m, num_qubits=3,
            description="双控 Z 门"
        )


# === 门工厂 ===

class GateFactory:
    """量子门工厂"""

    _gates = {
        'X': PauliX, 'Y': PauliY, 'Z': PauliZ,
        'H': Hadamard, 'S': SGate, 'T': TGate,
        'CNOT': CNOT, 'CX': CNOT, 'CZ': CZ, 'CY': CY,
        'SWAP': SWAP, 'iSWAP': ISWAP, 'DCNOT': DCNOT,
        'Toffoli': Toffoli, 'CCX': Toffoli,
        'Fredkin': Fredkin, 'CCZ': CCZ,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> QuantumGate:
        """创建量子门"""
        if name in cls._gates:
            return cls._gates[name]()
        elif name == 'P':
            return PhaseGate(kwargs.get('phi', 0.0))
        elif name == 'Rx':
            return RxGate(kwargs.get('theta', 0.0))
        elif name == 'Ry':
            return RyGate(kwargs.get('theta', 0.0))
        elif name == 'Rz':
            return RzGate(kwargs.get('theta', 0.0))
        elif name == 'U3':
            return U3Gate(kwargs.get('theta', 0.0), kwargs.get('phi', 0.0), kwargs.get('lam', 0.0))
        elif name == 'CPhase':
            return CPhaseGate(kwargs.get('phi', 0.0))
        elif name == 'XX':
            return XXGate(kwargs.get('theta', 0.0))
        elif name == 'YY':
            return YYGate(kwargs.get('theta', 0.0))
        elif name == 'ZZ':
            return ZZGate(kwargs.get('theta', 0.0))
        raise ValueError(f"未知门: {name}")

    @classmethod
    def list_gates(cls) -> List[str]:
        """列出所有可用门"""
        return list(cls._gates.keys()) + ['P', 'Rx', 'Ry', 'Rz', 'U3', 'CPhase', 'XX', 'YY', 'ZZ']

    @classmethod
    def random_single_gate(cls, seed: Optional[int] = None) -> QuantumGate:
        """随机单量子比特门"""
        rng = np.random.RandomState(seed)
        single = ['X', 'Y', 'Z', 'H', 'S', 'T']
        name = rng.choice(single)
        return cls.create(name)

    @classmethod
    def random_two_gate(cls, seed: Optional[int] = None) -> QuantumGate:
        """随机双量子比特门"""
        rng = np.random.RandomState(seed)
        double = ['CNOT', 'CZ', 'CY', 'SWAP']
        name = rng.choice(double)
        return cls.create(name)

    @classmethod
    def random_rotation(cls, seed: Optional[int] = None) -> QuantumGate:
        """随机旋转门"""
        rng = np.random.RandomState(seed)
        axis = rng.choice(['Rx', 'Ry', 'Rz'])
        theta = rng.uniform(0, 2 * np.pi)
        return cls.create(axis, theta=theta)


def identity(n: int = 1) -> np.ndarray:
    """n 量子比特恒等矩阵"""
    return np.eye(2**n, dtype=complex)


def tensor_product(*gates: QuantumGate) -> np.ndarray:
    """多个门的张量积"""
    result = gates[0].matrix
    for gate in gates[1:]:
        result = np.kron(result, gate.matrix)
    return result


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """对易子 [A, B] = AB - BA"""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """反对易子 {A, B} = AB + BA"""
    return A @ B + B @ A


# Pauli 矩阵常量
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
