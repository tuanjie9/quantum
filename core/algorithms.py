"""量子算法库 - Deutsch-Jozsa/Grover/QFT/Teleportation 等"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
from enum import Enum
import math


class AlgorithmType(Enum):
    """量子算法类型"""
    DEUTSCH_JOZSA = "deutsch_jozsa"
    BERNSTEIN_VAZIRANI = "bernstein_vazirani"
    SIMON = "simon"
    GROVER = "grover"
    QPE = "quantum_phase_estimation"
    QFT = "quantum_fourier_transform"
    TELEPORTATION = "teleportation"
    SUPERDENSE = "superdense_coding"
    QKD = "quantum_key_distribution"
    WALK = "quantum_walk"


@dataclass
class AlgorithmResult:
    """算法执行结果"""
    algorithm: AlgorithmType
    success: bool
    output: Any
    num_qubits: int
    num_gates: int
    depth: int
    shots: int = 1024
    probability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm.value,
            'success': self.success,
            'output': str(self.output),
            'num_qubits': self.num_qubits,
            'num_gates': self.num_gates,
            'depth': self.depth,
            'probability': self.probability
        }


class QuantumFourierTransform:
    """量子傅里叶变换 (QFT)"""

    def __init__(self, n: int):
        self.n = n
        self.dim = 2**n

    def qft_matrix(self) -> np.ndarray:
        """QFT 矩阵"""
        omega = np.exp(2j * np.pi / self.dim)
        Q = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim):
            for j in range(self.dim):
                Q[i, j] = omega**(i * j) / np.sqrt(self.dim)
        return Q

    def apply(self, state: np.ndarray) -> np.ndarray:
        """应用 QFT"""
        return self.qft_matrix() @ state

    def inverse(self, state: np.ndarray) -> np.ndarray:
        """应用逆 QFT"""
        return self.qft_matrix().conj().T @ state

    def circuit(self) -> List[Dict]:
        """QFT 电路分解"""
        gates = []
        for i in range(self.n):
            gates.append({'gate': 'H', 'qubits': [i]})
            for j in range(i + 1, self.n):
                angle = np.pi / (2**(j - i))
                gates.append({'gate': 'CPhase', 'qubits': [j, i], 'params': [angle]})
        # 交换量子比特
        for i in range(self.n // 2):
            gates.append({'gate': 'SWAP', 'qubits': [i, self.n - 1 - i]})
        return gates

    def gate_count(self) -> int:
        """门数量"""
        return self.n + self.n * (self.n - 1) // 2 + self.n // 2


class DeutschJozsa:
    """Deutsch-Jozsa 算法"""

    def __init__(self, n: int, oracle_type: str = "balanced"):
        self.n = n
        self.oracle_type = oracle_type

    def oracle_matrix(self) -> np.ndarray:
        """Oracle 矩阵"""
        dim = 2**(self.n + 1)
        U = np.eye(dim, dtype=complex)
        if self.oracle_type == "constant":
            pass  # 恒等矩阵 = 常数函数 f(x)=0
        elif self.oracle_type == "balanced":
            for x in range(2**self.n):
                bit = bin(x).count('1') % 2
                if bit == 1:
                    idx = x * 2 + 1
                    U[idx, idx] = -1
        return U

    def run(self, shots: int = 1024) -> AlgorithmResult:
        """运行 Deutsch-Jozsa 算法"""
        dim = 2**(self.n + 1)
        state = np.zeros(dim, dtype=complex)
        # 初始化: |0⟩^n |1⟩
        state[2**self.n] = 1.0  # |0...0⟩|1⟩ (index = 2^n)
        # Hadamard on all qubits
        H_n = np.ones((2**self.n, 2**self.n), dtype=complex) / np.sqrt(2**self.n)
        H_2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_all = np.kron(H_n, H_2)
        state = H_all @ state
        # Oracle
        state = self.oracle_matrix() @ state
        # Hadamard on first n qubits
        state = np.kron(H_n.conj().T, np.eye(2, dtype=complex)) @ state
        # 测量前 n 个量子比特
        probs = np.abs(state)**2
        probs_0 = sum(probs[i] for i in range(0, dim, 2))
        is_constant = probs_0 > 0.99
        return AlgorithmResult(
            algorithm=AlgorithmType.DEUTSCH_JOZSA,
            success=True,
            output="constant" if is_constant else "balanced",
            num_qubits=self.n + 1,
            num_gates=self.n * 2 + 1,
            depth=3,
            shots=shots,
            probability=float(probs_0)
        )


class GroverSearch:
    """Grover 搜索算法"""

    def __init__(self, n: int, target: int):
        self.n = n
        self.target = target
        self.dim = 2**n
        self.num_iterations = int(np.pi / 4 * np.sqrt(self.dim))

    def oracle(self) -> np.ndarray:
        """搜索 Oracle"""
        U = np.eye(self.dim, dtype=complex)
        U[self.target, self.target] = -1
        return U

    def diffusion(self) -> np.ndarray:
        """扩散算子"""
        H = np.ones((self.dim, self.dim), dtype=complex) / np.sqrt(self.dim)
        return 2 * H - np.eye(self.dim, dtype=complex)

    def run(self, shots: int = 1024) -> AlgorithmResult:
        """运行 Grover 搜索"""
        state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        for _ in range(self.num_iterations):
            state = self.oracle() @ state
            state = self.diffusion() @ state
        probs = np.abs(state)**2
        measured_idx = np.argmax(probs)
        success = measured_idx == self.target
        return AlgorithmResult(
            algorithm=AlgorithmType.GROVER,
            success=success,
            output=format(measured_idx, f'0{self.n}b'),
            num_qubits=self.n,
            num_gates=self.num_iterations * 2,
            depth=self.num_iterations * 2,
            shots=shots,
            probability=float(probs[self.target]),
            metadata={'target': format(self.target, f'0{self.n}b'), 'iterations': self.num_iterations}
        )

    def optimal_iterations(self) -> int:
        """最优迭代次数"""
        return self.num_iterations


class QuantumPhaseEstimation:
    """量子相位估计 (QPE)"""

    def __init__(self, unitary: np.ndarray, precision_bits: int = 4):
        self.unitary = unitary
        self.precision_bits = precision_bits
        self.n_unitary = int(np.log2(unitary.shape[0]))

    def run(self, shots: int = 1024) -> AlgorithmResult:
        """运行 QPE"""
        n_c = self.precision_bits
        n_total = n_c + self.n_unitary
        dim_total = 2**n_total
        # 构造受控 U 矩阵
        eigenvalues, eigenvectors = np.linalg.eig(self.unitary)
        phases = np.angle(eigenvalues) / (2 * np.pi)
        # 估计相位
        estimated_phases = []
        for phase in phases:
            phase_bits = 0
            for k in range(n_c):
                bit_prob = (0.5 + 0.5 * np.cos(2 * np.pi * phase * 2**k))
                if np.random.random() < bit_prob:
                    phase_bits |= (1 << (n_c - 1 - k))
            estimated_phases.append(phase_bits / 2**n_c)
        best_phase = max(estimated_phases, key=lambda p: abs(np.cos(2 * np.pi * p)))
        return AlgorithmResult(
            algorithm=AlgorithmType.QPE,
            success=True,
            output=estimated_phases,
            num_qubits=n_total,
            num_gates=n_c * 3,
            depth=n_c * 2,
            shots=shots,
            probability=0.5,
            metadata={'estimated_phases': estimated_phases, 'eigenvalues': eigenvalues.tolist()}
        )


class QuantumTeleportation:
    """量子隐形传态"""

    def __init__(self):
        pass

    def run(self, state_to_teleport: np.ndarray, shots: int = 1024) -> AlgorithmResult:
        """运行量子隐形传态"""
        # 初始态 |ψ⟩ ⊗ |00⟩
        psi = state_to_teleport / np.linalg.norm(state_to_teleport)
        state = np.zeros(8, dtype=complex)
        state[0] = psi[0]
        state[4] = psi[1]
        # 创建 Bell 态 (H + CNOT on qubits 1,2)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        # 简化模拟: Bell 测量结果
        bell_states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        measurement = bell_states[np.random.randint(4)]
        # 纠正操作
        corrected_state = psi.copy()
        if measurement == (0, 1):
            corrected_state = np.array([psi[1], psi[0]])  # X
        elif measurement == (1, 0):
            corrected_state = np.array([psi[0], -psi[1]])  # Z
        elif measurement == (1, 1):
            corrected_state = np.array([-psi[1], psi[0]])  # ZX
        fidelity = abs(np.vdot(psi, corrected_state))**2
        return AlgorithmResult(
            algorithm=AlgorithmType.TELEPORTATION,
            success=fidelity > 0.99,
            output={'teleported_state': corrected_state.tolist(), 'bell_measurement': measurement},
            num_qubits=3,
            num_gates=4,
            depth=4,
            shots=shots,
            probability=fidelity,
            metadata={'fidelity': fidelity}
        )


class SuperdenseCoding:
    """超密编码"""

    def __init__(self):
        self.encodings = {
            '00': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),  # I
            '01': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),  # X
            '10': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # Z
            '11': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # ZX
        }

    def encode(self, message: str) -> np.ndarray:
        """编码 2 比特消息到 1 量子比特"""
        return self.encodings[message]

    def decode(self, state: np.ndarray) -> str:
        """解码"""
        best_match = '00'
        best_overlap = 0
        for msg, encoded in self.encodings.items():
            overlap = abs(np.vdot(state, encoded))**2
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = msg
        return best_match

    def run(self, message: str = "01", shots: int = 1024) -> AlgorithmResult:
        """运行超密编码"""
        if message not in self.encodings:
            raise ValueError(f"无效消息: {message}, 可选: {list(self.encodings.keys())}")
        encoded_state = self.encode(message)
        decoded = self.decode(encoded_state)
        return AlgorithmResult(
            algorithm=AlgorithmType.SUPERDENSE,
            success=decoded == message,
            output={'original': message, 'decoded': decoded},
            num_qubits=2,
            num_gates=2,
            depth=2,
            shots=shots,
            probability=1.0 if decoded == message else 0.0
        )


class QuantumKeyDistribution:
    """BB84 量子密钥分发"""

    def __init__(self, key_length: int = 16):
        self.key_length = key_length

    def run(self, eavesdrop: bool = False, shots: int = 1) -> AlgorithmResult:
        """运行 BB84 协议"""
        alice_bits = np.random.randint(0, 2, self.key_length * 2)
        alice_bases = np.random.randint(0, 2, self.key_length * 2)
        bob_bases = np.random.randint(0, 2, self.key_length * 2)
        bob_results = []
        for i in range(self.key_length * 2):
            bit = alice_bits[i]
            a_basis = alice_bases[i]
            b_basis = bob_bases[i]
            if eavesdrop:
                eve_basis = np.random.randint(0, 2)
                if eve_basis != a_basis:
                    bit = np.random.randint(0, 2)
            if a_basis == b_basis:
                bob_results.append(bit)
            else:
                bob_results.append(np.random.randint(0, 2))
        # 筛选: 保留基相同的结果
        sifted_alice = []
        sifted_bob = []
        for i in range(self.key_length * 2):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(int(alice_bits[i]))
                sifted_bob.append(int(bob_results[i]))
        # QBER 估计
        if len(sifted_alice) > 4:
            check_bits = min(len(sifted_alice) // 4, 4)
            errors = sum(1 for i in range(check_bits) if sifted_alice[i] != sifted_bob[i])
            qber = errors / check_bits
        else:
            qber = 0
        final_key = sifted_alice[:self.key_length]
        return AlgorithmResult(
            algorithm=AlgorithmType.QKD,
            success=qber < 0.11,
            output={'key': final_key, 'qber': qber, 'key_length': len(final_key)},
            num_qubits=1,
            num_gates=0,
            depth=0,
            shots=1,
            probability=1 - qber,
            metadata={'eavesdrop': eavesdrop, 'sifted_length': len(sifted_alice), 'qber': qber}
        )


class QuantumWalk:
    """量子随机游走"""

    def __init__(self, num_steps: int = 10, graph_type: str = "line"):
        self.num_steps = num_steps
        self.graph_type = graph_type

    def run_line_walk(self, start: int = 0) -> AlgorithmResult:
        """线性图上的量子游走"""
        n = self.num_steps
        positions = list(range(-n, n + 1))
        dim = len(positions)
        # 币空间 + 位置空间
        coin = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        # 状态: |position⟩ ⊗ |coin⟩
        state = np.zeros(dim * 2, dtype=complex)
        start_idx = positions.index(start)
        state[start_idx * 2] = 1.0 / np.sqrt(2)
        state[start_idx * 2 + 1] = 1.0 / np.sqrt(2)
        for _ in range(self.num_steps):
            new_state = np.zeros_like(state)
            for p in range(dim):
                for c in range(2):
                    idx = p * 2 + c
                    if abs(state[idx]) < 1e-10:
                        continue
                    # 应用币操作
                    for new_c in range(2):
                        amp = coin[new_c, c] * state[idx]
                        # 移动
                        new_p = p + (1 if new_c == 0 else -1)
                        if 0 <= new_p < dim:
                            new_state[new_p * 2 + new_c] += amp
            state = new_state
        # 计算位置概率
        probs = np.zeros(dim)
        for p in range(dim):
            probs[p] = abs(state[p * 2])**2 + abs(state[p * 2 + 1])**2
        return AlgorithmResult(
            algorithm=AlgorithmType.WALK,
            success=True,
            output={'positions': positions, 'probabilities': probs.tolist()},
            num_qubits=int(np.log2(dim)) + 1,
            num_gates=self.num_steps * 3,
            depth=self.num_steps * 2,
            probability=float(np.max(probs))
        )


class BernsteinVazirani:
    """Bernstein-Vazirani 算法"""

    def __init__(self, n: int, secret: int):
        self.n = n
        self.secret = secret

    def run(self, shots: int = 1024) -> AlgorithmResult:
        dim = 2**self.n
        # 初始化叠加态
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)
        # Oracle: f(x) = s·x mod 2
        for x in range(dim):
            parity = bin(x & self.secret).count('1') % 2
            if parity == 1:
                state[x] *= -1
        # Hadamard transform
        H_n = np.ones((dim, dim), dtype=complex) / np.sqrt(dim)
        state = H_n @ state
        measured = np.argmax(np.abs(state)**2)
        return AlgorithmResult(
            algorithm=AlgorithmType.BERNSTEIN_VAZIRANI,
            success=measured == self.secret,
            output=format(measured, f'0{self.n}b'),
            num_qubits=self.n,
            num_gates=self.n + 1,
            depth=2,
            shots=shots,
            probability=float(np.abs(state[self.secret])**2)
        )


class Simon:
    """Simon 算法"""

    def __init__(self, n: int, secret: int):
        self.n = n
        self.secret = secret

    def run(self, shots: int = 1024) -> AlgorithmResult:
        # 简化实现: 返回满足 s·y=0 的 y 值
        solutions = []
        for y in range(2**self.n):
            if bin(y & self.secret).count('1') % 2 == 0:
                solutions.append(y)
        return AlgorithmResult(
            algorithm=AlgorithmType.SIMON,
            success=len(solutions) > 0,
            output={'solutions': solutions[:10], 'secret_found': self.secret in solutions},
            num_qubits=2 * self.n,
            num_gates=self.n * 2,
            depth=self.n * 2,
            probability=1.0
        )


class QuantumAlgorithmLibrary:
    """量子算法库 - 统一接口"""

    def __init__(self):
        self.algorithms: Dict[str, Any] = {}

    def register(self, name: str, algorithm: Any):
        """注册算法"""
        self.algorithms[name] = algorithm

    def run(self, name: str, **kwargs) -> AlgorithmResult:
        """运行算法"""
        if name == "qft":
            n = kwargs.get('n', 4)
            qft = QuantumFourierTransform(n)
            state = np.zeros(2**n, dtype=complex)
            state[0] = 1.0
            result_state = qft.apply(state)
            return AlgorithmResult(
                algorithm=AlgorithmType.QFT, success=True, output=result_state.tolist(),
                num_qubits=n, num_gates=qft.gate_count(), depth=n
            )
        elif name == "grover":
            n = kwargs.get('n', 4)
            target = kwargs.get('target', 0)
            grover = GroverSearch(n, target)
            return grover.run(kwargs.get('shots', 1024))
        elif name == "deutsch_jozsa":
            n = kwargs.get('n', 3)
            oracle_type = kwargs.get('oracle_type', 'balanced')
            dj = DeutschJozsa(n, oracle_type)
            return dj.run()
        elif name == "teleportation":
            state = kwargs.get('state', np.array([1, 0], dtype=complex))
            tp = QuantumTeleportation()
            return tp.run(state)
        elif name == "superdense":
            message = kwargs.get('message', '01')
            sd = SuperdenseCoding()
            return sd.run(message)
        elif name == "qkd":
            eavesdrop = kwargs.get('eavesdrop', False)
            qkd = QuantumKeyDistribution(kwargs.get('key_length', 16))
            return qkd.run(eavesdrop)
        elif name == "qpe":
            unitary = kwargs.get('unitary', np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]))
            qpe = QuantumPhaseEstimation(unitary, kwargs.get('precision', 4))
            return qpe.run()
        elif name == "bernstein_vazirani":
            n = kwargs.get('n', 4)
            secret = kwargs.get('secret', 0b1010)
            bv = BernsteinVazirani(n, secret)
            return bv.run()
        elif name == "simon":
            n = kwargs.get('n', 4)
            secret = kwargs.get('secret', 0b1100)
            simon = Simon(n, secret)
            return simon.run()
        elif name == "quantum_walk":
            walk = QuantumWalk(kwargs.get('steps', 10))
            return walk.run_line_walk()
        raise ValueError(f"未知算法: {name}")

    def list_algorithms(self) -> List[str]:
        """列出所有可用算法"""
        return ["qft", "grover", "deutsch_jozsa", "teleportation", "superdense",
                "qkd", "qpe", "bernstein_vazirani", "simon", "quantum_walk"]
