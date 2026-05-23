"""量子电路设计与模拟"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import copy
from .quantum_state import QuantumStateVector
from .quantum_gates import QuantumGate, GateFactory, GateType, identity


class CircuitOperation(Enum):
    """电路操作类型"""
    GATE = "gate"
    MEASUREMENT = "measurement"
    BARRIER = "barrier"
    RESET = "reset"


@dataclass
class CircuitInstruction:
    """电路指令"""
    operation: CircuitOperation
    qubits: List[int]
    gate: Optional[QuantumGate] = None
    classical_bits: Optional[List[int]] = None
    label: str = ""

    def __repr__(self) -> str:
        if self.operation == CircuitOperation.GATE:
            return f"{self.gate.name}({','.join(map(str, self.qubits))})"
        elif self.operation == CircuitOperation.MEASUREMENT:
            return f"M({','.join(map(str, self.qubits))})"
        return f"{self.operation.value}({','.join(map(str, self.qubits))})"


class QuantumCircuit:
    """量子电路"""

    def __init__(self, num_qubits: int, num_classical: int = 0):
        self.num_qubits = num_qubits
        self.num_classical = num_classical if num_classical > 0 else num_qubits
        self.instructions: List[CircuitInstruction] = []
        self.metadata: Dict[str, Any] = {}

    def add_gate(self, gate: QuantumGate, qubits: List[int]) -> 'QuantumCircuit':
        """添加量子门"""
        if len(qubits) != gate.num_qubits:
            raise ValueError(f"门 {gate.name} 需要 {gate.num_qubits} 个量子比特, 提供了 {len(qubits)}")
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"量子比特 {q} 超出范围 [0, {self.num_qubits-1}]")
        self.instructions.append(CircuitInstruction(
            operation=CircuitOperation.GATE,
            qubits=qubits,
            gate=gate
        ))
        return self

    def add_measurement(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """添加测量操作"""
        if classical_bits is None:
            classical_bits = qubits
        self.instructions.append(CircuitInstruction(
            operation=CircuitOperation.MEASUREMENT,
            qubits=qubits,
            classical_bits=classical_bits
        ))
        return self

    def add_barrier(self) -> 'QuantumCircuit':
        """添加屏障"""
        self.instructions.append(CircuitInstruction(
            operation=CircuitOperation.BARRIER,
            qubits=list(range(self.num_qubits))
        ))
        return self

    def h(self, qubit: int) -> 'QuantumCircuit':
        """快捷添加 Hadamard 门"""
        return self.add_gate(GateFactory.create('H'), [qubit])

    def x(self, qubit: int) -> 'QuantumCircuit':
        """快捷添加 Pauli-X 门"""
        return self.add_gate(GateFactory.create('X'), [qubit])

    def y(self, qubit: int) -> 'QuantumCircuit':
        """快捷添加 Pauli-Y 门"""
        return self.add_gate(GateFactory.create('Y'), [qubit])

    def z(self, qubit: int) -> 'QuantumCircuit':
        """快捷添加 Pauli-Z 门"""
        return self.add_gate(GateFactory.create('Z'), [qubit])

    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """快捷添加 CNOT 门"""
        return self.add_gate(GateFactory.create('CNOT'), [control, target])

    def cz(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """快捷添加 CZ 门"""
        return self.add_gate(GateFactory.create('CZ'), [qubit1, qubit2])

    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """快捷添加 SWAP 门"""
        return self.add_gate(GateFactory.create('SWAP'), [qubit1, qubit2])

    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.add_gate(GateFactory.create('Rx', theta=theta), [qubit])

    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.add_gate(GateFactory.create('Ry', theta=theta), [qubit])

    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.add_gate(GateFactory.create('Rz', theta=theta), [qubit])

    def measure_all(self) -> 'QuantumCircuit':
        """测量所有量子比特"""
        return self.add_measurement(list(range(self.num_qubits)))

    @property
    def depth(self) -> int:
        """电路深度"""
        layers = []
        qubit_layer = [0] * self.num_qubits
        for instr in self.instructions:
            if instr.operation == CircuitOperation.BARRIER:
                continue
            max_layer = max(qubit_layer[q] for q in instr.qubits)
            for q in instr.qubits:
                qubit_layer[q] = max_layer + 1
        return max(qubit_layer) if qubit_layer else 0

    @property
    def width(self) -> int:
        """电路宽度"""
        return self.num_qubits

    @property
    def gate_count(self) -> int:
        """门数量"""
        return sum(1 for i in self.instructions if i.operation == CircuitOperation.GATE)

    @property
    def gate_types(self) -> Dict[str, int]:
        """各类型门的计数"""
        counts = {}
        for instr in self.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                name = instr.gate.name.split('(')[0]
                counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def two_qubit_gate_count(self) -> int:
        """双量子比特门数量"""
        return sum(1 for i in self.instructions
                   if i.operation == CircuitOperation.GATE and i.gate and i.gate.num_qubits >= 2)

    def simulate(self, initial_state: Optional[QuantumStateVector] = None, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, Any]:
        """模拟电路执行"""
        if initial_state is None:
            state = QuantumStateVector.zero_state(self.num_qubits)
        else:
            state = copy.deepcopy(initial_state)

        measurements = {}
        for instr in self.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                self._apply_gate(state, instr.gate, instr.qubits)
            elif instr.operation == CircuitOperation.MEASUREMENT:
                results = state.measure(shots=shots, seed=seed)
                measurements.update(results)
                break  # 测量后终止

        if not measurements:
            measurements = state.measure(shots=shots, seed=seed)

        return {
            'counts': measurements,
            'state_vector': state,
            'depth': self.depth,
            'gate_count': self.gate_count,
            'shots': shots
        }

    def _apply_gate(self, state: QuantumStateVector, gate: QuantumGate, qubits: List[int]):
        """将门应用到量子态"""
        n = self.num_qubits
        dim = 2**n
        gate_matrix = gate.matrix
        gate_qubits = gate.num_qubits

        if gate_qubits == 1:
            q = qubits[0]
            full_matrix = np.eye(1, dtype=complex)
            for i in range(n):
                if i == q:
                    full_matrix = np.kron(full_matrix, gate_matrix)
                else:
                    full_matrix = np.kron(full_matrix, identity(1))
            state.amplitudes = full_matrix @ state.amplitudes
        elif gate_qubits == 2:
            q1, q2 = qubits
            # 简化: 直接构造全空间矩阵
            new_amps = np.zeros(dim, dtype=complex)
            for i in range(dim):
                b1 = (i >> (n - 1 - q1)) & 1
                b2 = (i >> (n - 1 - q2)) & 1
                idx_in = b1 * 2 + b2
                for j_out in range(4):
                    b1_out = (j_out >> 1) & 1
                    b2_out = j_out & 1
                    if gate_matrix[j_out, idx_in] != 0:
                        new_i = i
                        if b1 != b1_out:
                            new_i ^= (1 << (n - 1 - q1))
                        if b2 != b2_out:
                            new_i ^= (1 << (n - 1 - q2))
                        new_amps[new_i] += gate_matrix[j_out, idx_in] * state.amplitudes[i]
            state.amplitudes = new_amps
        else:
            # 通用多比特门
            new_amps = np.zeros(dim, dtype=complex)
            gate_dim = 2**gate_qubits
            for i in range(dim):
                bits = [(i >> (n - 1 - q)) & 1 for q in qubits]
                idx_in = sum(b << (gate_qubits - 1 - k) for k, b in enumerate(bits))
                for j_out in range(gate_dim):
                    if gate_matrix[j_out, idx_in] != 0:
                        new_i = i
                        for k, q in enumerate(qubits):
                            b_old = bits[k]
                            b_new = (j_out >> (gate_qubits - 1 - k)) & 1
                            if b_old != b_new:
                                new_i ^= (1 << (n - 1 - q))
                        new_amps[new_i] += gate_matrix[j_out, idx_in] * state.amplitudes[i]
            state.amplitudes = new_amps
        state.normalize()

    def optimize(self) -> 'QuantumCircuit':
        """优化电路（门消除、合并）"""
        optimized = QuantumCircuit(self.num_qubits, self.num_classical)
        i = 0
        while i < len(self.instructions):
            curr = self.instructions[i]
            if i + 1 < len(self.instructions):
                nxt = self.instructions[i + 1]
                if (curr.operation == CircuitOperation.GATE and nxt.operation == CircuitOperation.GATE
                    and curr.gate and nxt.gate and curr.qubits == nxt.qubits):
                    # 检测 HH, XX, YY, ZZ 消除
                    if curr.gate.name == nxt.gate.name and curr.gate.name in ['H', 'X', 'Y', 'Z', 'CNOT', 'SWAP']:
                        i += 2  # 跳过这对门
                        continue
            optimized.instructions.append(curr)
            i += 1
        return optimized

    def draw(self) -> str:
        """ASCII 电路图"""
        lines = [[] for _ in range(self.num_qubits)]
        wire_labels = [f"q{q}: " for q in range(self.num_qubits)]

        for instr in self.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                gate_name = instr.gate.name.split('(')[0]
                max_len = max(len(gate_name) + 2, 3)
                for q in range(self.num_qubits):
                    if q in instr.qubits:
                        if len(instr.qubits) == 1:
                            lines[q].append(f"[{gate_name}]")
                        elif q == instr.qubits[0]:
                            lines[q].append(f"[{gate_name}]")
                        else:
                            lines[q].append(" ● " if q == instr.qubits[-1] else " | ")
                    else:
                        if any(q2 > q for q2 in instr.qubits) and any(q2 < q for q2 in instr.qubits):
                            lines[q].append(" | ")
                        else:
                            lines[q].append("---")

        result = []
        for q in range(self.num_qubits):
            line = wire_labels[q] + "---" + "---".join(lines[q]) + "---"
            result.append(line)
        return "\n".join(result)

    def to_qasm(self) -> str:
        """导出为 OpenQASM 格式"""
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{self.num_qubits}];',
            f'creg c[{self.num_classical}];'
        ]
        gate_map = {
            'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
            'CNOT': 'cx', 'CZ': 'cz', 'SWAP': 'swap',
            'T': 't', 'S': 's', 'Toffoli': 'ccx',
        }
        for instr in self.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                name = instr.gate.name.split('(')[0]
                qasm_name = gate_map.get(name, name.lower())
                qubits = ','.join(f'q[{q}]' for q in instr.qubits)
                if instr.gate.parameters:
                    params = ','.join(f'{p}' for p in instr.gate.parameters)
                    lines.append(f'{qasm_name}({params}) {qubits};')
                else:
                    lines.append(f'{qasm_name} {qubits};')
            elif instr.operation == CircuitOperation.MEASUREMENT:
                for q in instr.qubits:
                    lines.append(f'measure q[{q}] -> c[{q}];')
        return '\n'.join(lines)

    @classmethod
    def from_qasm(cls, qasm_str: str) -> 'QuantumCircuit':
        """从 QASM 解析"""
        lines = qasm_str.strip().split('\n')
        num_qubits = 0
        for line in lines:
            if 'qreg' in line:
                num_qubits = int(line.split('[')[1].split(']')[0])
                break
        circuit = cls(num_qubits)
        for line in lines:
            line = line.strip().rstrip(';')
            if line.startswith(('OPENQASM', 'include', 'qreg', 'creg', '//')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                gate_name = parts[0].upper()
                qubits = [int(q.split('[')[1].split(']')[0]) for q in parts[1].split(',')]
                try:
                    gate = GateFactory.create(gate_name)
                    circuit.add_gate(gate, qubits)
                except ValueError:
                    pass
        return circuit

    def copy(self) -> 'QuantumCircuit':
        """深拷贝电路"""
        return copy.deepcopy(self)

    def compose(self, other: 'QuantumCircuit', qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """组合两个电路"""
        if qubits is None:
            qubits = list(range(other.num_qubits))
        result = self.copy()
        for instr in other.instructions:
            new_qubits = [qubits[q] for q in instr.qubits]
            result.instructions.append(CircuitInstruction(
                operation=instr.operation,
                qubits=new_qubits,
                gate=instr.gate,
                classical_bits=instr.classical_bits
            ))
        return result

    def inverse(self) -> 'QuantumCircuit':
        """逆电路"""
        inv = QuantumCircuit(self.num_qubits, self.num_classical)
        for instr in reversed(self.instructions):
            if instr.operation == CircuitOperation.GATE and instr.gate:
                inv.add_gate(instr.gate.inverse(), instr.qubits)
            else:
                inv.instructions.append(instr)
        return inv

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_qubits': self.num_qubits,
            'num_classical': self.num_classical,
            'instructions': [str(i) for i in self.instructions],
            'depth': self.depth,
            'gate_count': self.gate_count
        }

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits}q, {self.gate_count} gates, depth={self.depth})"



class CircuitOptimizer:
    """电路优化器"""

    def __init__(self):
        self.optimization_passes = [
            self._gate_cancellation,
            self._gate_merging,
            self._rotation_merging,
            self._commutation_optimization
        ]

    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """运行所有优化通道"""
        result = circuit.copy()
        for pass_fn in self.optimization_passes:
            result = pass_fn(result)
        return result

    def _gate_cancellation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """门消除: XX=I, HH=I, etc."""
        optimized = QuantumCircuit(circuit.num_qubits, circuit.num_classical)
        i = 0
        while i < len(circuit.instructions):
            curr = circuit.instructions[i]
            if i + 1 < len(circuit.instructions):
                nxt = circuit.instructions[i + 1]
                if (curr.operation == CircuitOperation.GATE and nxt.operation == CircuitOperation.GATE
                    and curr.gate and nxt.gate and curr.qubits == nxt.qubits):
                    if curr.gate.name == nxt.gate.name and curr.gate.name in ['H', 'X', 'Y', 'Z', 'CNOT', 'SWAP']:
                        i += 2
                        continue
            optimized.instructions.append(curr)
            i += 1
        return optimized

    def _gate_merging(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """门合并: 连续旋转门合并"""
        optimized = QuantumCircuit(circuit.num_qubits, circuit.num_classical)
        i = 0
        while i < len(circuit.instructions):
            curr = circuit.instructions[i]
            if i + 1 < len(circuit.instructions):
                nxt = circuit.instructions[i + 1]
                if (curr.operation == CircuitOperation.GATE and nxt.operation == CircuitOperation.GATE
                    and curr.gate and nxt.gate and curr.qubits == nxt.qubits):
                    # 合并同轴旋转
                    if hasattr(curr.gate, 'theta') and hasattr(nxt.gate, 'theta'):
                        if type(curr.gate).__name__ == type(nxt.gate).__name__:
                            merged_theta = curr.gate.theta + nxt.gate.theta
                            from .quantum_gates import GateFactory
                            merged_gate = GateFactory.create(
                                curr.gate.name.split('(')[0],
                                theta=merged_theta
                            )
                            optimized.add_gate(merged_gate, curr.qubits)
                            i += 2
                            continue
            optimized.instructions.append(curr)
            i += 1
        return optimized

    def _rotation_merging(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """旋转门合并"""
        return circuit  # 简化实现

    def _commutation_optimization(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """交换优化"""
        return circuit  # 简化实现

    def count_optimizations(self, original: QuantumCircuit, optimized: QuantumCircuit) -> Dict:
        return {
            'original_gates': original.gate_count,
            'optimized_gates': optimized.gate_count,
            'gates_saved': original.gate_count - optimized.gate_count,
            'original_depth': original.depth,
            'optimized_depth': optimized.depth,
            'depth_saved': original.depth - optimized.depth
        }


class CircuitCompiler:
    """电路编译器"""

    def __init__(self, target_gates: List[str] = None):
        self.target_gates = target_gates or ['CNOT', 'Rx', 'Ry', 'Rz']

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """编译电路到目标门集"""
        compiled = QuantumCircuit(circuit.num_qubits, circuit.num_classical)
        for instr in circuit.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                decomposed = self._decompose_gate(instr.gate)
                for gate, qubits in decomposed:
                    compiled.add_gate(gate, [instr.qubits[q] for q in qubits])
            else:
                compiled.instructions.append(instr)
        return compiled

    def _decompose_gate(self, gate: 'QuantumGate') -> List[Tuple]:
        """门分解"""
        from .quantum_gates import GateFactory
        if gate.name == 'H':
            return [(GateFactory.create('Ry', theta=np.pi/2), [0]),
                    (GateFactory.create('Rz', theta=np.pi), [0])]
        elif gate.name == 'X':
            return [(GateFactory.create('Rz', theta=np.pi), [0]),
                    (GateFactory.create('Ry', theta=np.pi), [0])]
        elif gate.name == 'Y':
            return [(GateFactory.create('Ry', theta=np.pi), [0])]
        elif gate.name == 'Z':
            return [(GateFactory.create('Rz', theta=np.pi), [0])]
        elif gate.name == 'S':
            return [(GateFactory.create('Rz', theta=np.pi/2), [0])]
        elif gate.name == 'T':
            return [(GateFactory.create('Rz', theta=np.pi/4), [0])]
        return [(gate, list(range(gate.num_qubits)))]

    def gate_cost(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """门代价统计"""
        cost = {}
        for instr in circuit.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                name = instr.gate.name.split('(')[0]
                if instr.gate.num_qubits == 1:
                    cost[name] = cost.get(name, 0) + 1
                else:
                    cost[name] = cost.get(name, 0) + 10  # 双比特门代价更高
        return cost


class CircuitVerifier:
    """电路验证器"""

    def verify_unitarity(self, circuit: QuantumCircuit) -> bool:
        """验证电路的酉性"""
        for instr in circuit.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate:
                if not instr.gate.is_unitary():
                    return False
        return True

    def verify_connectivity(self, circuit: QuantumCircuit, connectivity: List[List[int]]) -> bool:
        """验证电路的连通性约束"""
        for instr in circuit.instructions:
            if instr.operation == CircuitOperation.GATE and instr.gate and instr.gate.num_qubits >= 2:
                q1, q2 = instr.qubits[0], instr.qubits[1]
                if q2 not in connectivity[q1]:
                    return False
        return True

    def verify_qubit_range(self, circuit: QuantumCircuit) -> bool:
        """验证量子比特范围"""
        for instr in circuit.instructions:
            for q in instr.qubits:
                if q < 0 or q >= circuit.num_qubits:
                    return False
        return True

    def get_circuit_properties(self, circuit: QuantumCircuit) -> Dict:
        return {
            'num_qubits': circuit.num_qubits,
            'gate_count': circuit.gate_count,
            'depth': circuit.depth,
            'two_qubit_gates': circuit.two_qubit_gate_count,
            'is_unitary': self.verify_unitarity(circuit),
            'is_valid_range': self.verify_qubit_range(circuit),
            'gate_types': circuit.gate_types
        }


class CircuitSimulator:
    """电路模拟器 - 支持多种模拟模式"""

    def __init__(self, mode: str = "statevector"):
        self.mode = mode

    def simulate_statevector(self, circuit: QuantumCircuit, shots: int = 1024, seed: Optional[int] = None) -> Dict:
        """状态矢量模拟"""
        return circuit.simulate(shots=shots, seed=seed)

    def simulate_density_matrix(self, circuit: QuantumCircuit, noise_model: Optional[Any] = None, shots: int = 1024) -> Dict:
        """密度矩阵模拟"""
        from .quantum_state import DensityMatrix
        n = circuit.num_qubits
        dm = DensityMatrix(n)
        # 简化: 先做状态矢量模拟，再转密度矩阵
        sv_result = circuit.simulate(shots=shots)
        sv = sv_result['state_vector']
        dm = DensityMatrix.from_state_vector(sv)
        if noise_model:
            dm.matrix = noise_model.apply_to_state(dm.matrix, n)
        return {
            'density_matrix': dm.matrix.tolist(),
            'purity': dm.purity(),
            'entropy': dm.von_neumann_entropy(),
            'counts': sv_result['counts']
        }

    def simulate_stabilizer(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict:
        """稳定子模拟（Clifford 电路）"""
        # 简化: 使用状态矢量模拟
        return self.simulate_statevector(circuit, shots)

    def get_supported_modes(self) -> List[str]:
        return ["statevector", "density_matrix", "stabilizer"]
