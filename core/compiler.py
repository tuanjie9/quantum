"""量子编译器 - 门分解/映射/优化/路由"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math


class GateType(Enum):
    """门类型枚举"""
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    THREE_QUBIT = "three_qubit"
    PARAMETRIC = "parametric"


@dataclass
class CompiledGate:
    """编译后的门"""
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    gate_type: GateType = GateType.SINGLE_QUBIT
    matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'qubits': self.qubits,
            'params': self.params,
            'type': self.gate_type.value
        }


@dataclass
class CompilationResult:
    """编译结果"""
    gates: List[CompiledGate]
    original_count: int = 0
    compiled_count: int = 0
    depth: int = 0
    optimization_passes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compression_ratio(self) -> float:
        """压缩比"""
        if self.original_count == 0:
            return 1.0
        return self.compiled_count / self.original_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_count': self.original_count,
            'compiled_count': self.compiled_count,
            'depth': self.depth,
            'compression_ratio': self.compression_ratio(),
            'optimization_passes': self.optimization_passes
        }


class SingleQubitDecomposer:
    """单量子比特门分解为 Rz-Ry-Rz 序列"""

    def __init__(self):
        self.tolerance = 1e-10

    def decompose(self, matrix: np.ndarray) -> List[CompiledGate]:
        """ZYZ 分解: U = e^(iα) Rz(β) Ry(γ) Rz(δ)"""
        if matrix.shape != (2, 2):
            raise ValueError("仅支持 2x2 矩阵")
        det = np.linalg.det(matrix)
        alpha = np.angle(det) / 2
        U = matrix * np.exp(-1j * alpha)
        theta = 2 * np.arccos(min(1.0, abs(U[0, 0])))
        if abs(theta) < self.tolerance:
            return [CompiledGate('Rz', [0], [2 * alpha])]
        phi = np.angle(U[1, 0]) - np.angle(U[0, 0]) + np.pi / 2
        lam = -np.angle(U[1, 0]) - np.angle(U[0, 0]) - np.pi / 2
        gates = []
        if abs(lam) > self.tolerance:
            gates.append(CompiledGate('Rz', [0], [lam], GateType.PARAMETRIC))
        if abs(theta) > self.tolerance:
            gates.append(CompiledGate('Ry', [0], [theta], GateType.PARAMETRIC))
        if abs(phi) > self.tolerance:
            gates.append(CompiledGate('Rz', [0], [phi], GateType.PARAMETRIC))
        return gates

    def decompose_to_u3(self, matrix: np.ndarray) -> CompiledGate:
        """分解为 U3 门"""
        det = np.linalg.det(matrix)
        alpha = np.angle(det) / 2
        U = matrix * np.exp(-1j * alpha)
        theta = 2 * np.arccos(min(1.0, abs(U[0, 0])))
        phi = np.angle(U[1, 0]) - np.angle(U[0, 0]) + np.pi / 2
        lam = -np.angle(U[1, 0]) - np.angle(U[0, 0]) - np.pi / 2
        return CompiledGate('U3', [0], [theta, phi, lam], GateType.PARAMETRIC)


class TwoQubitDecomposer:
    """两量子比特门分解"""

    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    def __init__(self):
        self.single_decomp = SingleQubitDecomposer()

    def decompose_cnot(self, control: int, target: int) -> List[CompiledGate]:
        """CNOT 门保留（基础门）"""
        return [CompiledGate('CNOT', [control, target], gate_type=GateType.TWO_QUBIT)]

    def decompose_cz(self, control: int, target: int) -> List[CompiledGate]:
        """CZ = H(target) · CNOT · H(target)"""
        return [
            CompiledGate('H', [target]),
            CompiledGate('CNOT', [control, target], gate_type=GateType.TWO_QUBIT),
            CompiledGate('H', [target])
        ]

    def decompose_swap(self, q0: int, q1: int) -> List[CompiledGate]:
        """SWAP = CNOT(0,1) · CNOT(1,0) · CNOT(0,1)"""
        return [
            CompiledGate('CNOT', [q0, q1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [q1, q0], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [q0, q1], gate_type=GateType.TWO_QUBIT)
        ]

    def decompose_iswap(self, q0: int, q1: int) -> List[CompiledGate]:
        """iSWAP 分解"""
        return [
            CompiledGate('S', [q0]),
            CompiledGate('S', [q1]),
            CompiledGate('H', [q0]),
            CompiledGate('CNOT', [q0, q1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [q1, q0], gate_type=GateType.TWO_QUBIT),
            CompiledGate('H', [q1])
        ]

    def count_cnots(self, gates: List[CompiledGate]) -> int:
        """统计 CNOT 门数量"""
        return sum(1 for g in gates if g.name == 'CNOT')


class CircuitRouter:
    """量子电路路由器 - 处理量子比特映射"""

    def __init__(self, coupling_map: List[Tuple[int, int]]):
        self.coupling_map = coupling_map
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        """构建邻接表"""
        adj: Dict[int, List[int]] = {}
        for a, b in self.coupling_map:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        return adj

    def shortest_path(self, src: int, dst: int) -> List[int]:
        """BFS 最短路径"""
        if src == dst:
            return [src]
        visited = {src}
        queue = [[src]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in self.adjacency.get(node, []):
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def route_circuit(self, gates: List[CompiledGate], num_physical_qubits: int) -> List[CompiledGate]:
        """路由电路，插入 SWAP 门"""
        mapping = list(range(num_physical_qubits))
        routed = []
        for gate in gates:
            if len(gate.qubits) == 1:
                routed.append(gate)
            elif len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                phys_q0, phys_q1 = mapping[q0], mapping[q1]
                if (phys_q0, phys_q1) in self.coupling_map or (phys_q1, phys_q0) in self.coupling_map:
                    routed.append(CompiledGate(gate.name, [phys_q0, phys_q1],
                                               gate.params, gate.gate_type))
                else:
                    path = self.shortest_path(phys_q0, phys_q1)
                    if len(path) >= 2:
                        for i in range(len(path) - 2):
                            swap_gates = self._swap_gates(path[i], path[i + 1])
                            routed.extend(swap_gates)
                            idx_a = mapping.index(path[i])
                            idx_b = mapping.index(path[i + 1])
                            mapping[idx_a], mapping[idx_b] = mapping[idx_b], mapping[idx_a]
                        routed.append(CompiledGate(gate.name,
                                                   [mapping.index(phys_q0), mapping.index(phys_q1)],
                                                   gate.params, gate.gate_type))
                    else:
                        routed.append(gate)
            else:
                routed.append(gate)
        return routed

    def _swap_gates(self, q0: int, q1: int) -> List[CompiledGate]:
        """SWAP 门的 CNOT 分解"""
        return [
            CompiledGate('CNOT', [q0, q1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [q1, q0], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [q0, q1], gate_type=GateType.TWO_QUBIT)
        ]


class GateOptimizer:
    """门优化器"""

    def __init__(self, max_passes: int = 10):
        self.max_passes = max_passes
        self.optimization_rules = [
            self._remove_identity_pairs,
            self._merge_rotations,
            self._cancel_cnot_pairs,
            self._commute_h_through_cnot,
        ]

    def optimize(self, gates: List[CompiledGate]) -> Tuple[List[CompiledGate], int]:
        """执行多轮优化"""
        optimized = gates.copy()
        total_passes = 0
        for _ in range(self.max_passes):
            changed = False
            for rule in self.optimization_rules:
                new_gates, rule_changed = rule(optimized)
                if rule_changed:
                    optimized = new_gates
                    changed = True
                    total_passes += 1
            if not changed:
                break
        return optimized, total_passes

    def _remove_identity_pairs(self, gates: List[CompiledGate]) -> Tuple[List[CompiledGate], bool]:
        """移除恒等门对 (H·H, X·X, Z·Z)"""
        self_inverse = {'H', 'X', 'Y', 'Z'}
        new_gates = []
        i = 0
        changed = False
        while i < len(gates):
            if (i + 1 < len(gates) and
                gates[i].name in self_inverse and
                gates[i].name == gates[i + 1].name and
                gates[i].qubits == gates[i + 1].qubits):
                i += 2
                changed = True
            else:
                new_gates.append(gates[i])
                i += 1
        return new_gates, changed

    def _merge_rotations(self, gates: List[CompiledGate]) -> Tuple[List[CompiledGate], bool]:
        """合并连续旋转门"""
        rotation_gates = {'Rx', 'Ry', 'Rz'}
        new_gates = []
        i = 0
        changed = False
        while i < len(gates):
            if (i + 1 < len(gates) and
                gates[i].name in rotation_gates and
                gates[i].name == gates[i + 1].name and
                gates[i].qubits == gates[i + 1].qubits):
                total_angle = gates[i].params[0] + gates[i + 1].params[0]
                total_angle = total_angle % (2 * math.pi)
                if abs(total_angle) > 1e-10:
                    new_gates.append(CompiledGate(gates[i].name, gates[i].qubits,
                                                  [total_angle], GateType.PARAMETRIC))
                i += 2
                changed = True
            else:
                new_gates.append(gates[i])
                i += 1
        return new_gates, changed

    def _cancel_cnot_pairs(self, gates: List[CompiledGate]) -> Tuple[List[CompiledGate], bool]:
        """取消相邻 CNOT 对"""
        new_gates = []
        i = 0
        changed = False
        while i < len(gates):
            if (i + 1 < len(gates) and
                gates[i].name == 'CNOT' and gates[i + 1].name == 'CNOT' and
                gates[i].qubits == gates[i + 1].qubits):
                i += 2
                changed = True
            else:
                new_gates.append(gates[i])
                i += 1
        return new_gates, changed

    def _commute_h_through_cnot(self, gates: List[CompiledGate]) -> Tuple[List[CompiledGate], bool]:
        """H 与 CNOT 交换以优化"""
        return gates, False


class CompilerPipeline:
    """编译器管线 - 统一接口"""

    def __init__(self, coupling_map: Optional[List[Tuple[int, int]]] = None):
        self.single_decomp = SingleQubitDecomposer()
        self.two_qubit_decomp = TwoQubitDecomposer()
        self.optimizer = GateOptimizer()
        self.coupling_map = coupling_map
        self.router = CircuitRouter(coupling_map) if coupling_map else None

    def compile(self, gates: List[CompiledGate], num_qubits: int) -> CompilationResult:
        """完整编译流程"""
        original_count = len(gates)
        # 1. 门分解
        decomposed = []
        for gate in gates:
            decomposed.extend(self._decompose_gate(gate))
        # 2. 优化
        optimized, passes = self.optimizer.optimize(decomposed)
        # 3. 路由 (如果有耦合映射)
        if self.router:
            optimized = self.router.route_circuit(optimized, num_qubits)
        depth = self._calculate_depth(optimized, num_qubits)
        return CompilationResult(
            gates=optimized,
            original_count=original_count,
            compiled_count=len(optimized),
            depth=depth,
            optimization_passes=passes
        )

    def _decompose_gate(self, gate: CompiledGate) -> List[CompiledGate]:
        """分解单个门"""
        if gate.name == 'CZ':
            return self.two_qubit_decomp.decompose_cz(gate.qubits[0], gate.qubits[1])
        elif gate.name == 'SWAP':
            return self.two_qubit_decomp.decompose_swap(gate.qubits[0], gate.qubits[1])
        return [gate]

    def _calculate_depth(self, gates: List[CompiledGate], num_qubits: int) -> int:
        """计算电路深度"""
        qubit_depths = [0] * num_qubits
        for gate in gates:
            max_depth = max(qubit_depths[q] for q in gate.qubits if q < num_qubits)
            for q in gate.qubits:
                if q < num_qubits:
                    qubit_depths[q] = max_depth + 1
        return max(qubit_depths) if qubit_depths else 0
