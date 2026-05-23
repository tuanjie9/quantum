"""量子噪声建模"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import math


class NoiseModelType(Enum):
    """噪声模型类型"""
    IDEAL = "ideal"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    THERMAL = "thermal"
    CUSTOM = "custom"


@dataclass
class NoiseParameters:
    """噪声参数"""
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    t1_time: float = 50e-6
    t2_time: float = 70e-6
    gate_time_1q: float = 20e-9
    gate_time_2q: float = 300e-9
    temperature: float = 0.015  # Kelvin
    readout_error_0: float = 0.01
    readout_error_1: float = 0.02
    crosstalk_strength: float = 0.005
    coherent_error_angle: float = 0.001


class DepolarizingChannel:
    """去极化噪声通道"""

    def __init__(self, p: float):
        self.p = p

    def kraus_operators(self) -> List[np.ndarray]:
        """Kraus 算子"""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        K0 = np.sqrt(1 - 3 * self.p / 4) * I
        K1 = np.sqrt(self.p / 4) * X
        K2 = np.sqrt(self.p / 4) * Y
        K3 = np.sqrt(self.p / 4) * Z
        return [K0, K1, K2, K3]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """应用去极化通道"""
        kraus = self.kraus_operators()
        result = np.zeros_like(rho)
        for K in kraus:
            result += K @ rho @ K.conj().T
        return result

    def two_qubit_kraus(self) -> List[np.ndarray]:
        """双量子比特去极化通道"""
        p = self.p
        dim = 4
        K = []
        paulis = [np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
        for i in range(4):
            for j in range(4):
                op = np.kron(paulis[i], paulis[j])
                if i == 0 and j == 0:
                    K.append(np.sqrt(1 - 15*p/16) * op)
                else:
                    K.append(np.sqrt(p/16) * op)
        return K


class AmplitudeDampingChannel:
    """振幅阻尼通道"""

    def __init__(self, gamma: float):
        self.gamma = gamma

    def kraus_operators(self) -> List[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]], dtype=complex)
        return [K0, K1]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        kraus = self.kraus_operators()
        result = np.zeros_like(rho)
        for K in kraus:
            result += K @ rho @ K.conj().T
        return result

    def steady_state(self) -> np.ndarray:
        """稳态"""
        return np.array([[1, 0], [0, 0]], dtype=complex)


class PhaseDampingChannel:
    """相位阻尼通道"""

    def __init__(self, gamma: float):
        self.gamma = gamma

    def kraus_operators(self) -> List[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(self.gamma)]], dtype=complex)
        return [K0, K1]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        kraus = self.kraus_operators()
        result = np.zeros_like(rho)
        for K in kraus:
            result += K @ rho @ K.conj().T
        return result


class ThermalRelaxationChannel:
    """热弛豫通道"""

    def __init__(self, t1: float, t2: float, gate_time: float, temperature: float):
        self.t1 = t1
        self.t2 = t2
        self.gate_time = gate_time
        self.temperature = temperature

    def apply(self, rho: np.ndarray) -> np.ndarray:
        t = self.gate_time
        p_reset = 1 - np.exp(-t / self.t1)
        p_phase = np.exp(-t / self.t2) / np.exp(-t / (2 * self.t1))
        n_thermal = 1 / (np.exp(6.626e-34 * 5e9 / (1.38e-23 * self.temperature)) - 1) if self.temperature > 0 else 0
        p_excited = p_reset * n_thermal / (1 + 2 * n_thermal)
        p_ground = p_reset / (1 + 2 * n_thermal)
        result = rho.copy()
        # 相位退相干
        result[0, 1] *= p_phase
        result[1, 0] *= p_phase
        # 热弛豫
        trace_0 = result[0, 0]
        trace_1 = result[1, 1]
        result[0, 0] = trace_0 * (1 - p_excited) + trace_1 * p_ground
        result[1, 1] = trace_0 * p_excited + trace_1 * (1 - p_ground)
        return result


class CrosstalkNoise:
    """串扰噪声"""

    def __init__(self, strength: float = 0.005, connectivity: Optional[List[List[int]]] = None):
        self.strength = strength
        self.connectivity = connectivity

    def apply(self, state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """应用串扰噪声"""
        result = state_vector.copy()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if self.connectivity and j not in self.connectivity[i]:
                    continue
                # ZZ 串扰
                phase = self.strength * np.random.randn()
                for idx in range(len(state_vector)):
                    bi = (idx >> i) & 1
                    bj = (idx >> j) & 1
                    result[idx] *= np.exp(1j * phase * (2*bi - 1) * (2*bj - 1))
        return result / np.linalg.norm(result)


class CoherentError:
    """相干误差"""

    def __init__(self, angle: float = 0.001):
        self.angle = angle

    def apply_single(self, gate_matrix: np.ndarray) -> np.ndarray:
        """对单量子比特门施加相干误差"""
        # 过/欠旋转误差
        rotation_error = np.array([
            [np.cos(self.angle/2), -np.sin(self.angle/2)],
            [np.sin(self.angle/2), np.cos(self.angle/2)]
        ], dtype=complex)
        return rotation_error @ gate_matrix

    def apply_two(self, gate_matrix: np.ndarray) -> np.ndarray:
        """对双量子比特门施加相干误差"""
        error = np.eye(4, dtype=complex)
        error[1, 1] = np.cos(self.angle)
        error[1, 2] = -np.sin(self.angle)
        error[2, 1] = np.sin(self.angle)
        error[2, 2] = np.cos(self.angle)
        return error @ gate_matrix


class ReadoutNoise:
    """读出噪声"""

    def __init__(self, p0_1: float = 0.01, p1_0: float = 0.02):
        self.p0_1 = p0_1
        self.p1_0 = p1_0

    def apply(self, counts: Dict[str, int]) -> Dict[str, int]:
        """对测量结果施加读出误差"""
        noisy_counts = {}
        for bitstring, count in counts.items():
            new_bits = []
            for bit in bitstring:
                r = np.random.random()
                if bit == '0':
                    new_bits.append('1' if r < self.p0_1 else '0')
                else:
                    new_bits.append('0' if r < self.p1_0 else '1')
            new_key = ''.join(new_bits)
            noisy_counts[new_key] = noisy_counts.get(new_key, 0) + count
        return noisy_counts


class NoiseModel:
    """统一噪声模型"""

    def __init__(self, model_type: NoiseModelType = NoiseModelType.IDEAL, params: Optional[NoiseParameters] = None):
        self.model_type = model_type
        self.params = params or NoiseParameters()
        self.channels: Dict[str, Any] = {}
        self._build_model()

    def _build_model(self):
        """构建噪声模型"""
        if self.model_type == NoiseModelType.DEPOLARIZING:
            self.channels['single'] = DepolarizingChannel(self.params.single_qubit_error)
            self.channels['two'] = DepolarizingChannel(self.params.two_qubit_error)
        elif self.model_type == NoiseModelType.AMPLITUDE_DAMPING:
            gamma = 1 - np.exp(-self.params.gate_time_1q / self.params.t1_time)
            self.channels['single'] = AmplitudeDampingChannel(gamma)
        elif self.model_type == NoiseModelType.PHASE_DAMPING:
            gamma = 1 - np.exp(-self.params.gate_time_1q / self.params.t2_time)
            self.channels['single'] = PhaseDampingChannel(gamma)
        elif self.model_type == NoiseModelType.THERMAL:
            self.channels['single'] = ThermalRelaxationChannel(
                self.params.t1_time, self.params.t2_time,
                self.params.gate_time_1q, self.params.temperature
            )
        self.channels['readout'] = ReadoutNoise(
            self.params.readout_error_0, self.params.readout_error_1
        )
        self.channels['crosstalk'] = CrosstalkNoise(self.params.crosstalk_strength)
        self.channels['coherent'] = CoherentError(self.params.coherent_error_angle)

    def apply_to_state(self, state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """对量子态施加噪声"""
        if self.model_type == NoiseModelType.IDEAL:
            return state_vector
        if 'single' in self.channels:
            # 简化: 对每个量子比特施加噪声
            result = state_vector.copy()
            if hasattr(self.channels['single'], 'apply'):
                # 需要密度矩阵形式
                rho = np.outer(result, result.conj())
                rho = self.channels['single'].apply(rho)
                # 提取态矢量（取主特征向量）
                eigenvalues, eigenvectors = np.linalg.eigh(rho)
                result = eigenvectors[:, -1]
            return result / np.linalg.norm(result)
        return state_vector

    def apply_to_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        """对测量结果施加读出噪声"""
        if 'readout' in self.channels:
            return self.channels['readout'].apply(counts)
        return counts

    def apply_crosstalk(self, state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """施加串扰噪声"""
        if 'crosstalk' in self.channels:
            return self.channels['crosstalk'].apply(state_vector, num_qubits)
        return state_vector

    def coherent_error_matrix(self, gate_matrix: np.ndarray, num_qubits: int) -> np.ndarray:
        """施加相干误差"""
        if 'coherent' in self.channels:
            if num_qubits == 1:
                return self.channels['coherent'].apply_single(gate_matrix)
            elif num_qubits == 2:
                return self.channels['coherent'].apply_two(gate_matrix)
        return gate_matrix

    def get_noise_level(self) -> Dict[str, float]:
        """获取噪声水平"""
        return {
            'single_qubit_error': self.params.single_qubit_error,
            'two_qubit_error': self.params.two_qubit_error,
            'readout_error_0': self.params.readout_error_0,
            'readout_error_1': self.params.readout_error_1,
            'crosstalk': self.params.crosstalk_strength,
            'coherent_error': self.params.coherent_error_angle
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'noise_level': self.get_noise_level()
        }

    def __repr__(self) -> str:
        return f"NoiseModel({self.model_type.value})"



class NoiseCharacterizer:
    """噪声特征化工具"""

    def __init__(self, num_qubits: int = 1):
        self.num_qubits = num_qubits
        self.measurements: List[Dict] = []

    def randomized_benchmarking(self, depths: List[int], num_samples: int = 20, seed: Optional[int] = None) -> Dict:
        """随机基准测试"""
        rng = np.random.RandomState(seed)
        results = {}
        for depth in depths:
            fidelities = []
            for _ in range(num_samples):
                # 生成随机 Clifford 序列
                fidelity = 1.0
                for _ in range(depth):
                    # 简化: 随机门误差累积
                    error_rate = 0.001 * (1 + 0.1 * rng.randn())
                    fidelity *= (1 - max(0, error_rate))
                fidelities.append(fidelity)
            results[depth] = {
                'mean_fidelity': float(np.mean(fidelities)),
                'std_fidelity': float(np.std(fidelities)),
                'min_fidelity': float(np.min(fidelities)),
                'max_fidelity': float(np.max(fidelities))
            }
        self.measurements.append({'type': 'rb', 'results': results})
        return results

    def process_tomography_noise(self, ideal_gates: List[np.ndarray], noisy_gates: List[np.ndarray]) -> Dict:
        """过程层析噪声分析"""
        results = []
        for ideal, noisy in zip(ideal_gates, noisy_gates):
            if ideal.shape != noisy.shape:
                continue
            fidelity = float(np.abs(np.trace(ideal.conj().T @ noisy)) / ideal.shape[0])
            error = 1 - fidelity
            results.append({
                'fidelity': fidelity,
                'error': error,
                'diamond_norm': error * 2  # 简化
            })
        return {
            'gate_fidelities': results,
            'average_fidelity': float(np.mean([r['fidelity'] for r in results])) if results else 0,
            'average_error': float(np.mean([r['error'] for r in results])) if results else 0
        }

    def estimate_t1(self, delays: List[float], populations: List[float]) -> float:
        """估计 T1 弛豫时间"""
        # 指数拟合: P(t) = exp(-t/T1)
        if len(delays) < 2:
            return 0.0
        log_pops = [np.log(max(p, 1e-10)) for p in populations]
        # 简化: 线性回归
        n = len(delays)
        sum_x = sum(delays)
        sum_y = sum(log_pops)
        sum_xy = sum(d * p for d, p in zip(delays, log_pops))
        sum_x2 = sum(d**2 for d in delays)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)
        t1 = -1.0 / slope if slope < 0 else float('inf')
        return t1

    def estimate_t2(self, delays: List[float], coherences: List[float]) -> float:
        """估计 T2 退相干时间"""
        return self.estimate_t1(delays, coherences)  # 简化: 使用相同方法

    def gate_set_tomography(self, num_gates: int = 3) -> Dict:
        """门集层析"""
        results = {}
        for i in range(num_gates):
            # 简化: 生成理想和噪声门
            ideal_fidelity = 1.0
            noisy_fidelity = 1.0 - 0.001 * (i + 1)
            results[f'gate_{i}'] = {
                'ideal_fidelity': ideal_fidelity,
                'noisy_fidelity': noisy_fidelity,
                'error_rate': 1 - noisy_fidelity
            }
        return results


class NoiseMitigator:
    """噪声缓解工具"""

    def __init__(self):
        self.calibration_data: Dict[str, Any] = {}

    def zero_noise_extrapolation(self, results: List[Dict], noise_levels: List[float], target_noise: float = 0.0) -> Dict:
        """零噪声外推"""
        if len(results) < 2:
            return results[0] if results else {}
        # 线性外推
        values = [r.get('value', 0) for r in results]
        n = len(noise_levels)
        sum_x = sum(noise_levels)
        sum_y = sum(values)
        sum_xy = sum(n * v for n, v in zip(noise_levels, values))
        sum_x2 = sum(n**2 for n in noise_levels)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)
        intercept = (sum_y - slope * sum_x) / n
        extrapolated = slope * target_noise + intercept
        return {
            'extrapolated_value': float(extrapolated),
            'original_values': values,
            'noise_levels': noise_levels,
            'slope': float(slope),
            'intercept': float(intercept)
        }

    def probabilistic_error_cancellation(self, counts: Dict[str, int], noise_model: NoiseModel) -> Dict[str, float]:
        """概率误差消除"""
        total = sum(counts.values())
        mitigated = {}
        for bitstring, count in counts.items():
            # 简化: 使用噪声模型概率修正
            prob = count / total
            # 假设去极化噪声 p
            p = noise_model.params.single_qubit_error
            corrected_prob = (prob - p/4) / (1 - p) if prob > p/4 else 0
            mitigated[bitstring] = corrected_prob
        # 归一化
        total_corrected = sum(mitigated.values())
        if total_corrected > 0:
            for k in mitigated:
                mitigated[k] /= total_corrected
        return mitigated

    def measurement_error_mitigation(self, counts: Dict[str, int], calibration_matrix: np.ndarray) -> Dict[str, float]:
        """测量误差缓解"""
        n_bits = len(list(counts.keys())[0]) if counts else 0
        dim = 2**n_bits
        raw_probs = np.zeros(dim)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            raw_probs[idx] = count / total
        # 逆校准矩阵
        try:
            inv_cal = np.linalg.inv(calibration_matrix)
            corrected_probs = inv_cal @ raw_probs
            corrected_probs = np.maximum(corrected_probs, 0)
            corrected_probs /= corrected_probs.sum()
        except np.linalg.LinAlgError:
            corrected_probs = raw_probs
        result = {}
        for i in range(dim):
            bitstring = format(i, f'0{n_bits}')
            result[bitstring] = float(corrected_probs[i])
        return result

    def virtual_distillation(self, states: List[np.ndarray]) -> np.ndarray:
        """虚拟蒸馏"""
        if not states:
            return np.array([])
        avg_state = np.mean(states, axis=0)
        return avg_state / np.linalg.norm(avg_state)


class QuantumErrorBudget:
    """量子误差预算"""

    def __init__(self, total_budget: float = 0.01):
        self.total_budget = total_budget
        self.allocations: Dict[str, float] = {}

    def allocate(self, component: str, budget: float) -> bool:
        remaining = self.total_budget - sum(self.allocations.values())
        if budget > remaining:
            return False
        self.allocations[component] = budget
        return True

    def get_remaining(self) -> float:
        return self.total_budget - sum(self.allocations.values())

    def get_utilization(self) -> float:
        return sum(self.allocations.values()) / self.total_budget if self.total_budget > 0 else 0

    def get_report(self) -> Dict:
        return {
            'total_budget': self.total_budget,
            'allocated': sum(self.allocations.values()),
            'remaining': self.get_remaining(),
            'utilization': self.get_utilization(),
            'components': self.allocations.copy()
        }
