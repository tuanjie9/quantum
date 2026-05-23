"""量子纠错引擎 - Steane/Shor/Surface 重复码"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import random


class ErrorType(Enum):
    """量子错误类型"""
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    BIT_PHASE_FLIP = "bit_phase_flip"
    DEPOLARIZING = "depolarizing"
    ERASURE = "erasure"


class SyndromeResult:
    """校验子解码结果"""
    def __init__(self, syndrome: List[int], error_location: int, error_type: ErrorType, corrected: bool):
        self.syndrome = syndrome
        self.error_location = error_location
        self.error_type = error_type
        self.corrected = corrected

    def __repr__(self):
        return f"Syndrome(syndrome={self.syndrome}, error={self.error_type.value}@{self.error_location}, corrected={self.corrected})"


class RepetCode:
    """重复码 (d=3, 5, 7, ...)"""

    def __init__(self, distance: int = 3):
        if distance % 2 == 0 or distance < 3:
            raise ValueError("重复码距离必须为奇数且 >= 3")
        self.distance = distance
        self.num_data_qubits = distance
        self.num_ancilla_qubits = distance - 1
        self.total_qubits = self.num_data_qubits + self.num_ancilla_qubits

    def encode(self, state: List[complex]) -> np.ndarray:
        """编码逻辑量子比特"""
        logical_0 = np.zeros(2**self.total_qubits, dtype=complex)
        logical_1 = np.zeros(2**self.total_qubits, dtype=complex)
        # |0_L⟩ = |00...0⟩
        logical_0[0] = 1.0
        # |1_L⟩ = |11...1⟩ (数据比特全1)
        idx = sum(1 << i for i in range(self.num_data_qubits))
        logical_1[idx] = 1.0
        return state[0] * logical_0 + state[1] * logical_1

    def syndrome_measurement(self, noisy_state: np.ndarray) -> List[int]:
        """校验子测量"""
        syndrome = []
        for i in range(self.num_ancilla_qubits):
            # Z_i Z_{i+1} 校验
            prob = 0.0
            for idx in range(len(noisy_state)):
                if abs(noisy_state[idx]) > 1e-10:
                    bi = (idx >> i) & 1
                    bi1 = (idx >> (i + 1)) & 1
                    parity = bi ^ bi1
                    prob += abs(noisy_state[idx])**2 * (1 - 2 * parity)
            syndrome.append(0 if prob > 0 else 1)
        return syndrome

    def decode_syndrome(self, syndrome: List[int]) -> SyndromeResult:
        """解码校验子"""
        error_pos = sum(s * 2**i for i, s in enumerate(syndrome))
        if error_pos == 0:
            return SyndromeResult(syndrome, -1, ErrorType.BIT_FLIP, True)
        location = 0
        for i, s in enumerate(syndrome):
            if s == 1:
                location = i
                break
        return SyndromeResult(syndrome, location, ErrorType.BIT_FLIP, True)

    def correct_error(self, state: np.ndarray, syndrome_result: SyndromeResult) -> np.ndarray:
        """纠错"""
        if syndrome_result.error_location < 0:
            return state
        corrected = state.copy()
        n = self.total_qubits
        q = syndrome_result.error_location
        for idx in range(len(state)):
            new_idx = idx ^ (1 << q)
            corrected[new_idx] = state[idx]
        return corrected

    def logical_error_rate(self, physical_error_rate: float) -> float:
        """逻辑错误率估算"""
        t = (self.distance - 1) // 2
        rate = 0.0
        for k in range(t + 1, self.distance):
            rate += self._comb(self.distance, k) * physical_error_rate**k * (1 - physical_error_rate)**(self.distance - k)
        return rate

    def _comb(self, n: int, k: int) -> int:
        """组合数 C(n, k)"""
        if k > n or k < 0:
            return 0
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result


class SteaneCode:
    """Steane [[7,1,3]] 码"""

    def __init__(self):
        self.distance = 3
        self.num_data_qubits = 7
        self.num_ancilla_qubits = 6
        self.total_qubits = 13
        # 稳定子生成元
        self.x_stabilizers = [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1]
        ]
        self.z_stabilizers = [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1]
        ]

    def encode(self, state: List[complex]) -> np.ndarray:
        """编码逻辑量子比特"""
        logical_0 = np.zeros(2**self.total_qubits, dtype=complex)
        logical_1 = np.zeros(2**self.total_qubits, dtype=complex)
        # |0_L⟩ = 均匀叠加偶校验态
        even_parities = [0b0000000, 0b1111000, 0b1100110, 0b1010101,
                         0b0110011, 0b0101010, 0b0011001, 0b1101001]
        norm = 1.0 / np.sqrt(len(even_parities))
        for p in even_parities:
            logical_0[p] = norm
        # |1_L⟩ = 均匀叠加奇校验态
        odd_parities = [0b1111111, 0b0000111, 0b0011001, 0b0101010,
                        0b1001100, 0b1010101, 0b1100110, 0b0010110]
        for p in odd_parities:
            logical_1[p] = norm
        return state[0] * logical_0 + state[1] * logical_1

    def syndrome_measurement(self, noisy_state: np.ndarray) -> Tuple[List[int], List[int]]:
        """X 和 Z 校验子测量"""
        x_syndrome = []
        z_syndrome = []
        for stab in self.x_stabilizers:
            parity = 0
            for idx in range(len(noisy_state)):
                if abs(noisy_state[idx]) > 1e-10:
                    bits = [(idx >> i) & 1 for i in range(7)]
                    s = sum(bits[i] for i in range(7) if stab[i] == 1) % 2
                    parity += abs(noisy_state[idx])**2 * (1 - 2 * s)
            x_syndrome.append(0 if parity > 0 else 1)
        for stab in self.z_stabilizers:
            parity = 0
            for idx in range(len(noisy_state)):
                if abs(noisy_state[idx]) > 1e-10:
                    bits = [(idx >> i) & 1 for i in range(7)]
                    s = sum(bits[i] for i in range(7) if stab[i] == 1) % 2
                    parity += abs(noisy_state[idx])**2 * (1 - 2 * s)
            z_syndrome.append(0 if parity > 0 else 1)
        return x_syndrome, z_syndrome

    def decode_syndrome(self, x_syndrome: List[int], z_syndrome: List[int]) -> Dict[str, Any]:
        """解码校验子"""
        x_pos = sum(s << i for i, s in enumerate(x_syndrome))
        z_pos = sum(s << i for i, s in enumerate(z_syndrome))
        return {
            'x_error': x_pos > 0,
            'z_error': z_pos > 0,
            'x_location': x_pos,
            'z_location': z_pos,
            'correctable': True
        }

    def correct(self, state: np.ndarray, syndrome_info: Dict[str, Any]) -> np.ndarray:
        """Steane 码纠错"""
        corrected = state.copy()
        if syndrome_info['x_error'] and syndrome_info['x_location'] < 7:
            q = syndrome_info['x_location']
            for idx in range(len(corrected)):
                new_idx = idx ^ (1 << q)
                if new_idx < len(corrected):
                    corrected[new_idx] = state[idx]
        if syndrome_info['z_error'] and syndrome_info['z_location'] < 7:
            q = syndrome_info['z_location']
            for idx in range(len(corrected)):
                phase = (-1) ** ((idx >> q) & 1)
                corrected[idx] *= phase
        return corrected


class ShorCode:
    """Shor [[9,1,3]] 码"""

    def __init__(self):
        self.distance = 3
        self.num_data_qubits = 9
        self.total_qubits = 9

    def encode(self, state: List[complex]) -> np.ndarray:
        """Shor 码编码: 3个块 × 3个比特"""
        norm = 1.0 / (2 * np.sqrt(2))
        logical_0 = np.zeros(2**9, dtype=complex)
        logical_1 = np.zeros(2**9, dtype=complex)
        # |0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
        for b1 in [0b000, 0b111]:
            for b2 in [0b000, 0b111]:
                for b3 in [0b000, 0b111]:
                    idx = (b1 << 6) | (b2 << 3) | b3
                    logical_0[idx] = norm
        # |1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2
        for b1 in [0b000, 0b111]:
            for b2 in [0b000, 0b111]:
                for b3 in [0b000, 0b111]:
                    idx = (b1 << 6) | (b2 << 3) | b3
                    sign = 1
                    for block in [b1, b2, b3]:
                        if block == 0b111:
                            sign *= -1
                    logical_1[idx] = norm * sign
        return state[0] * logical_0 + state[1] * logical_1

    def syndrome_measurement(self, noisy_state: np.ndarray) -> Dict[str, List[int]]:
        """Shor 码校验子"""
        bit_flip_syndromes = []
        phase_flip_syndromes = []
        for block in range(3):
            syndrome = []
            for pair in range(2):
                q1 = block * 3 + pair
                q2 = block * 3 + pair + 1
                parity = 0
                for idx in range(len(noisy_state)):
                    if abs(noisy_state[idx]) > 1e-10:
                        b1 = (idx >> q1) & 1
                        b2 = (idx >> q2) & 1
                        p = b1 ^ b2
                        parity += abs(noisy_state[idx])**2 * (1 - 2 * p)
                syndrome.append(0 if parity > 0 else 1)
            bit_flip_syndromes.append(syndrome)
        for pair in range(2):
            q1 = pair * 3
            q2 = (pair + 1) * 3
            syndrome = []
            for i in range(3):
                parity = 0
                for idx in range(len(noisy_state)):
                    if abs(noisy_state[idx]) > 1e-10:
                        b1 = (idx >> (q1 + i)) & 1
                        b2 = (idx >> (q2 + i)) & 1
                        p = b1 ^ b2
                        parity += abs(noisy_state[idx])**2 * (1 - 2 * p)
                syndrome.append(0 if parity > 0 else 1)
            phase_flip_syndromes.append(syndrome)
        return {
            'bit_flip': bit_flip_syndromes,
            'phase_flip': phase_flip_syndromes
        }


class SurfaceCode:
    """Surface 码 (简化的距离 d 码)"""

    def __init__(self, distance: int = 3):
        if distance % 2 == 0 or distance < 3:
            raise ValueError("Surface 码距离必须为奇数且 >= 3")
        self.distance = distance
        self.num_data_qubits = distance**2
        self.num_x_stabilizers = (distance**2 - 1) // 2
        self.num_z_stabilizers = (distance**2 - 1) // 2

    def encode(self, state: List[complex]) -> np.ndarray:
        """编码逻辑量子比特"""
        dim = 2**self.num_data_qubits
        logical_0 = np.zeros(dim, dtype=complex)
        logical_1 = np.zeros(dim, dtype=complex)
        logical_0[0] = 1.0
        logical_1[-1] = 1.0
        norm = 1.0 / np.sqrt(2)
        return state[0] * norm * logical_0 + state[1] * norm * logical_1

    def x_syndrome(self, state: np.ndarray) -> List[int]:
        """X 稳定子校验子"""
        syndrome = []
        for i in range(self.num_x_stabilizers):
            parity = 0
            for idx in range(len(state)):
                if abs(state[idx]) > 1e-10:
                    bits = [(idx >> j) & 1 for j in range(self.num_data_qubits)]
                    relevant = bits[i % self.num_data_qubits] ^ bits[(i + 1) % self.num_data_qubits]
                    parity += abs(state[idx])**2 * (1 - 2 * relevant)
            syndrome.append(0 if parity > 0 else 1)
        return syndrome

    def z_syndrome(self, state: np.ndarray) -> List[int]:
        """Z 稳定子校验子"""
        syndrome = []
        for i in range(self.num_z_stabilizers):
            parity = 0
            for idx in range(len(state)):
                if abs(state[idx]) > 1e-10:
                    bits = [(idx >> j) & 1 for j in range(self.num_data_qubits)]
                    relevant = bits[i % self.num_data_qubits] ^ bits[(i + 2) % self.num_data_qubits]
                    parity += abs(state[idx])**2 * (1 - 2 * relevant)
            syndrome.append(0 if parity > 0 else 1)
        return syndrome

    def decode_mwpm(self, syndrome: List[int]) -> List[int]:
        """最小权重完美匹配解码器"""
        error_positions = []
        for i, s in enumerate(syndrome):
            if s == 1:
                error_positions.append(i)
        # 简化: 最近邻匹配
        corrections = []
        while len(error_positions) >= 2:
            p1 = error_positions.pop(0)
            p2 = error_positions.pop(0)
            corrections.append((p1 + p2) // 2)
        if error_positions:
            corrections.append(error_positions[0])
        return corrections

    def threshold_error_rate(self) -> float:
        """Surface 码阈值错误率"""
        return 0.01  # 约 1% 阈值


class NoiseChannel:
    """量子噪声通道"""

    @staticmethod
    def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
        """去极化通道"""
        dim = rho.shape[0]
        return (1 - p) * rho + p * np.eye(dim, dtype=complex) / dim

    @staticmethod
    def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
        """比特翻转通道"""
        n = int(np.log2(rho.shape[0]))
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        full_X = X
        for _ in range(n - 1):
            full_X = np.kron(full_X, np.eye(2, dtype=complex))
        return (1 - p) * rho + p * full_X @ rho @ full_X.conj().T

    @staticmethod
    def phase_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
        """相位翻转通道"""
        n = int(np.log2(rho.shape[0]))
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        full_Z = Z
        for _ in range(n - 1):
            full_Z = np.kron(full_Z, np.eye(2, dtype=complex))
        return (1 - p) * rho + p * full_Z @ rho @ full_Z.conj().T

    @staticmethod
    def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
        """振幅阻尼通道"""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        if rho.shape[0] > 2:
            n = int(np.log2(rho.shape[0]))
            K0_full = K0
            K1_full = K1
            for _ in range(n - 1):
                K0_full = np.kron(K0_full, np.eye(2, dtype=complex))
                K1_full = np.kron(K1_full, np.eye(2, dtype=complex))
            K0, K1 = K0_full, K1_full
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    @staticmethod
    def phase_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
        """相位阻尼通道"""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        if rho.shape[0] > 2:
            n = int(np.log2(rho.shape[0]))
            K0_full = K0
            K1_full = K1
            for _ in range(n - 1):
                K0_full = np.kron(K0_full, np.eye(2, dtype=complex))
                K1_full = np.kron(K1_full, np.eye(2, dtype=complex))
            K0, K1 = K0_full, K1_full
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    @staticmethod
    def thermal_relaxation(rho: np.ndarray, t: float, t1: float, t2: float) -> np.ndarray:
        """热弛豫通道"""
        e_t1 = np.exp(-t / t1)
        e_t2 = np.exp(-t / t2)
        dim = rho.shape[0]
        result = rho.copy()
        # T1 衰减 (对角元素)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    result[i, j] *= e_t2
        # T1 弛豫到基态
        result[0, 0] = 1 - (1 - result[0, 0]) * e_t1
        for i in range(1, dim):
            result[i, i] *= e_t1
        return result

    @staticmethod
    def readout_error(counts: Dict[str, int], p0_1: float, p1_0: float) -> Dict[str, int]:
        """读出误差"""
        noisy_counts = {}
        for bitstring, count in counts.items():
            new_bits = []
            for bit in bitstring:
                if bit == '0':
                    new_bits.append('1' if random.random() < p0_1 else '0')
                else:
                    new_bits.append('0' if random.random() < p1_0 else '1')
            new_key = ''.join(new_bits)
            noisy_counts[new_key] = noisy_counts.get(new_key, 0) + count
        return noisy_counts


class ErrorCorrectionEngine:
    """纠错引擎 - 统一接口"""

    def __init__(self, code_type: str = "repetition", distance: int = 3):
        self.code_type = code_type
        self.distance = distance
        if code_type == "repetition":
            self.code = RepetCode(distance)
        elif code_type == "steane":
            self.code = SteaneCode()
        elif code_type == "shor":
            self.code = ShorCode()
        elif code_type == "surface":
            self.code = SurfaceCode(distance)
        else:
            raise ValueError(f"不支持的纠错码: {code_type}")

    def encode(self, state: List[complex]) -> np.ndarray:
        """编码逻辑量子比特"""
        return self.code.encode(state)

    def detect_and_correct(self, noisy_state: np.ndarray) -> Dict[str, Any]:
        """检测并纠正错误"""
        if isinstance(self.code, RepetCode):
            syndrome = self.code.syndrome_measurement(noisy_state)
            result = self.code.decode_syndrome(syndrome)
            corrected = self.code.correct_error(noisy_state, result)
            return {'syndrome': syndrome, 'result': result, 'corrected': corrected}
        elif isinstance(self.code, SteaneCode):
            x_s, z_s = self.code.syndrome_measurement(noisy_state)
            info = self.code.decode_syndrome(x_s, z_s)
            corrected = self.code.correct(noisy_state, info)
            return {'x_syndrome': x_s, 'z_syndrome': z_s, 'info': info, 'corrected': corrected}
        elif isinstance(self.code, ShorCode):
            syndromes = self.code.syndrome_measurement(noisy_state)
            return {'syndromes': syndromes, 'corrected': noisy_state}
        elif isinstance(self.code, SurfaceCode):
            x_s = self.code.x_syndrome(noisy_state)
            z_s = self.code.z_syndrome(noisy_state)
            x_corrections = self.code.decode_mwpm(x_s)
            z_corrections = self.code.decode_mwpm(z_s)
            return {'x_syndrome': x_s, 'z_syndrome': z_s,
                    'x_corrections': x_corrections, 'z_corrections': z_corrections,
                    'corrected': noisy_state}
        return {'corrected': noisy_state}

    def logical_error_rate(self, physical_rate: float) -> float:
        """估算逻辑错误率"""
        if isinstance(self.code, RepetCode):
            return self.code.logical_error_rate(physical_rate)
        elif isinstance(self.code, SurfaceCode):
            p_th = self.code.threshold_error_rate()
            if physical_rate < p_th:
                return (physical_rate / p_th) ** ((self.distance + 1) / 2)
            return physical_rate
        return physical_rate  # 简化估算

    def get_code_params(self) -> Dict[str, int]:
        """获取纠错码参数"""
        return {
            'code_type': self.code_type,
            'distance': self.distance,
            'data_qubits': self.code.num_data_qubits if hasattr(self.code, 'num_data_qubits') else 0,
            'total_qubits': self.code.total_qubits if hasattr(self.code, 'total_qubits') else 0
        }
