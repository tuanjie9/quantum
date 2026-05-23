"""量子通道理论 - Kraus 表示/Pauli 通道/自定义通道/通道组合"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import math


@dataclass
class ChannelResult:
    """通道运算结果"""
    output_state: np.ndarray
    channel_name: str = ""
    fidelity: float = 0.0
    purity: float = 0.0
    trace_preserved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_name': self.channel_name,
            'fidelity': self.fidelity,
            'purity': self.purity,
            'trace_preserved': self.trace_preserved
        }


class PauliChannel:
    """Pauli 通道: ρ → pI·ρ + pX·XρX + pY·YρY + pZ·ZρZ"""

    def __init__(self, p_i: float = 0.0, p_x: float = 0.0, p_y: float = 0.0, p_z: float = 0.0):
        total = p_i + p_x + p_y + p_z
        if abs(total - 1.0) > 1e-6:
            p_i = 1.0 - p_x - p_y - p_z
        self.p_i = p_i
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """应用 Pauli 通道"""
        return (self.p_i * rho +
                self.p_x * self.X @ rho @ self.X +
                self.p_y * self.Y @ rho @ self.Y +
                self.p_z * self.Z @ rho @ self.Z)

    def kraus_operators(self) -> List[np.ndarray]:
        """Kraus 表示"""
        ops = []
        I = np.eye(2, dtype=complex)
        if self.p_i > 1e-15:
            ops.append(np.sqrt(self.p_i) * I)
        if self.p_x > 1e-15:
            ops.append(np.sqrt(self.p_x) * self.X)
        if self.p_y > 1e-15:
            ops.append(np.sqrt(self.p_y) * self.Y)
        if self.p_z > 1e-15:
            ops.append(np.sqrt(self.p_z) * self.Z)
        return ops

    def process_fidelity(self, rho_in: np.ndarray, rho_out: np.ndarray) -> float:
        """过程保真度"""
        rho_ideal = rho_in
        return float(np.real(np.trace(rho_ideal @ rho_out)))


class GeneralizedAmplitudeDamping:
    """广义振幅阻尼通道 (有限温度振幅阻尼)"""

    def __init__(self, gamma: float, n_bar: float = 0.0):
        self.gamma = gamma
        self.n_bar = n_bar
        # 发射率和吸收率
        self._gamma_down = gamma * (1 + n_bar)  # 发射
        self._gamma_up = gamma * n_bar           # 吸收

    def kraus_operators(self) -> List[np.ndarray]:
        """Kraus 算子: K0=对角衰减, K1=发射, K2=吸收"""
        gd = min(self._gamma_down, 1.0)
        gu = min(self._gamma_up, 1.0)
        K0 = np.array([[np.sqrt(1 - gu), 0], [0, np.sqrt(1 - gd)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gd)], [0, 0]], dtype=complex)
        ops = [K0, K1]
        if gu > 1e-15:
            K2 = np.array([[0, 0], [np.sqrt(gu), 0]], dtype=complex)
            ops.append(K2)
        return ops

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """应用通道"""
        result = np.zeros_like(rho)
        for K in self.kraus_operators():
            result += K @ rho @ K.conj().T
        return result


class PhaseDampingChannel:
    """相位阻尼通道"""

    def __init__(self, gamma: float):
        self.gamma = gamma

    def kraus_operators(self) -> List[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(self.gamma)]], dtype=complex)
        return [K0, K1]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        result = np.zeros_like(rho)
        for K in self.kraus_operators():
            result += K @ rho @ K.conj().T
        return result


class RandomUnitaryChannel:
    """随机酉通道"""

    def __init__(self, probabilities: List[float], unitaries: List[np.ndarray], seed: Optional[int] = None):
        self.probabilities = probabilities
        self.unitaries = unitaries
        self.rng = np.random.RandomState(seed)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """应用随机酉通道"""
        result = np.zeros_like(rho)
        for p, U in zip(self.probabilities, self.unitaries):
            result += p * U @ rho @ U.conj().T
        return result

    def kraus_operators(self) -> List[np.ndarray]:
        return [np.sqrt(p) * U for p, U in zip(self.probabilities, self.unitaries)]

    @classmethod
    def random_pauli_channel(cls, p_x: float, p_y: float, p_z: float) -> 'RandomUnitaryChannel':
        """创建随机 Pauli 通道"""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        p_i = 1 - p_x - p_y - p_z
        return cls([p_i, p_x, p_y, p_z], [I, X, Y, Z])


class ChannelCompositor:
    """通道组合器"""

    @staticmethod
    def compose(channel1_kraus: List[np.ndarray], channel2_kraus: List[np.ndarray]) -> List[np.ndarray]:
        """通道组合: E = E2 ∘ E1"""
        composed = []
        for K2 in channel2_kraus:
            for K1 in channel1_kraus:
                composed.append(K2 @ K1)
        return composed

    @staticmethod
    def tensor_product(channel1_kraus: List[np.ndarray], channel2_kraus: List[np.ndarray]) -> List[np.ndarray]:
        """通道张量积"""
        result = []
        for K1 in channel1_kraus:
            for K2 in channel2_kraus:
                result.append(np.kron(K1, K2))
        return result

    @staticmethod
    def verify_complete(kraus_ops: List[np.ndarray], tolerance: float = 1e-8) -> bool:
        """验证 Kraus 算子完备性: Σ Kᵢ†Kᵢ = I"""
        dim = kraus_ops[0].shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        for K in kraus_ops:
            total += K.conj().T @ K
        return bool(np.allclose(total, np.eye(dim), atol=tolerance))

    @staticmethod
    def channel_fidelity(kraus1: List[np.ndarray], kraus2: List[np.ndarray]) -> float:
        """两个通道的平均保真度"""
        dim = kraus1[0].shape[0]
        total = 0.0
        num_samples = dim * dim
        for i in range(dim):
            for j in range(dim):
                rho = np.zeros((dim, dim), dtype=complex)
                rho[i, j] = 1.0
                out1 = np.zeros_like(rho)
                out2 = np.zeros_like(rho)
                for K in kraus1:
                    out1 += K @ rho @ K.conj().T
                for K in kraus2:
                    out2 += K @ rho @ K.conj().T
                total += np.real(np.trace(out1 @ out2))
        return float(total / num_samples)


class ChannelAnalyzer:
    """通道分析器 - 统一接口"""

    def __init__(self):
        self.results: List[ChannelResult] = []

    def apply_channel(self, rho: np.ndarray, kraus_ops: List[np.ndarray],
                      name: str = "custom") -> ChannelResult:
        """应用通道并返回结果"""
        result = np.zeros_like(rho)
        for K in kraus_ops:
            result += K @ rho @ K.conj().T
        purity = float(np.real(np.trace(result @ result)))
        trace_ok = abs(np.trace(result) - 1.0) < 1e-8
        fidelity = float(np.real(np.trace(rho @ result)))
        ch_result = ChannelResult(
            output_state=result, channel_name=name,
            fidelity=fidelity, purity=purity, trace_preserved=trace_ok
        )
        self.results.append(ch_result)
        return ch_result

    def compare_channels(self, rho: np.ndarray, kraus_ops_list: List[List[np.ndarray]],
                         names: List[str]) -> Dict[str, Dict[str, float]]:
        """比较多个通道的效果"""
        comparison = {}
        for kraus_ops, name in zip(kraus_ops_list, names):
            result = self.apply_channel(rho, kraus_ops, name)
            comparison[name] = {
                'fidelity': result.fidelity,
                'purity': result.purity,
                'trace_preserved': result.trace_preserved
            }
        return comparison

    def noise_strength(self, kraus_ops: List[np.ndarray]) -> float:
        """估计噪声强度"""
        if not kraus_ops:
            return 0.0
        dim = kraus_ops[0].shape[0]
        I = np.eye(dim, dtype=complex)
        total = np.zeros((dim, dim), dtype=complex)
        for K in kraus_ops:
            total += K.conj().T @ K
        return float(np.linalg.norm(total - I, 'fro'))

    def get_history(self) -> List[Dict]:
        return [r.to_dict() for r in self.results]
