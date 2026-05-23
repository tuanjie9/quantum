"""量子引擎全局配置"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class QuantumBackend(Enum):
    """量子模拟后端"""
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    STABILIZER = "stabilizer"
    MPS = "matrix_product_state"


class NoiseType(Enum):
    """噪声类型"""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    THERMAL_RELAXATION = "thermal_relaxation"
    READOUT_ERROR = "readout_error"
    CROSSTALK = "crosstalk"
    COHERENT = "coherent"


class OptimizerType(Enum):
    """优化器类型"""
    COBYLA = "cobyla"
    SPSA = "spsa"
    ADAM = "adam"
    NATURAL_GRADIENT = "natural_gradient"
    L_BFGS = "l_bfgs"
    NELDER_MEAD = "nelder_mead"
    POWELL = "powell"


class ErrorCode(Enum):
    """纠错码类型"""
    REPETITION = "repetition"
    STEANE = "steane"
    SHOR = "shor"
    SURFACE = "surface"
    CSS = "css"
    BACON_SHOR = "bacon_shor"


@dataclass
class SimulationConfig:
    """模拟配置"""
    backend: QuantumBackend = QuantumBackend.STATEVECTOR
    max_qubits: int = 20
    shots: int = 1024
    precision: float = 1e-10
    seed: Optional[int] = None
    parallel: bool = True
    max_threads: int = 4


@dataclass
class NoiseConfig:
    """噪声配置"""
    enabled: bool = False
    noise_type: NoiseType = NoiseType.DEPOLARIZING
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    t1_time: float = 50e-6  # 退相干时间 T1
    t2_time: float = 70e-6  # 退相干时间 T2
    gate_time: float = 20e-9  # 门操作时间
    temperature: float = 0.015  # 温度 (Kelvin)
    readout_error_0: float = 0.01
    readout_error_1: float = 0.02


@dataclass
class OptimizationConfig:
    """优化配置"""
    optimizer: OptimizerType = OptimizerType.COBYLA
    max_iterations: int = 1000
    tolerance: float = 1e-8
    learning_rate: float = 0.01
    momentum: float = 0.9
    parameter_shift: float = 0.01
    num_restarts: int = 10
    convergence_window: int = 50


@dataclass
class ErrorCorrectionConfig:
    """纠错配置"""
    enabled: bool = True
    code: ErrorCode = ErrorCode.STEANE
    code_distance: int = 3
    syndrome_rounds: int = 3
    decoder: str = "mwpm"
    logical_error_rate: float = 1e-6


@dataclass
class PipelineConfig:
    """管道配置"""
    layers: List[str] = field(default_factory=lambda: [
        "circuit_design", "gate_synthesis", "state_preparation",
        "quantum_evolution", "error_correction", "measurement", "optimization"
    ])
    parallel_layers: bool = True
    timeout_per_layer: float = 300.0
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class AgentConfig:
    """Agent 配置"""
    num_agents: int = 10
    max_tasks_per_agent: int = 5
    task_timeout: float = 60.0
    enable_evolution: bool = True
    evolution_rate: float = 0.1
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7


@dataclass
class QuantumConfig:
    """全局量子引擎配置"""
    project_name: str = "Quantum"
    version: str = "0.1.0"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    error_correction: ErrorCorrectionConfig = field(default_factory=ErrorCorrectionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    log_level: str = "INFO"
    output_dir: str = "/data/quantum/output"
    cache_dir: str = "/data/quantum/.cache"

    def validate(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        if self.simulation.max_qubits < 1 or self.simulation.max_qubits > 30:
            errors.append(f"max_qubits must be 1-30, got {self.simulation.max_qubits}")
        if self.simulation.shots < 1:
            errors.append(f"shots must be > 0, got {self.simulation.shots}")
        if self.noise.single_qubit_error < 0 or self.noise.single_qubit_error > 1:
            errors.append(f"single_qubit_error must be 0-1")
        if self.optimization.max_iterations < 1:
            errors.append(f"max_iterations must be > 0")
        if self.error_correction.code_distance < 1:
            errors.append(f"code_distance must be > 0")
        return errors

    def to_dict(self) -> Dict:
        """序列化为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'QuantumConfig':
        """从字典反序列化"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# 全局默认配置
DEFAULT_CONFIG = QuantumConfig()



class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_simulation_config(config: SimulationConfig) -> List[str]:
        errors = []
        if config.max_qubits < 1 or config.max_qubits > 30:
            errors.append(f"max_qubits must be 1-30, got {config.max_qubits}")
        if config.shots < 1:
            errors.append(f"shots must be > 0")
        if config.precision < 0:
            errors.append(f"precision must be >= 0")
        return errors

    @staticmethod
    def validate_noise_config(config: NoiseConfig) -> List[str]:
        errors = []
        if config.single_qubit_error < 0 or config.single_qubit_error > 1:
            errors.append(f"single_qubit_error must be 0-1")
        if config.two_qubit_error < 0 or config.two_qubit_error > 1:
            errors.append(f"two_qubit_error must be 0-1")
        if config.t1_time <= 0:
            errors.append(f"t1_time must be > 0")
        if config.t2_time <= 0:
            errors.append(f"t2_time must be > 0")
        if config.temperature < 0:
            errors.append(f"temperature must be >= 0")
        return errors

    @staticmethod
    def validate_optimization_config(config: OptimizationConfig) -> List[str]:
        errors = []
        if config.max_iterations < 1:
            errors.append(f"max_iterations must be > 0")
        if config.tolerance < 0:
            errors.append(f"tolerance must be >= 0")
        if config.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0")
        return errors

    @staticmethod
    def validate_all(config: QuantumConfig) -> List[str]:
        errors = []
        errors.extend(ConfigValidator.validate_simulation_config(config.simulation))
        errors.extend(ConfigValidator.validate_noise_config(config.noise))
        errors.extend(ConfigValidator.validate_optimization_config(config.optimization))
        return errors


class ConfigPresets:
    """配置预设"""

    @staticmethod
    def high_precision() -> QuantumConfig:
        config = QuantumConfig()
        config.simulation.precision = 1e-15
        config.simulation.shots = 10000
        config.optimization.max_iterations = 5000
        config.optimization.tolerance = 1e-12
        return config

    @staticmethod
    def noisy_hardware() -> QuantumConfig:
        config = QuantumConfig()
        config.noise.enabled = True
        config.noise.noise_type = NoiseType.DEPOLARIZING
        config.noise.single_qubit_error = 0.005
        config.noise.two_qubit_error = 0.02
        config.noise.t1_time = 30e-6
        config.noise.t2_time = 40e-6
        return config

    @staticmethod
    def fast_prototype() -> QuantumConfig:
        config = QuantumConfig()
        config.simulation.shots = 100
        config.optimization.max_iterations = 50
        config.simulation.max_qubits = 5
        return config

    @staticmethod
    def quantum_advantage() -> QuantumConfig:
        config = QuantumConfig()
        config.simulation.max_qubits = 20
        config.simulation.shots = 10000
        config.error_correction.enabled = True
        config.error_correction.code = ErrorCode.SURFACE
        config.error_correction.code_distance = 5
        return config

    @staticmethod
    def list_presets() -> List[str]:
        return ['high_precision', 'noisy_hardware', 'fast_prototype', 'quantum_advantage']
