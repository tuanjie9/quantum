"""量子引擎核心 - Agent基础/进化/模式/分类/工具"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any, Type
from enum import Enum
from .quantum_state import QuantumStateVector, DensityMatrix
from .circuit import QuantumCircuit
from .entanglement import EntanglementAnalyzer
from .optimization import OptimizationEngine
import time
import uuid
import math


class AgentRole(Enum):
    """Agent 角色"""
    CIRCUIT_ARCHITECT = "circuit_architect"
    GATE_SMITH = "gate_smith"
    STATE_PREPARER = "state_preparer"
    QUANTUM_EVOLVER = "quantum_evolver"
    ERROR_CORRECTOR = "error_corrector"
    MEASUREMENT_ENGINE = "measurement_engine"
    OPTIMIZATION_PILOT = "optimization_pilot"
    NOISE_MODELER = "noise_modeler"
    ENTANGLEMENT_ANALYZER = "entanglement_analyzer"
    QUANTUM_ORACLE = "quantum_oracle"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Agent 任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    task_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.task_type,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'error': self.error
        }


@dataclass
class AgentMetrics:
    """Agent 指标"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    evolution_generation: int = 0
    fitness_score: float = 0.0

    def update(self, execution_time: float, success: bool):
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.total_execution_time += execution_time
        total = self.tasks_completed + self.tasks_failed
        self.average_execution_time = self.total_execution_time / max(total, 1)
        self.success_rate = self.tasks_completed / max(total, 1)


class AgentBase:
    """Agent 基类"""

    def __init__(self, role: AgentRole, name: str):
        self.role = role
        self.name = name
        self.id = str(uuid.uuid4())[:8]
        self.metrics = AgentMetrics()
        self.task_queue: List[AgentTask] = []
        self.is_active = True
        self.config: Dict[str, Any] = {}

    def execute(self, task: AgentTask) -> AgentTask:
        """执行任务"""
        task.status = TaskStatus.RUNNING
        start = time.time()
        try:
            result = self._run(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        task.execution_time = time.time() - start
        task.completed_at = time.time()
        self.metrics.update(task.execution_time, task.status == TaskStatus.COMPLETED)
        return task

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        """子类实现具体逻辑"""
        raise NotImplementedError

    def validate(self, task: AgentTask) -> bool:
        """验证任务参数"""
        return True

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role.value,
            'is_active': self.is_active,
            'tasks_completed': self.metrics.tasks_completed,
            'tasks_failed': self.metrics.tasks_failed,
            'success_rate': self.metrics.success_rate,
            'avg_execution_time': self.metrics.average_execution_time,
            'fitness': self.metrics.fitness_score,
            'generation': self.metrics.evolution_generation
        }

    def __repr__(self) -> str:
        return f"Agent({self.name}, role={self.role.value}, tasks={self.metrics.tasks_completed})"


class AgentRegistry:
    """Agent 注册表"""

    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.role_map: Dict[AgentRole, List[str]] = {r: [] for r in AgentRole}

    def register(self, agent: AgentBase):
        self.agents[agent.id] = agent
        self.role_map[agent.role].append(agent.id)

    def get(self, agent_id: str) -> Optional[AgentBase]:
        return self.agents.get(agent_id)

    def get_by_role(self, role: AgentRole) -> List[AgentBase]:
        return [self.agents[aid] for aid in self.role_map.get(role, []) if aid in self.agents]

    def get_all(self) -> List[AgentBase]:
        return list(self.agents.values())

    def get_active(self) -> List[AgentBase]:
        return [a for a in self.agents.values() if a.is_active]

    def get_metrics_summary(self) -> Dict[str, Any]:
        total_tasks = sum(a.metrics.tasks_completed + a.metrics.tasks_failed for a in self.agents.values())
        total_success = sum(a.metrics.tasks_completed for a in self.agents.values())
        return {
            'total_agents': len(self.agents),
            'active_agents': len(self.get_active()),
            'total_tasks': total_tasks,
            'total_success': total_success,
            'overall_success_rate': total_success / max(total_tasks, 1)
        }


class TaskDispatcher:
    """任务调度器"""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.task_history: List[AgentTask] = []

    def dispatch(self, task: AgentTask, role: AgentRole) -> AgentTask:
        """分派任务"""
        agents = self.registry.get_by_role(role)
        if not agents:
            task.status = TaskStatus.FAILED
            task.error = f"无可用 Agent: {role.value}"
            return task
        # 选择最空闲的 Agent
        best_agent = min(agents, key=lambda a: len(a.task_queue))
        result = best_agent.execute(task)
        self.task_history.append(result)
        return result

    def dispatch_parallel(self, tasks: List[AgentTask], role: AgentRole) -> List[AgentTask]:
        """并行分派"""
        results = []
        agents = self.registry.get_by_role(role)
        if not agents:
            for task in tasks:
                task.status = TaskStatus.FAILED
                task.error = f"无可用 Agent: {role.value}"
                results.append(task)
            return results
        for i, task in enumerate(tasks):
            agent = agents[i % len(agents)]
            results.append(agent.execute(task))
        return results

    def get_history(self, limit: int = 100) -> List[Dict]:
        return [t.to_dict() for t in self.task_history[-limit:]]


class EvolutionEngine:
    """进化引擎 - 遗传算法"""

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.population: List[np.ndarray] = []
        self.fitness_history: List[float] = []

    def initialize(self, dimension: int, seed: Optional[int] = None):
        """初始化种群"""
        rng = np.random.RandomState(seed)
        self.population = [rng.randn(dimension) for _ in range(self.population_size)]

    def evolve(self, fitness_fn: Callable[[np.ndarray], float], generations: int = 100) -> Dict[str, Any]:
        """进化"""
        history = []
        for gen in range(generations):
            fitness_scores = [fitness_fn(ind) for ind in self.population]
            best_fitness = max(fitness_scores)
            history.append(best_fitness)
            # 选择
            parents = self._selection(fitness_scores)
            # 交叉
            offspring = self._crossover(parents)
            # 变异
            offspring = self._mutate(offspring)
            self.population = offspring
            self.generation += 1
        self.fitness_history = history
        best_idx = np.argmax([fitness_fn(ind) for ind in self.population])
        return {
            'best_individual': self.population[best_idx].tolist(),
            'best_fitness': fitness_fn(self.population[best_idx]),
            'generations': self.generation,
            'fitness_history': history
        }

    def _selection(self, fitness_scores: List[float]) -> List[np.ndarray]:
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size):
            i, j = np.random.randint(0, len(self.population), 2)
            winner = i if fitness_scores[i] > fitness_scores[j] else j
            selected.append(self.population[winner].copy())
        return selected

    def _crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """均匀交叉"""
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if np.random.random() < self.crossover_rate:
                mask = np.random.randint(0, 2, len(parents[i])).astype(float)
                child1 = parents[i] * mask + parents[i + 1] * (1 - mask)
                child2 = parents[i + 1] * mask + parents[i] * (1 - mask)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i + 1]])
        if len(offspring) < self.population_size:
            offspring.append(parents[-1].copy())
        return offspring[:self.population_size]

    def _mutate(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """高斯变异"""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                population[i] += np.random.randn(len(population[i])) * 0.1
        return population


class CircuitComplexityClassifier:
    """电路复杂度分类器"""

    COMPLEXITY_LEVELS = ['trivial', 'simple', 'moderate', 'complex', 'very_complex']

    def classify(self, num_qubits: int, gate_count: int, depth: int, two_qubit_gates: int) -> str:
        score = num_qubits * 1.0 + gate_count * 0.5 + depth * 0.3 + two_qubit_gates * 2.0
        if score < 5:
            return 'trivial'
        elif score < 15:
            return 'simple'
        elif score < 30:
            return 'moderate'
        elif score < 60:
            return 'complex'
        return 'very_complex'

    def recommend_optimization(self, complexity: str) -> List[str]:
        recommendations = {
            'trivial': ['无需优化'],
            'simple': ['门消除', '门合并'],
            'moderate': ['门消除', '门合并', '电路重写', '量子比特映射'],
            'complex': ['变分优化', '门分解', '近似合成'],
            'very_complex': ['分层优化', '自适应纠错', '噪声感知编译']
        }
        return recommendations.get(complexity, [])


class QuantumToolKit:
    """量子工具集"""

    @staticmethod
    def random_circuit(num_qubits: int, depth: int, seed: Optional[int] = None) -> List[Dict]:
        """生成随机电路"""
        rng = np.random.RandomState(seed)
        gates = []
        for _ in range(depth):
            for q in range(num_qubits):
                if rng.random() < 0.3 and q < num_qubits - 1:
                    gates.append({'gate': 'CNOT', 'qubits': [q, q + 1]})
                else:
                    gate = rng.choice(['H', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'])
                    if gate in ['Rx', 'Ry', 'Rz']:
                        gates.append({'gate': gate, 'qubits': [q], 'params': [rng.uniform(0, 2 * np.pi)]})
                    else:
                        gates.append({'gate': gate, 'qubits': [q]})
        return gates

    @staticmethod
    def calculate_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """计算保真度"""
        return float(np.abs(np.vdot(state1, state2))**2)

    @staticmethod
    def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
        """计算迹距离"""
        diff = rho1 - rho2
        eigenvalues = np.linalg.eigvalsh(diff)
        return float(np.sum(np.abs(eigenvalues)) / 2)

    @staticmethod
    def entropy(rho: np.ndarray) -> float:
        """冯诺依曼熵"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """汉明距离"""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @staticmethod
    def bitstring_to_int(bitstring: str) -> int:
        """比特串转整数"""
        return int(bitstring, 2)

    @staticmethod
    def int_to_bitstring(n: int, num_bits: int) -> str:
        """整数转比特串"""
        return format(n, f'0{num_bits}b')


class QuantumEngine:
    """量子引擎 - 统一接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = AgentRegistry()
        self.dispatcher = TaskDispatcher(self.registry)
        self.evolution = EvolutionEngine()
        self.classifier = CircuitComplexityClassifier()
        self.toolkit = QuantumToolKit()

    def register_agent(self, agent: AgentBase):
        self.registry.register(agent)

    def execute_task(self, task_name: str, task_type: str, role: AgentRole, **params) -> AgentTask:
        task = AgentTask(name=task_name, task_type=task_type, parameters=params)
        return self.dispatcher.dispatch(task, role)

    def get_status(self) -> Dict[str, Any]:
        return {
            'registry': self.registry.get_metrics_summary(),
            'dispatch_history': len(self.dispatcher.task_history),
            'evolution_generation': self.evolution.generation
        }

    def evolve_population(self, fitness_fn: Callable, dimension: int = 10, generations: int = 50) -> Dict:
        self.evolution.initialize(dimension)
        return self.evolution.evolve(fitness_fn, generations)



class QuantumStateManager:
    """量子态管理器"""

    def __init__(self):
        self.states: Dict[str, QuantumStateVector] = {}
        self.state_counter = 0

    def register(self, state: QuantumStateVector, name: Optional[str] = None) -> str:
        if name is None:
            name = f"state_{self.state_counter}"
            self.state_counter += 1
        self.states[name] = state
        return name

    def get(self, name: str) -> Optional[QuantumStateVector]:
        return self.states.get(name)

    def remove(self, name: str) -> bool:
        return self.states.pop(name, None) is not None

    def list_states(self) -> List[str]:
        return list(self.states.keys())

    def get_all_metrics(self) -> Dict[str, Dict]:
        metrics = {}
        for name, state in self.states.items():
            metrics[name] = {
                'num_qubits': state.num_qubits,
                'entropy': state.entropy(),
                'norm': float(np.linalg.norm(state.amplitudes))
            }
        return metrics

    def compare_states(self, name1: str, name2: str) -> Dict[str, float]:
        s1 = self.get(name1)
        s2 = self.get(name2)
        if s1 is None or s2 is None:
            return {'error': 'State not found'}
        return {
            'fidelity': s1.fidelity(s2),
            'trace_distance': s1.trace_distance(s2),
            'overlap': float(np.abs(s1.inner_product(s2))**2)
        }


class CircuitLibrary:
    """电路库"""

    def __init__(self):
        self.templates: Dict[str, Callable] = {}

    def register(self, name: str, builder: Callable):
        self.templates[name] = builder

    def build(self, name: str, **kwargs) -> QuantumCircuit:
        if name not in self.templates:
            raise ValueError(f"未知电路模板: {name}")
        return self.templates[name](**kwargs)

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())

    @classmethod
    def default_library(cls) -> 'CircuitLibrary':
        lib = cls()
        lib.register('bell_pair', lambda: cls._build_bell_pair())
        lib.register('ghz', lambda n=3: cls._build_ghz(n))
        lib.register('qft', lambda n=3: cls._build_qft(n))
        lib.register('random', lambda n=3, d=5: cls._build_random(n, d))
        return lib

    @staticmethod
    def _build_bell_pair() -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    @staticmethod
    def _build_ghz(n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        return qc

    @staticmethod
    def _build_qft(n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                qc.rz(j, np.pi / (2 ** (j - i)))
                qc.cx(j, i)
                qc.rz(j, -np.pi / (2 ** (j - i)))
                qc.cx(j, i)
        return qc

    @staticmethod
    def _build_random(n: int, depth: int) -> QuantumCircuit:
        rng = np.random.RandomState(42)
        qc = QuantumCircuit(n)
        for _ in range(depth):
            for q in range(n):
                r = rng.random()
                if r < 0.25:
                    qc.h(q)
                elif r < 0.5:
                    qc.x(q)
                elif r < 0.75:
                    qc.rz(q, rng.uniform(0, 2*np.pi))
                elif q < n - 1:
                    qc.cx(q, q + 1)
        return qc


class QuantumBenchmark:
    """量子基准测试"""

    def __init__(self):
        self.results: List[Dict] = []

    def benchmark_state_preparation(self, num_qubits: int, state_type: str = 'random') -> Dict:
        start = time.time()
        if state_type == 'random':
            sv = QuantumStateVector.random_state(num_qubits)
        elif state_type == 'ghz':
            sv = QuantumStateVector.ghz_state(num_qubits)
        elif state_type == 'bell':
            sv = QuantumStateVector.bell_state(0)
        else:
            sv = QuantumStateVector.zero_state(num_qubits)
        elapsed = time.time() - start
        result = {
            'test': 'state_preparation',
            'num_qubits': num_qubits,
            'state_type': state_type,
            'time': elapsed,
            'norm': float(np.linalg.norm(sv.amplitudes))
        }
        self.results.append(result)
        return result

    def benchmark_simulation(self, num_qubits: int, depth: int = 10) -> Dict:
        start = time.time()
        qc = QuantumCircuit(num_qubits)
        for _ in range(depth):
            for q in range(num_qubits):
                qc.h(q)
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        result = qc.simulate(shots=100)
        elapsed = time.time() - start
        benchmark = {
            'test': 'simulation',
            'num_qubits': num_qubits,
            'depth': depth,
            'time': elapsed,
            'num_outcomes': len(result['counts'])
        }
        self.results.append(benchmark)
        return benchmark

    def benchmark_entanglement(self, num_qubits: int) -> Dict:
        sv = QuantumStateVector.ghz_state(num_qubits)
        analyzer = EntanglementAnalyzer()
        start = time.time()
        measure = analyzer.analyze_state_vector(sv.amplitudes, num_qubits)
        elapsed = time.time() - start
        result = {
            'test': 'entanglement',
            'num_qubits': num_qubits,
            'time': elapsed,
            'entropy': measure.entanglement_entropy,
            'is_entangled': measure.is_entangled()
        }
        self.results.append(result)
        return result

    def benchmark_optimization(self, num_params: int = 10) -> Dict:
        engine = OptimizationEngine("cobyla", max_iter=100)
        target = np.ones(num_params) * 0.5
        def loss(x): return float(np.linalg.norm(x - target))
        start = time.time()
        result = engine.minimize(loss, np.random.randn(num_params))
        elapsed = time.time() - start
        benchmark = {
            'test': 'optimization',
            'num_params': num_params,
            'time': elapsed,
            'optimal_value': result.optimal_value,
            'iterations': result.iterations
        }
        self.results.append(benchmark)
        return benchmark

    def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整基准测试"""
        summary = {'tests': []}
        for n in [2, 3, 4, 5]:
            summary['tests'].append(self.benchmark_state_preparation(n, 'random'))
            summary['tests'].append(self.benchmark_simulation(n, 5))
            summary['tests'].append(self.benchmark_entanglement(n))
        summary['tests'].append(self.benchmark_optimization(5))
        summary['total_tests'] = len(summary['tests'])
        summary['total_time'] = sum(t['time'] for t in summary['tests'])
        return summary

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_benchmarks': len(self.results),
            'total_time': sum(r['time'] for r in self.results),
            'average_time': sum(r['time'] for r in self.results) / max(len(self.results), 1)
        }


class QuantumDebugger:
    """量子调试器"""

    def __init__(self):
        self.log: List[Dict] = []

    def inspect_state(self, state: QuantumStateVector) -> Dict:
        probs = state.probabilities()
        top_states = np.argsort(probs)[-5:][::-1]
        result = {
            'num_qubits': state.num_qubits,
            'entropy': state.entropy(),
            'norm': float(np.linalg.norm(state.amplitudes)),
            'top_states': []
        }
        for idx in top_states:
            if probs[idx] > 1e-6:
                result['top_states'].append({
                    'index': int(idx),
                    'bitstring': format(idx, f'0{state.num_qubits}b'),
                    'probability': float(probs[idx]),
                    'amplitude': complex(state.amplitudes[idx])
                })
        self.log.append({'action': 'inspect_state', 'result': result})
        return result

    def inspect_circuit(self, circuit: QuantumCircuit) -> Dict:
        result = {
            'num_qubits': circuit.num_qubits,
            'gate_count': circuit.gate_count,
            'depth': circuit.depth,
            'two_qubit_gates': circuit.two_qubit_gate_count,
            'gate_types': circuit.gate_types
        }
        self.log.append({'action': 'inspect_circuit', 'result': result})
        return result

    def inspect_density_matrix(self, dm: DensityMatrix) -> Dict:
        result = {
            'num_qubits': dm.num_qubits,
            'purity': dm.purity(),
            'entropy': dm.von_neumann_entropy(),
            'is_valid': dm.is_valid(),
            'trace': float(np.trace(dm.matrix))
        }
        self.log.append({'action': 'inspect_density_matrix', 'result': result})
        return result

    def get_log(self) -> List[Dict]:
        return self.log

    def clear_log(self):
        self.log.clear()


class QuantumExporter:
    """量子态/电路导出器"""

    @staticmethod
    def state_to_json(state: QuantumStateVector) -> str:
        import json
        data = {
            'num_qubits': state.num_qubits,
            'amplitudes_real': [float(a.real) for a in state.amplitudes],
            'amplitudes_imag': [float(a.imag) for a in state.amplitudes],
            'probabilities': state.probabilities().tolist()
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def circuit_to_json(circuit: QuantumCircuit) -> str:
        import json
        return json.dumps(circuit.to_dict(), indent=2)

    @staticmethod
    def counts_to_csv(counts: Dict[str, int]) -> str:
        lines = ['bitstring,count,probability']
        total = sum(counts.values())
        for bs, count in sorted(counts.items()):
            lines.append(f'{bs},{count},{count/total:.6f}')
        return '\n'.join(lines)

    @staticmethod
    def export_qasm(circuit: QuantumCircuit) -> str:
        return circuit.to_qasm()
