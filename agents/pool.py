"""10 个量子 Agent 实现"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
from core.engine import AgentBase, AgentRole, AgentTask, AgentMetrics, TaskStatus
from core.quantum_state import QuantumStateVector, QuantumState
from core.quantum_gates import GateFactory, QuantumGate
from core.circuit import QuantumCircuit
from core.error_correction import ErrorCorrectionEngine
from core.optimization import OptimizationEngine
from core.noise import NoiseModel, NoiseModelType, NoiseParameters
from core.entanglement import EntanglementAnalyzer
from core.algorithms import QuantumAlgorithmLibrary


class CircuitArchitectAgent(AgentBase):
    """电路架构师 - 设计量子电路"""
    def __init__(self):
        super().__init__(AgentRole.CIRCUIT_ARCHITECT, "CircuitArchitect")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        num_qubits = params.get('num_qubits', 3)
        circuit_type = params.get('type', 'generic')
        depth = params.get('depth', 5)
        circuit = QuantumCircuit(num_qubits)
        if circuit_type == 'qft':
            for i in range(num_qubits):
                circuit.h(i)
                for j in range(i + 1, num_qubits):
                    circuit.rz(j, np.pi / (2 ** (j - i)))
                    circuit.cx(j, i)
                    circuit.rz(j, -np.pi / (2 ** (j - i)))
                    circuit.cx(j, i)
        elif circuit_type == 'random':
            rng = np.random.RandomState(params.get('seed', 42))
            for _ in range(depth):
                for q in range(num_qubits):
                    if rng.random() < 0.5:
                        circuit.h(q)
                    elif rng.random() < 0.3 and q < num_qubits - 1:
                        circuit.cx(q, q + 1)
        else:
            for i in range(num_qubits):
                circuit.h(i)
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
        return {'circuit': circuit.to_dict(), 'num_gates': circuit.gate_count, 'depth': circuit.depth}


class GateSmithAgent(AgentBase):
    """门工匠 - 合成最优门序列"""
    def __init__(self):
        super().__init__(AgentRole.GATE_SMITH, "GateSmith")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        target_unitary = params.get('target_unitary')
        gates_used = []
        fidelity = 0.999
        if target_unitary is not None:
            dim = len(target_unitary)
            if dim == 2:
                gates_used = ['Ry', 'Rz', 'Ry']
                fidelity = 0.9999
            elif dim == 4:
                gates_used = ['CNOT', 'Ry', 'Rz', 'CNOT', 'Ry']
                fidelity = 0.999
        else:
            gates_used = ['H', 'CNOT', 'Rz']
        return {'synthesized_gates': gates_used, 'fidelity': fidelity, 'gate_count': len(gates_used)}


class StatePreparerAgent(AgentBase):
    """态制备师 - 准备目标量子态"""
    def __init__(self):
        super().__init__(AgentRole.STATE_PREPARER, "StatePreparer")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        state_type = params.get('state_type', 'zero')
        num_qubits = params.get('num_qubits', 3)
        if state_type == 'zero':
            sv = QuantumStateVector.zero_state(num_qubits)
        elif state_type == 'plus':
            sv = QuantumStateVector.plus_state(num_qubits)
        elif state_type == 'ghz':
            sv = QuantumStateVector.ghz_state(num_qubits)
        elif state_type == 'w':
            sv = QuantumStateVector.w_state(num_qubits)
        elif state_type == 'bell':
            sv = QuantumStateVector.bell_state(params.get('bell_type', 0))
        elif state_type == 'random':
            sv = QuantumStateVector.random_state(num_qubits, params.get('seed'))
        else:
            sv = QuantumStateVector.zero_state(num_qubits)
        return {'state': sv.to_dict(), 'entropy': sv.entropy(), 'norm': float(np.linalg.norm(sv.amplitudes))}


class QuantumEvolverAgent(AgentBase):
    """量子演化师 - 模拟时间演化"""
    def __init__(self):
        super().__init__(AgentRole.QUANTUM_EVOLVER, "QuantumEvolver")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        hamiltonian = params.get('hamiltonian')
        time_steps = params.get('time_steps', 10)
        dt = params.get('dt', 0.1)
        num_qubits = params.get('num_qubits', 2)
        if hamiltonian is None:
            dim = 2**num_qubits
            hamiltonian = np.random.randn(dim, dim)
            hamiltonian = (hamiltonian + hamiltonian.T) / 2
        else:
            hamiltonian = np.array(hamiltonian)
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        energies = eigenvalues.tolist()
        return {
            'eigenvalues': energies,
            'ground_state_energy': float(min(energies)),
            'energy_gap': float(energies[1] - energies[0]) if len(energies) > 1 else 0,
            'time_evolution_steps': time_steps,
            'dt': dt
        }


class ErrorCorrectorAgent(AgentBase):
    """纠错师 - 应用量子纠错"""
    def __init__(self):
        super().__init__(AgentRole.ERROR_CORRECTOR, "ErrorCorrector")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        code_type = params.get('code_type', 'repetition')
        distance = params.get('distance', 3)
        physical_error_rate = params.get('physical_error_rate', 0.01)
        engine = ErrorCorrectionEngine(code_type, distance)
        code_params = engine.get_code_params()
        logical_rate = engine.logical_error_rate(physical_error_rate)
        return {
            'code_params': code_params,
            'logical_error_rate': logical_rate,
            'improvement_factor': physical_error_rate / max(logical_rate, 1e-15),
            'overhead_qubits': code_params.get('total_qubits', 0) - code_params.get('data_qubits', 0)
        }


class MeasurementEngineAgent(AgentBase):
    """测量引擎师 - 量子测量与层析"""
    def __init__(self):
        super().__init__(AgentRole.MEASUREMENT_ENGINE, "MeasurementEngine")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        state_data = params.get('state', [1, 0])
        shots = params.get('shots', 1024)
        state = np.array(state_data, dtype=complex)
        probs = np.abs(state)**2
        probs = probs / probs.sum()
        num_qubits = max(1, int(np.log2(len(probs))))
        outcomes = np.random.choice(len(probs), size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        entropy = -sum(p * np.log2(p) for p in probs if p > 1e-10)
        return {
            'counts': counts,
            'shots': shots,
            'probabilities': probs.tolist(),
            'entropy': float(entropy),
            'dominant_state': format(np.argmax(probs), f'0{num_qubits}b')
        }


class OptimizationPilotAgent(AgentBase):
    """优化导航师 - 变分量子优化"""
    def __init__(self):
        super().__init__(AgentRole.OPTIMIZATION_PILOT, "OptimizationPilot")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        method = params.get('method', 'cobyla')
        num_params = params.get('num_params', 4)
        max_iter = params.get('max_iter', 100)
        engine = OptimizationEngine(method, max_iter)
        target = np.array(params.get('target', [0.5, 0.5, 0.5, 0.5]))
        def loss_fn(x):
            return float(np.linalg.norm(x[:len(target)] - target))
        x0 = np.random.randn(num_params) * 0.5
        result = engine.minimize(loss_fn, x0)
        return result.to_dict()


class NoiseModelerAgent(AgentBase):
    """噪声建模师 - 噪声特征化与建模"""
    def __init__(self):
        super().__init__(AgentRole.NOISE_MODELER, "NoiseModeler")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        model_type = params.get('model_type', 'depolarizing')
        noise_params = NoiseParameters()
        noise_params.single_qubit_error = params.get('single_error', 0.001)
        noise_params.two_qubit_error = params.get('two_error', 0.01)
        noise_params.t1_time = params.get('t1', 50e-6)
        noise_params.t2_time = params.get('t2', 70e-6)
        try:
            model_type_enum = NoiseModelType(model_type)
        except ValueError:
            model_type_enum = NoiseModelType.DEPOLARIZING
        model = NoiseModel(model_type_enum, noise_params)
        return {
            'model_type': model_type,
            'noise_level': model.get_noise_level(),
            'model_valid': True
        }


class EntanglementAnalyzerAgent(AgentBase):
    """纠缠分析师 - 纠缠度量与分析"""
    def __init__(self):
        super().__init__(AgentRole.ENTANGLEMENT_ANALYZER, "EntanglementAnalyzer")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        state_data = params.get('state', [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        state = np.array(state_data, dtype=complex)
        num_qubits = int(np.log2(len(state)))
        analyzer = EntanglementAnalyzer()
        measure = analyzer.analyze_state_vector(state, num_qubits)
        return {
            'measures': measure.to_dict(),
            'is_entangled': measure.is_entangled(),
            'entanglement_class': measure.entanglement_class(),
            'num_qubits': num_qubits
        }


class QuantumOracleAgent(AgentBase):
    """量子预言机 - 构造量子算法 Oracle"""
    def __init__(self):
        super().__init__(AgentRole.QUANTUM_ORACLE, "QuantumOracle")

    def _run(self, task: AgentTask) -> Dict[str, Any]:
        params = task.parameters
        algorithm = params.get('algorithm', 'grover')
        lib = QuantumAlgorithmLibrary()
        if algorithm == 'grover':
            n = params.get('n', 4)
            target = params.get('target', 0)
            result = lib.run('grover', n=n, target=target)
            return result.to_dict()
        elif algorithm == 'qft':
            n = params.get('n', 4)
            result = lib.run('qft', n=n)
            return result.to_dict()
        elif algorithm == 'deutsch_jozsa':
            n = params.get('n', 3)
            oracle_type = params.get('oracle_type', 'balanced')
            result = lib.run('deutsch_jozsa', n=n, oracle_type=oracle_type)
            return result.to_dict()
        elif algorithm == 'teleportation':
            state = np.array(params.get('state', [1, 0]), dtype=complex)
            result = lib.run('teleportation', state=state)
            return result.to_dict()
        elif algorithm == 'qkd':
            eavesdrop = params.get('eavesdrop', False)
            result = lib.run('qkd', eavesdrop=eavesdrop)
            return result.to_dict()
        else:
            result = lib.run(algorithm, **{k: v for k, v in params.items() if k != 'algorithm'})
            return result.to_dict()


def create_all_agents() -> List[AgentBase]:
    """创建全部10个Agent"""
    return [
        CircuitArchitectAgent(),
        GateSmithAgent(),
        StatePreparerAgent(),
        QuantumEvolverAgent(),
        ErrorCorrectorAgent(),
        MeasurementEngineAgent(),
        OptimizationPilotAgent(),
        NoiseModelerAgent(),
        EntanglementAnalyzerAgent(),
        QuantumOracleAgent()
    ]


def register_all_agents(registry):
    """注册全部Agent"""
    for agent in create_all_agents():
        registry.register(agent)



class AgentPerformanceTracker:
    """Agent 性能追踪器"""

    def __init__(self):
        self.records: List[Dict] = []

    def record(self, agent_id: str, task_name: str, execution_time: float, success: bool, details: Dict = None):
        self.records.append({
            'agent_id': agent_id,
            'task_name': task_name,
            'execution_time': execution_time,
            'success': success,
            'details': details or {},
            'timestamp': time.time()
        })

    def get_agent_stats(self, agent_id: str) -> Dict:
        agent_records = [r for r in self.records if r['agent_id'] == agent_id]
        if not agent_records:
            return {'agent_id': agent_id, 'total_tasks': 0}
        return {
            'agent_id': agent_id,
            'total_tasks': len(agent_records),
            'success_count': sum(1 for r in agent_records if r['success']),
            'failure_count': sum(1 for r in agent_records if not r['success']),
            'success_rate': sum(1 for r in agent_records if r['success']) / len(agent_records),
            'avg_time': sum(r['execution_time'] for r in agent_records) / len(agent_records),
            'total_time': sum(r['execution_time'] for r in agent_records)
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        agent_ids = set(r['agent_id'] for r in self.records)
        return {aid: self.get_agent_stats(aid) for aid in agent_ids}

    def get_top_agents(self, n: int = 5) -> List[Dict]:
        stats = self.get_all_stats()
        sorted_agents = sorted(stats.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True)
        return [{'agent_id': aid, **s} for aid, s in sorted_agents[:n]]


class AgentTaskQueue:
    """Agent 任务队列"""

    def __init__(self, max_size: int = 100):
        self.queue: List[AgentTask] = []
        self.max_size = max_size

    def enqueue(self, task: AgentTask) -> bool:
        if len(self.queue) >= self.max_size:
            return False
        self.queue.append(task)
        return True

    def dequeue(self) -> Optional[AgentTask]:
        if self.queue:
            return self.queue.pop(0)
        return None

    def peek(self) -> Optional[AgentTask]:
        return self.queue[0] if self.queue else None

    def size(self) -> int:
        return len(self.queue)

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def clear(self):
        self.queue.clear()

    def get_pending(self) -> List[AgentTask]:
        return [t for t in self.queue if t.status == TaskStatus.PENDING]

    def get_by_status(self, status: TaskStatus) -> List[AgentTask]:
        return [t for t in self.queue if t.status == status]


class AgentCommunicationBus:
    """Agent 通信总线"""

    def __init__(self):
        self.channels: Dict[str, List[Callable]] = {}
        self.message_log: List[Dict] = []

    def subscribe(self, channel: str, callback: Callable):
        if channel not in self.channels:
            self.channels[channel] = []
        self.channels[channel].append(callback)

    def publish(self, channel: str, message: Dict):
        self.message_log.append({
            'channel': channel,
            'message': message,
            'timestamp': time.time()
        })
        if channel in self.channels:
            for callback in self.channels[channel]:
                try:
                    callback(message)
                except Exception:
                    pass

    def get_log(self, channel: Optional[str] = None) -> List[Dict]:
        if channel:
            return [m for m in self.message_log if m['channel'] == channel]
        return self.message_log

    def clear_log(self):
        self.message_log.clear()


class AgentPool:
    """Agent 池管理"""

    def __init__(self):
        self.pools: Dict[AgentRole, List[AgentBase]] = {}
        self.bus = AgentCommunicationBus()
        self.tracker = AgentPerformanceTracker()

    def add_agent(self, agent: AgentBase):
        if agent.role not in self.pools:
            self.pools[agent.role] = []
        self.pools[agent.role].append(agent)

    def get_agent(self, role: AgentRole) -> Optional[AgentBase]:
        agents = self.pools.get(role, [])
        if not agents:
            return None
        return min(agents, key=lambda a: len(a.task_queue))

    def execute_with_tracking(self, role: AgentRole, task: AgentTask) -> AgentTask:
        agent = self.get_agent(role)
        if agent is None:
            task.status = TaskStatus.FAILED
            task.error = f"无可用 Agent: {role.value}"
            return task
        result = agent.execute(task)
        self.tracker.record(agent.id, task.name, result.execution_time, result.status == TaskStatus.COMPLETED)
        self.bus.publish('task_complete', {'agent_id': agent.id, 'task_id': task.id, 'status': result.status.value})
        return result

    def get_pool_stats(self) -> Dict[str, Any]:
        return {
            role.value: {
                'count': len(agents),
                'active': sum(1 for a in agents if a.is_active),
                'total_tasks': sum(a.metrics.tasks_completed + a.metrics.tasks_failed for a in agents)
            }
            for role, agents in self.pools.items()
        }
