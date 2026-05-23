"""FastAPI REST API - 22+ 端点"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import time

app = FastAPI(title="Quantum Engine API", version="0.1.0", description="AI 量子计算模拟与优化蜂群引擎")

# === 全局引擎 ===
from core.engine import QuantumEngine, AgentRole, AgentTask
from core.quantum_state import QuantumStateVector, QuantumState, DensityMatrix
from core.quantum_gates import GateFactory
from core.circuit import QuantumCircuit
from core.error_correction import ErrorCorrectionEngine
from core.optimization import OptimizationEngine
from core.noise import NoiseModel, NoiseModelType, NoiseParameters
from core.entanglement import EntanglementAnalyzer
from core.algorithms import QuantumAlgorithmLibrary
from core.pipeline import QuantumPipeline
from agents.pool import create_all_agents, register_all_agents

engine = QuantumEngine()
register_all_agents(engine.registry)
pipeline = QuantumPipeline()
algo_lib = QuantumAlgorithmLibrary()


# === 请求模型 ===
class CircuitRequest(BaseModel):
    num_qubits: int = 3
    gates: List[Dict[str, Any]] = []
    shots: int = 1024

class StateRequest(BaseModel):
    num_qubits: int = 3
    state_type: str = "zero"
    bell_type: int = 0

class OptimizationRequest(BaseModel):
    method: str = "cobyla"
    num_params: int = 4
    max_iter: int = 100

class NoiseRequest(BaseModel):
    model_type: str = "depolarizing"
    single_error: float = 0.001
    two_error: float = 0.01

class EntanglementRequest(BaseModel):
    state: List[float] = [0.707, 0, 0, 0.707]
    num_qubits: int = 2

class GroverRequest(BaseModel):
    n: int = 4
    target: int = 0
    shots: int = 1024

class QFTRequest(BaseModel):
    n: int = 4

class QKDRequest(BaseModel):
    key_length: int = 16
    eavesdrop: bool = False

class TaskRequest(BaseModel):
    name: str = ""
    task_type: str = ""
    role: str = "circuit_architect"
    parameters: Dict[str, Any] = {}

class PipelineRequest(BaseModel):
    num_qubits: int = 3
    circuit_type: str = "generic"
    shots: int = 1024

class EvolutionRequest(BaseModel):
    dimension: int = 10
    generations: int = 50
    population_size: int = 20


# === 端点 ===
@app.get("/")
async def root():
    return {"name": "Quantum Engine", "version": "0.1.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": len(engine.registry.get_active())}

@app.get("/metrics")
async def metrics():
    return engine.get_status()

@app.post("/circuit/simulate")
async def simulate_circuit(req: CircuitRequest):
    circuit = QuantumCircuit(req.num_qubits)
    for gate_info in req.gates:
        gate = GateFactory.create(gate_info.get('name', 'H'))
        qubits = gate_info.get('qubits', [0])
        circuit.add_gate(gate, qubits)
    if not req.gates:
        for i in range(req.num_qubits):
            circuit.h(i)
    result = circuit.simulate(shots=req.shots)
    return {"counts": result['counts'], "depth": result['depth'], "gate_count": result['gate_count']}

@app.post("/state/create")
async def create_state(req: StateRequest):
    if req.state_type == "bell":
        sv = QuantumStateVector.bell_state(req.bell_type)
    elif req.state_type == "ghz":
        sv = QuantumStateVector.ghz_state(req.num_qubits)
    elif req.state_type == "w":
        sv = QuantumStateVector.w_state(req.num_qubits)
    elif req.state_type == "random":
        sv = QuantumStateVector.random_state(req.num_qubits)
    else:
        sv = QuantumStateVector.zero_state(req.num_qubits)
    return {"state": sv.to_dict(), "entropy": sv.entropy()}

@app.get("/gates")
async def list_gates():
    return {"gates": GateFactory.list_gates()}

@app.post("/gates/create")
async def create_gate(name: str, theta: float = 0.0, phi: float = 0.0):
    try:
        gate = GateFactory.create(name, theta=theta, phi=phi)
        return {"name": gate.name, "is_unitary": gate.is_unitary(), "matrix": gate.matrix.tolist()}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/error-correction/analyze")
async def analyze_error_correction(code_type: str = "repetition", distance: int = 3, physical_rate: float = 0.01):
    engine_ec = ErrorCorrectionEngine(code_type, distance)
    return {
        "code_params": engine_ec.get_code_params(),
        "logical_error_rate": engine_ec.logical_error_rate(physical_rate),
        "improvement": physical_rate / max(engine_ec.logical_error_rate(physical_rate), 1e-15)
    }

@app.post("/optimization/minimize")
async def minimize(req: OptimizationRequest):
    opt_engine = OptimizationEngine(req.method, req.max_iter)
    target = np.ones(req.num_params) * 0.5
    def loss(x): return float(np.linalg.norm(x - target))
    x0 = np.random.randn(req.num_params)
    result = opt_engine.minimize(loss, x0)
    return result.to_dict()

@app.post("/noise/model")
async def noise_model(req: NoiseRequest):
    params = NoiseParameters(single_qubit_error=req.single_error, two_qubit_error=req.two_error)
    try:
        model = NoiseModel(NoiseModelType(req.model_type), params)
        return model.to_dict()
    except ValueError:
        raise HTTPException(400, f"Invalid model type: {req.model_type}")

@app.post("/entanglement/analyze")
async def analyze_entanglement(req: EntanglementRequest):
    state = np.array(req.state, dtype=complex)
    analyzer = EntanglementAnalyzer()
    measure = analyzer.analyze_state_vector(state, req.num_qubits)
    return {"measures": measure.to_dict(), "is_entangled": measure.is_entangled(), "class": measure.entanglement_class()}

@app.post("/algorithms/grover")
async def run_grover(req: GroverRequest):
    result = algo_lib.run("grover", n=req.n, target=req.target, shots=req.shots)
    return result.to_dict()

@app.post("/algorithms/qft")
async def run_qft(req: QFTRequest):
    result = algo_lib.run("qft", n=req.n)
    return result.to_dict()

@app.post("/algorithms/qkd")
async def run_qkd(req: QKDRequest):
    result = algo_lib.run("qkd", key_length=req.key_length, eavesdrop=req.eavesdrop)
    return result.to_dict()

@app.post("/algorithms/deutsch_jozsa")
async def run_deutsch_jozsa(n: int = 3, oracle_type: str = "balanced"):
    result = algo_lib.run("deutsch_jozsa", n=n, oracle_type=oracle_type)
    return result.to_dict()

@app.post("/algorithms/teleportation")
async def run_teleportation():
    result = algo_lib.run("teleportation")
    return result.to_dict()

@app.post("/algorithms/superdense")
async def run_superdense(message: str = "01"):
    result = algo_lib.run("superdense", message=message)
    return result.to_dict()

@app.post("/task/dispatch")
async def dispatch_task(req: TaskRequest):
    try:
        role = AgentRole(req.role)
    except ValueError:
        raise HTTPException(400, f"Invalid role: {req.role}")
    task = AgentTask(name=req.name, task_type=req.task_type, parameters=req.parameters)
    result = engine.dispatcher.dispatch(task, role)
    return result.to_dict()

@app.get("/agents")
async def list_agents():
    return {"agents": [a.get_metrics() for a in engine.registry.get_all()]}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = engine.registry.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent.get_metrics()

@app.post("/pipeline/execute")
async def execute_pipeline(req: PipelineRequest):
    result = pipeline.execute({'num_qubits': req.num_qubits, 'circuit_type': req.circuit_type, 'shots': req.shots})
    return {"metrics": pipeline.get_metrics(), "layers_passed": result['metrics'].layers_passed}

@app.post("/evolution/run")
async def run_evolution(req: EvolutionRequest):
    def fitness(x): return -float(np.sum(x**2))
    evo = engine.evolution
    evo.population_size = req.population_size
    evo.initialize(req.dimension)
    result = evo.evolve(fitness, req.generations)
    return {"best_fitness": result['best_fitness'], "generations": result['generations']}

@app.get("/qft/matrix")
async def qft_matrix(n: int = 4):
    from core.algorithms import QuantumFourierTransform
    qft = QuantumFourierTransform(n)
    return {"matrix": qft.qft_matrix().tolist(), "gate_count": qft.gate_count()}



# === 高级端点 ===

class TomographyRequest(BaseModel):
    num_qubits: int = 2
    shots: int = 1024

class CircuitCompileRequest(BaseModel):
    num_qubits: int = 3
    gates: List[Dict[str, Any]] = []
    target_gates: List[str] = ["CNOT", "Rx", "Ry", "Rz"]

class BenchmarkRequest(BaseModel):
    num_qubits: int = 3
    depth: int = 10
    test_type: str = "simulation"

class StateCompareRequest(BaseModel):
    state1: List[float] = [1, 0, 0, 0]
    state2: List[float] = [1, 0, 0, 0]

class ChannelRequest(BaseModel):
    channel_type: str = "depolarizing"
    parameter: float = 0.1
    num_qubits: int = 1

class WalkRequest(BaseModel):
    steps: int = 10
    start: int = 0

class SimonRequest(BaseModel):
    n: int = 4
    secret: int = 0b1100

class QPERequest(BaseModel):
    precision_bits: int = 4
    phase: float = 0.25


@app.post("/tomography/reconstruct")
async def reconstruct_state(req: TomographyRequest):
    from core.quantum_state import QuantumStateTomography
    tomo = QuantumStateTomography(req.num_qubits)
    bases = tomo.generate_measurement_bases()
    return {
        "num_bases": len(bases),
        "num_qubits": req.num_qubits,
        "measurement_bases": len(bases)
    }


@app.post("/circuit/compile")
async def compile_circuit(req: CircuitCompileRequest):
    from core.circuit import CircuitCompiler
    circuit = QuantumCircuit(req.num_qubits)
    for gate_info in req.gates:
        gate = GateFactory.create(gate_info.get('name', 'H'))
        circuit.add_gate(gate, gate_info.get('qubits', [0]))
    compiler = CircuitCompiler(req.target_gates)
    compiled = compiler.compile(circuit)
    return {
        "original_gates": circuit.gate_count,
        "compiled_gates": compiled.gate_count,
        "gate_cost": compiler.gate_cost(compiled)
    }


@app.post("/circuit/optimize")
async def optimize_circuit_endpoint(req: CircuitCompileRequest):
    from core.circuit import CircuitOptimizer
    circuit = QuantumCircuit(req.num_qubits)
    for gate_info in req.gates:
        gate = GateFactory.create(gate_info.get('name', 'H'))
        circuit.add_gate(gate, gate_info.get('qubits', [0]))
    optimizer = CircuitOptimizer()
    optimized = optimizer.optimize(circuit)
    report = optimizer.count_optimizations(circuit, optimized)
    return report


@app.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    from core.engine import QuantumBenchmark
    bench = QuantumBenchmark()
    if req.test_type == "simulation":
        result = bench.benchmark_simulation(req.num_qubits, req.depth)
    elif req.test_type == "entanglement":
        result = bench.benchmark_entanglement(req.num_qubits)
    elif req.test_type == "optimization":
        result = bench.benchmark_optimization(req.depth)
    else:
        result = bench.benchmark_state_preparation(req.num_qubits, 'random')
    return result


@app.post("/state/compare")
async def compare_states(req: StateCompareRequest):
    from core.quantum_state import QuantumStateVector
    sv1 = QuantumStateVector(num_qubits=int(np.log2(len(req.state1))), amplitudes=np.array(req.state1, dtype=complex))
    sv2 = QuantumStateVector(num_qubits=int(np.log2(len(req.state2))), amplitudes=np.array(req.state2, dtype=complex))
    return {
        "fidelity": sv1.fidelity(sv2),
        "trace_distance": sv1.trace_distance(sv2),
        "inner_product": complex(sv1.inner_product(sv2))
    }


@app.post("/channel/analyze")
async def analyze_channel(req: ChannelRequest):
    from core.noise import NoiseCharacterizer, NoiseModel, NoiseModelType, NoiseParameters
    params = NoiseParameters()
    params.single_qubit_error = req.parameter
    model = NoiseModel(NoiseModelType(req.channel_type), params)
    return {
        "channel_type": req.channel_type,
        "noise_level": model.get_noise_level()
    }


@app.post("/algorithms/walk")
async def run_walk(req: WalkRequest):
    result = algo_lib.run("quantum_walk", steps=req.steps)
    return result.to_dict()


@app.post("/algorithms/simon")
async def run_simon(req: SimonRequest):
    result = algo_lib.run("simon", n=req.n, secret=req.secret)
    return result.to_dict()


@app.post("/algorithms/qpe")
async def run_qpe(req: QPERequest):
    U = np.array([[1, 0], [0, np.exp(1j * 2 * np.pi * req.phase)]], dtype=complex)
    result = algo_lib.run("qpe", unitary=U, precision=req.precision_bits)
    return result.to_dict()


@app.get("/config/presets")
async def list_presets():
    from config.settings import ConfigPresets
    return {"presets": ConfigPresets.list_presets()}


@app.get("/system/info")
async def system_info():
    return {
        "name": "Quantum Engine",
        "version": "0.1.0",
        "agents": 10,
        "dag_layers": 7,
        "quantum_dimensions": 10,
        "gates": len(GateFactory.list_gates()),
        "algorithms": len(algo_lib.list_algorithms()),
        "noise_channels": 7,
        "error_correction_codes": 4,
        "optimizers": 7
    }
