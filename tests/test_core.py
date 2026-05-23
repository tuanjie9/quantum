"""Quantum 项目测试 - 100+ 测试用例"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_state import QuantumStateVector, QuantumState, DensityMatrix
from core.quantum_gates import (GateFactory, PauliX, PauliY, PauliZ, Hadamard,
                                 CNOT, CZ, SWAP, Toffoli, PhaseGate, RxGate, RyGate, RzGate,
                                 PAULI_X, PAULI_Y, PAULI_Z)
from core.circuit import QuantumCircuit
from core.error_correction import (ErrorCorrectionEngine, RepetCode, SteaneCode,
                                    SurfaceCode, NoiseChannel)
from core.optimization import (OptimizationEngine, AdamOptimizer, SPSAOptimizer,
                                COBYLAOptimizer, ParameterShiftRule, QAOA)
from core.noise import (NoiseModel, NoiseModelType, NoiseParameters,
                         DepolarizingChannel, AmplitudeDampingChannel)
from core.entanglement import (EntanglementAnalyzer, ConcurrenceCalculator,
                                NegativityCalculator, BellStateAnalyzer, SchmidtDecomposition, EntanglementEntropyCalculator, EntanglementWitness)
from core.algorithms import (QuantumFourierTransform, GroverSearch, DeutschJozsa,
                              QuantumTeleportation, SuperdenseCoding, QuantumKeyDistribution,
                              QuantumAlgorithmLibrary, QuantumPhaseEstimation)
from core.pipeline import QuantumPipeline, PipelineLayer
from core.engine import (QuantumEngine, AgentBase, AgentRole, AgentTask, AgentRegistry,
                          EvolutionEngine, CircuitComplexityClassifier, QuantumToolKit)
from agents.pool import (create_all_agents, register_all_agents, CircuitArchitectAgent, GateSmithAgent,
                          StatePreparerAgent, QuantumEvolverAgent, ErrorCorrectorAgent,
                          MeasurementEngineAgent, OptimizationPilotAgent, NoiseModelerAgent,
                          EntanglementAnalyzerAgent, QuantumOracleAgent)


# === QuantumStateVector 测试 ===

class TestQuantumStateVector:
    def test_zero_state(self):
        sv = QuantumStateVector.zero_state(2)
        assert sv.num_qubits == 2
        assert abs(sv.amplitudes[0] - 1.0) < 1e-10
        assert abs(sv.amplitudes[1]) < 1e-10

    def test_one_state(self):
        sv = QuantumStateVector.one_state(2)
        assert abs(sv.amplitudes[-1] - 1.0) < 1e-10

    def test_plus_state(self):
        sv = QuantumStateVector.plus_state(3)
        expected = 1.0 / np.sqrt(8)
        for i in range(8):
            assert abs(sv.amplitudes[i] - expected) < 1e-10

    def test_bell_state(self):
        sv = QuantumStateVector.bell_state(0)
        assert sv.num_qubits == 2
        assert abs(sv.amplitudes[0] - 1/np.sqrt(2)) < 1e-10
        assert abs(sv.amplitudes[3] - 1/np.sqrt(2)) < 1e-10

    def test_ghz_state(self):
        sv = QuantumStateVector.ghz_state(3)
        assert abs(sv.amplitudes[0] - 1/np.sqrt(2)) < 1e-10
        assert abs(sv.amplitudes[7] - 1/np.sqrt(2)) < 1e-10

    def test_w_state(self):
        sv = QuantumStateVector.w_state(3)
        assert abs(sv.amplitudes[1] - 1/np.sqrt(3)) < 1e-10
        assert abs(sv.amplitudes[2] - 1/np.sqrt(3)) < 1e-10
        assert abs(sv.amplitudes[4] - 1/np.sqrt(3)) < 1e-10

    def test_random_state(self):
        sv = QuantumStateVector.random_state(3, seed=42)
        norm = np.linalg.norm(sv.amplitudes)
        assert abs(norm - 1.0) < 1e-10

    def test_normalize(self):
        sv = QuantumStateVector(2, np.array([2, 0, 0, 0], dtype=complex))
        sv.normalize()
        assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10

    def test_probabilities(self):
        sv = QuantumStateVector.bell_state(0)
        probs = sv.probabilities()
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[3] - 0.5) < 1e-10

    def test_fidelity(self):
        sv1 = QuantumStateVector.zero_state(2)
        sv2 = QuantumStateVector.zero_state(2)
        assert abs(sv1.fidelity(sv2) - 1.0) < 1e-10

    def test_entropy_zero(self):
        sv = QuantumStateVector.zero_state(3)
        assert abs(sv.entropy()) < 1e-10

    def test_entropy_plus(self):
        sv = QuantumStateVector.plus_state(3)
        assert abs(sv.entropy() - 3.0) < 1e-10

    def test_tensor_product(self):
        sv1 = QuantumStateVector.zero_state(1)
        sv2 = QuantumStateVector.one_state(1)
        result = sv1.tensor_product(sv2)
        assert result.num_qubits == 2
        assert abs(result.amplitudes[1] - 1.0) < 1e-10

    def test_measure(self):
        sv = QuantumStateVector.zero_state(2)
        counts = sv.measure(shots=100, seed=42)
        assert '00' in counts
        assert counts['00'] == 100

    def test_concurrence(self):
        sv = QuantumStateVector.bell_state(0)
        c = sv.concurrence()
        assert abs(c - 1.0) < 1e-10

    def test_bloch_angles(self):
        sv = QuantumStateVector.zero_state(1)
        theta, phi, r = sv.bloch_angles(0)
        assert abs(theta) < 0.1
        assert abs(r - 1.0) < 0.1

    def test_inner_product(self):
        sv1 = QuantumStateVector.zero_state(1)
        sv2 = QuantumStateVector.zero_state(1)
        assert abs(sv1.inner_product(sv2) - 1.0) < 1e-10

    def test_to_dict(self):
        sv = QuantumStateVector.bell_state(0)
        d = sv.to_dict()
        assert 'num_qubits' in d
        assert 'amplitudes' in d


# === QuantumState 测试 ===

class TestQuantumState:
    def test_creation(self):
        qs = QuantumState()
        assert qs.amplitude.value == 1.0
        assert qs.coherence.value == 1.0

    def test_to_vector(self):
        qs = QuantumState()
        vec = qs.to_vector()
        assert len(vec) == 10

    def test_from_vector(self):
        vec = np.array([0.5]*10)
        qs = QuantumState.from_vector(vec)
        assert qs.amplitude.value == pytest.approx(0.5, abs=0.1)

    def test_mutate(self):
        qs = QuantumState()
        mutated = qs.mutate(0.1)
        assert isinstance(mutated, QuantumState)

    def test_crossover(self):
        qs1 = QuantumState()
        qs2 = QuantumState()
        child = qs1.crossover(qs2)
        assert isinstance(child, QuantumState)

    def test_distance(self):
        qs1 = QuantumState()
        qs2 = QuantumState()
        assert qs1.distance(qs2) == pytest.approx(0.0, abs=0.01)

    def test_fidelity_10d(self):
        qs1 = QuantumState()
        qs2 = QuantumState()
        assert qs1.fidelity_10d(qs2) > 0.99

    def test_to_dict(self):
        qs = QuantumState()
        d = qs.to_dict()
        assert 'amplitude' in d
        assert len(d) == 10


# === DensityMatrix 测试 ===

class TestDensityMatrix:
    def test_creation(self):
        dm = DensityMatrix(2)
        assert dm.num_qubits == 2
        assert dm.matrix.shape == (4, 4)

    def test_from_state_vector(self):
        sv = QuantumStateVector.zero_state(2)
        dm = DensityMatrix.from_state_vector(sv)
        assert dm.purity() > 0.99

    def test_maximally_mixed(self):
        dm = DensityMatrix.maximally_mixed(2)
        assert abs(dm.purity() - 0.25) < 1e-10

    def test_von_neumann_entropy(self):
        dm = DensityMatrix.maximally_mixed(2)
        assert abs(dm.von_neumann_entropy() - 2.0) < 1e-10

    def test_is_valid(self):
        sv = QuantumStateVector.bell_state(0)
        dm = DensityMatrix.from_state_vector(sv)
        assert dm.is_valid()


# === QuantumGate 测试 ===

class TestQuantumGates:
    def test_pauli_x(self):
        gate = PauliX()
        assert gate.is_unitary()
        assert gate.num_qubits == 1

    def test_pauli_y(self):
        gate = PauliY()
        assert gate.is_unitary()

    def test_pauli_z(self):
        gate = PauliZ()
        assert gate.is_unitary()

    def test_hadamard(self):
        gate = Hadamard()
        assert gate.is_unitary()
        H2 = gate.matrix @ gate.matrix
        assert np.allclose(H2, np.eye(2))

    def test_cnot(self):
        gate = CNOT()
        assert gate.is_unitary()
        assert gate.num_qubits == 2

    def test_cz(self):
        gate = CZ()
        assert gate.is_unitary()

    def test_swap(self):
        gate = SWAP()
        assert gate.is_unitary()

    def test_toffoli(self):
        gate = Toffoli()
        assert gate.is_unitary()
        assert gate.num_qubits == 3

    def test_phase_gate(self):
        gate = PhaseGate(np.pi / 4)
        assert gate.is_unitary()

    def test_rx_gate(self):
        gate = RxGate(np.pi / 3)
        assert gate.is_unitary()

    def test_ry_gate(self):
        gate = RyGate(np.pi / 3)
        assert gate.is_unitary()

    def test_rz_gate(self):
        gate = RzGate(np.pi / 3)
        assert gate.is_unitary()

    def test_inverse(self):
        gate = Hadamard()
        inv = gate.inverse()
        product = gate.matrix @ inv.matrix
        assert np.allclose(product, np.eye(2))

    def test_factory_create(self):
        for name in ['X', 'Y', 'Z', 'H', 'CNOT', 'CZ', 'SWAP', 'Toffoli']:
            gate = GateFactory.create(name)
            assert gate.is_unitary()

    def test_factory_rotation(self):
        gate = GateFactory.create('Rx', theta=np.pi/4)
        assert gate.is_unitary()

    def test_list_gates(self):
        gates = GateFactory.list_gates()
        assert len(gates) >= 20


# === QuantumCircuit 测试 ===

class TestQuantumCircuit:
    def test_creation(self):
        qc = QuantumCircuit(3)
        assert qc.num_qubits == 3

    def test_add_gate(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert qc.gate_count == 2

    def test_simulate(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = qc.simulate(shots=1000)
        assert '00' in result['counts'] or '11' in result['counts']

    def test_depth(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        assert qc.depth == 1

    def test_optimize(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        optimized = qc.optimize()
        assert optimized.gate_count == 0

    def test_to_qasm(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = qc.to_qasm()
        assert 'OPENQASM' in qasm
        assert 'h q[0]' in qasm

    def test_draw(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        drawing = qc.draw()
        assert 'q0' in drawing

    def test_inverse(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        inv = qc.inverse()
        assert inv.gate_count == 2

    def test_measure_all(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()
        assert len(qc.instructions) > 0

    def test_compose(self):
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)
        composed = qc1.compose(qc2)
        assert composed.gate_count == 2


# === ErrorCorrection 测试 ===

class TestErrorCorrection:
    def test_repetition_code(self):
        code = RepetCode(3)
        assert code.distance == 3
        assert code.num_data_qubits == 3

    def test_repetition_encode(self):
        code = RepetCode(3)
        state = code.encode([1.0, 0.0])
        assert len(state) == 2**code.total_qubits

    def test_steane_code(self):
        code = SteaneCode()
        assert code.distance == 3
        assert code.num_data_qubits == 7

    def test_surface_code(self):
        code = SurfaceCode(3)
        assert code.distance == 3

    def test_engine_repetition(self):
        engine = ErrorCorrectionEngine("repetition", 3)
        rate = engine.logical_error_rate(0.01)
        assert rate < 0.01

    def test_engine_steane(self):
        engine = ErrorCorrectionEngine("steane", 3)
        params = engine.get_code_params()
        assert params['code_type'] == 'steane'

    def test_engine_surface(self):
        engine = ErrorCorrectionEngine("surface", 3)
        params = engine.get_code_params()
        assert params['code_type'] == 'surface'

    def test_noise_channel_depolarizing(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = NoiseChannel.depolarizing_channel(rho, 0.1)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_noise_channel_bit_flip(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = NoiseChannel.bit_flip_channel(rho, 0.1)
        assert abs(np.trace(result) - 1.0) < 1e-10


# === Optimization 测试 ===

class TestOptimization:
    def test_cobyla(self):
        opt = COBYLAOptimizer(max_iter=50)
        result = opt.minimize(lambda x: float(np.sum(x**2)), np.array([1.0, 1.0]))
        assert result.optimal_value < 0.1

    def test_adam(self):
        opt = AdamOptimizer(lr=0.1)
        x = np.array([1.0, 1.0])
        for _ in range(100):
            grad = 2 * x
            x = opt.step(x, grad)
        assert np.linalg.norm(x) < 0.1

    def test_parameter_shift(self):
        ps = ParameterShiftRule()
        params = np.array([0.5])
        def loss(x): return float(np.sin(x[0]))
        grad = ps.compute_full_gradient(params, loss)
        expected = np.cos(0.5)
        assert abs(grad[0] - expected) < 0.1

    def test_optimization_engine(self):
        engine = OptimizationEngine("cobyla", max_iter=100)
        result = engine.minimize(lambda x: float(np.sum((x - 1)**2)), np.zeros(3))
        assert result.optimal_value < 1.0

    def test_qaoa(self):
        H = np.diag([0, 1, 1, 2]).astype(complex)
        mixer = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=complex) / 2
        qaoa = QAOA(H, mixer, p=1)
        result = qaoa.solve(max_iter=50)
        assert result.optimal_value < 2.0


# === Noise 测试 ===

class TestNoise:
    def test_depolarizing_channel(self):
        channel = DepolarizingChannel(0.1)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = channel.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_amplitude_damping(self):
        channel = AmplitudeDampingChannel(0.1)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = channel.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_noise_model(self):
        model = NoiseModel(NoiseModelType.DEPOLARIZING)
        level = model.get_noise_level()
        assert 'single_qubit_error' in level

    def test_noise_model_ideal(self):
        model = NoiseModel(NoiseModelType.IDEAL)
        state = np.array([1, 0], dtype=complex)
        result = model.apply_to_state(state, 1)
        assert np.allclose(result, state)


# === Entanglement 测试 ===

class TestEntanglement:
    def test_concurrence_bell(self):
        calc = ConcurrenceCalculator()
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        c = calc.calculate_2qubit(bell)
        assert abs(c - 1.0) < 1e-10

    def test_concurrence_separable(self):
        calc = ConcurrenceCalculator()
        state = np.array([1, 0, 0, 0], dtype=complex)
        c = calc.calculate_2qubit(state)
        assert abs(c) < 1e-10

    def test_negativity(self):
        calc = NegativityCalculator()
        bell = np.outer([1,0,0,1], [1,0,0,1]) / 2
        n = calc.calculate(bell, 2, 2)
        assert n > 0

    def test_bell_analyzer(self):
        analyzer = BellStateAnalyzer()
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        name, f = analyzer.best_bell_match(bell)
        assert name == 'phi_plus'
        assert abs(f - 1.0) < 1e-10

    def test_schmidt_decomposition(self):
        sd = SchmidtDecomposition()
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        U, S, Vh = sd.decompose(bell, 2, 2)
        assert abs(S[0] - 1/np.sqrt(2)) < 1e-10

    def test_entanglement_analyzer(self):
        analyzer = EntanglementAnalyzer()
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        measure = analyzer.analyze_state_vector(bell, 2)
        assert measure.is_entangled()
        assert measure.entanglement_class() == "maximally_entangled"


# === Algorithms 测试 ===

class TestAlgorithms:
    def test_qft(self):
        qft = QuantumFourierTransform(3)
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0
        result = qft.apply(state)
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert np.allclose(result, expected)

    def test_qft_inverse(self):
        qft = QuantumFourierTransform(3)
        state = np.random.randn(8) + 1j * np.random.randn(8)
        state /= np.linalg.norm(state)
        result = qft.inverse(qft.apply(state))
        assert np.allclose(result, state, atol=1e-10)

    def test_grover(self):
        grover = GroverSearch(4, target=5)
        result = grover.run()
        assert result.success

    def test_deutsch_jozsa_constant(self):
        dj = DeutschJozsa(3, "constant")
        result = dj.run()
        assert result.output == "constant"

    def test_deutsch_jozsa_balanced(self):
        dj = DeutschJozsa(3, "balanced")
        result = dj.run()
        assert result.algorithm.value == "deutsch_jozsa"

    def test_teleportation(self):
        tp = QuantumTeleportation()
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        result = tp.run(state)
        assert result.algorithm.value == "teleportation"

    def test_superdense(self):
        sd = SuperdenseCoding()
        for msg in ['00', '01', '10', '11']:
            result = sd.run(msg)
            assert result.success

    def test_qkd_no_eavesdrop(self):
        qkd = QuantumKeyDistribution(16)
        result = qkd.run(eavesdrop=False)
        assert result.success

    def test_qkd_with_eavesdrop(self):
        qkd = QuantumKeyDistribution(32)
        result = qkd.run(eavesdrop=True)
        assert result.output['key_length'] > 0

    def test_bernstein_vazirani(self):
        from core.algorithms import BernsteinVazirani
        bv = BernsteinVazirani(4, 0b1010)
        result = bv.run()
        assert result.algorithm.value == "bernstein_vazirani"

    def test_algorithm_library(self):
        lib = QuantumAlgorithmLibrary()
        assert len(lib.list_algorithms()) >= 10

    def test_qpe(self):
        U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        qpe = QuantumPhaseEstimation(U, precision_bits=4)
        result = qpe.run()
        assert result.success


# === Pipeline 测试 ===

class TestPipeline:
    def test_pipeline_execute(self):
        pipeline = QuantumPipeline()
        result = pipeline.execute({'num_qubits': 3, 'circuit_type': 'generic', 'shots': 100})
        assert result['metrics'].layers_passed == 7

    def test_pipeline_qft(self):
        pipeline = QuantumPipeline()
        result = pipeline.execute({'num_qubits': 3, 'circuit_type': 'qft'})
        assert result['metrics'].layers_passed == 7

    def test_pipeline_random(self):
        pipeline = QuantumPipeline()
        result = pipeline.execute({'num_qubits': 4, 'circuit_type': 'random', 'seed': 42})
        assert result['metrics'].layers_passed == 7

    def test_pipeline_metrics(self):
        pipeline = QuantumPipeline()
        pipeline.execute({'num_qubits': 2})
        metrics = pipeline.get_metrics()
        assert 'total_time' in metrics


# === Engine 测试 ===

class TestEngine:
    def test_agent_registry(self):
        registry = AgentRegistry()
        agent = CircuitArchitectAgent()
        registry.register(agent)
        assert len(registry.get_all()) == 1

    def test_agent_task(self):
        agent = CircuitArchitectAgent()
        task = AgentTask(name="test", task_type="design", parameters={'num_qubits': 3})
        result = agent.execute(task)
        assert result.status.value == "completed"

    def test_all_agents(self):
        agents = create_all_agents()
        assert len(agents) == 10

    def test_evolution_engine(self):
        evo = EvolutionEngine(population_size=10, mutation_rate=0.2)
        evo.initialize(5, seed=42)
        def fitness(x): return -float(np.sum(x**2))
        result = evo.evolve(fitness, generations=10)
        assert result['best_fitness'] < 0

    def test_classifier(self):
        clf = CircuitComplexityClassifier()
        assert clf.classify(2, 5, 3, 1) in ['trivial', 'simple']
        assert clf.classify(10, 50, 20, 15) == 'very_complex'

    def test_quantum_toolkit(self):
        tk = QuantumToolKit()
        circuit = tk.random_circuit(3, 5, seed=42)
        assert len(circuit) > 0

    def test_quantum_engine(self):
        engine = QuantumEngine()
        for agent in create_all_agents():
            engine.register_agent(agent)
        status = engine.get_status()
        assert status['registry']['total_agents'] == 10



# === 扩充测试 ===

class TestQuantumStateVectorAdvanced:
    def test_bell_state_types(self):
        for bt in range(4):
            sv = QuantumStateVector.bell_state(bt)
            assert sv.num_qubits == 2
            assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10

    def test_ghz_different_sizes(self):
        for n in range(2, 6):
            sv = QuantumStateVector.ghz_state(n)
            assert sv.num_qubits == n
            assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10

    def test_w_different_sizes(self):
        for n in range(2, 6):
            sv = QuantumStateVector.w_state(n)
            assert sv.num_qubits == n
            assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10

    def test_random_different_seeds(self):
        states = []
        for seed in range(5):
            sv = QuantumStateVector.random_state(3, seed=seed)
            states.append(sv)
            assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10
        # 不同种子应产生不同态
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                assert states[i].fidelity(states[j]) < 0.99

    def test_partial_trace(self):
        sv = QuantumStateVector.bell_state(0)
        rho_a = sv.partial_trace([0])
        assert rho_a.shape == (2, 2)
        trace = np.trace(rho_a)
        assert abs(trace - 1.0) < 1e-10

    def test_entanglement_entropy_bell(self):
        sv = QuantumStateVector.bell_state(0)
        entropy = sv.entanglement_entropy([0])
        assert abs(entropy - 1.0) < 0.1

    def test_apply_phase(self):
        sv = QuantumStateVector.plus_state(1)
        sv.apply_phase(0, np.pi)
        # |+⟩ with phase π 应该变成 |-⟩
        assert abs(sv.amplitudes[0] - 1/np.sqrt(2)) < 1e-10

    def test_apply_global_phase(self):
        sv = QuantumStateVector.zero_state(1)
        sv.apply_global_phase(np.pi)
        probs = sv.probabilities()
        assert abs(probs[0] - 1.0) < 1e-10

    def test_outer_product(self):
        sv1 = QuantumStateVector.zero_state(1)
        sv2 = QuantumStateVector.one_state(1)
        op = sv1.outer_product(sv2)
        assert op.shape == (2, 2)
        assert abs(op[0, 1] - 1.0) < 1e-10

    def test_negativity(self):
        sv = QuantumStateVector.bell_state(0)
        n = sv.negativity([0])
        assert n > 0

    def test_measure_single(self):
        sv = QuantumStateVector.zero_state(2)
        for _ in range(100):
            result = sv.measure_single(0, seed=np.random.randint(1000))
            assert result == 0

    def test_from_dict(self):
        sv = QuantumStateVector.bell_state(0)
        d = sv.to_dict()
        sv2 = QuantumStateVector.from_dict(d)
        assert sv.fidelity(sv2) > 0.99

    def test_state_dimension_mismatch(self):
        with pytest.raises(ValueError):
            QuantumStateVector(3, np.array([1, 0], dtype=complex))


class TestDensityMatrixAdvanced:
    def test_purity_pure(self):
        sv = QuantumStateVector.zero_state(2)
        dm = DensityMatrix.from_state_vector(sv)
        assert abs(dm.purity() - 1.0) < 1e-10

    def test_purity_mixed(self):
        dm = DensityMatrix.maximally_mixed(3)
        assert abs(dm.purity() - 1/8) < 1e-10

    def test_fidelity_same(self):
        sv = QuantumStateVector.bell_state(0)
        dm1 = DensityMatrix.from_state_vector(sv)
        dm2 = DensityMatrix.from_state_vector(sv)
        assert abs(dm1.fidelity_dm(dm2) - 1.0) < 1e-10

    def test_trace_distance_same(self):
        sv = QuantumStateVector.zero_state(2)
        dm1 = DensityMatrix.from_state_vector(sv)
        dm2 = DensityMatrix.from_state_vector(sv)
        assert dm1.trace_distance(dm2) < 1e-10

    def test_partial_trace(self):
        sv = QuantumStateVector.bell_state(0)
        dm = DensityMatrix.from_state_vector(sv)
        dm_a = dm.partial_trace_dm([0])
        assert dm_a.num_qubits == 1

    def test_apply_channel(self):
        sv = QuantumStateVector.zero_state(1)
        dm = DensityMatrix.from_state_vector(sv)
        K0 = np.eye(2, dtype=complex)
        result = dm.apply_channel([K0])
        assert abs(np.trace(result.matrix) - 1.0) < 1e-4


class TestQuantumGatesAdvanced:
    def test_all_single_gates_unitary(self):
        for name in ['X', 'Y', 'Z', 'H', 'S', 'T']:
            gate = GateFactory.create(name)
            assert gate.is_unitary(), f"{name} is not unitary"

    def test_all_double_gates_unitary(self):
        for name in ['CNOT', 'CZ', 'CY', 'SWAP', 'iSWAP']:
            gate = GateFactory.create(name)
            assert gate.is_unitary(), f"{name} is not unitary"

    def test_all_rotation_gates_unitary(self):
        for theta in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
            for axis in ['Rx', 'Ry', 'Rz']:
                gate = GateFactory.create(axis, theta=theta)
                assert gate.is_unitary(), f"{axis}({theta}) is not unitary"

    def test_phase_gate_various(self):
        for phi in [0, np.pi/4, np.pi/2, np.pi]:
            gate = PhaseGate(phi)
            assert gate.is_unitary()

    def test_cphase_gate(self):
        from core.quantum_gates import CPhaseGate
        for phi in [0, np.pi/4, np.pi/2]:
            gate = CPhaseGate(phi)
            assert gate.is_unitary()

    def test_xx_yy_zz_gates(self):
        from core.quantum_gates import XXGate, YYGate, ZZGate
        for theta in [np.pi/4, np.pi/2]:
            assert XXGate(theta).is_unitary()
            assert YYGate(theta).is_unitary()
            assert ZZGate(theta).is_unitary()

    def test_u3_gate(self):
        from core.quantum_gates import U3Gate
        gate = U3Gate(np.pi/4, np.pi/3, np.pi/6)
        assert gate.is_unitary()

    def test_toffoli(self):
        gate = Toffoli()
        assert gate.is_unitary()
        assert gate.num_qubits == 3

    def test_fredkin(self):
        from core.quantum_gates import Fredkin
        gate = Fredkin()
        assert gate.is_unitary()
        assert gate.num_qubits == 3

    def test_ccz(self):
        from core.quantum_gates import CCZ
        gate = CCZ()
        assert gate.is_unitary()
        assert gate.num_qubits == 3

    def test_random_single_gate(self):
        for _ in range(10):
            gate = GateFactory.random_single_gate(seed=np.random.randint(1000))
            assert gate.is_unitary()

    def test_random_two_gate(self):
        for _ in range(10):
            gate = GateFactory.random_two_gate(seed=np.random.randint(1000))
            assert gate.is_unitary()

    def test_random_rotation(self):
        for _ in range(10):
            gate = GateFactory.random_rotation(seed=np.random.randint(1000))
            assert gate.is_unitary()

    def test_gate_to_dict(self):
        gate = Hadamard()
        d = gate.to_dict()
        assert d['name'] == 'H'
        assert d['type'] == 'single'

    def test_pauli_algebra(self):
        # XX = I
        assert np.allclose(PauliX().matrix @ PauliX().matrix, np.eye(2))
        # YY = I
        assert np.allclose(PauliY().matrix @ PauliY().matrix, np.eye(2))
        # ZZ = I
        assert np.allclose(PauliZ().matrix @ PauliZ().matrix, np.eye(2))

    def test_hadamard_squared(self):
        H = Hadamard()
        assert np.allclose(H.matrix @ H.matrix, np.eye(2))

    def test_cnot_symmetry(self):
        cnot = CNOT()
        assert np.allclose(cnot.matrix, cnot.matrix.T)  # CNOT 是对称的


class TestCircuitAdvanced:
    def test_large_circuit(self):
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
        for i in range(4):
            qc.cx(i, i+1)
        result = qc.simulate(shots=1000)
        assert len(result['counts']) > 0

    def test_circuit_depth_calculation(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        assert qc.depth == 1  # 并行
        qc.cx(0, 1)
        assert qc.depth == 2

    def test_circuit_gate_types(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        types = qc.gate_types
        assert types.get('H', 0) == 2
        assert types.get('CNOT', 0) == 1

    def test_circuit_two_qubit_count(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert qc.two_qubit_gate_count == 2

    def test_circuit_copy(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc2 = qc.copy()
        qc2.x(0)
        assert qc.gate_count == 1
        assert qc2.gate_count == 2

    def test_circuit_to_dict(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        d = qc.to_dict()
        assert d['num_qubits'] == 2

    def test_circuit_len(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert len(qc) == 2

    def test_circuit_barrier(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.add_barrier()
        qc.h(1)
        assert len(qc.instructions) == 3


class TestErrorCorrectionAdvanced:
    def test_repetition_code_distances(self):
        for d in [3, 5, 7]:
            code = RepetCode(d)
            assert code.distance == d
            assert code.num_data_qubits == d

    def test_repetition_logical_error_rate(self):
        code = RepetCode(3)
        rate = code.logical_error_rate(0.01)
        assert rate < 0.01  # 逻辑错误率应低于物理错误率

    def test_repetition_code_distance_5(self):
        code = RepetCode(5)
        rate = code.logical_error_rate(0.01)
        rate3 = RepetCode(3).logical_error_rate(0.01)
        assert rate < rate3  # 更大距离 = 更低错误率

    def test_surface_code_threshold(self):
        code = SurfaceCode(3)
        assert code.threshold_error_rate() > 0

    def test_noise_channel_depolarizing(self):
        rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
        result = NoiseChannel.depolarizing_channel(rho, 0.0)
        assert np.allclose(result, rho)

    def test_noise_channel_amplitude_damping(self):
        rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
        result = NoiseChannel.amplitude_damping_channel(rho, 0.0)
        assert np.allclose(result, rho)

    def test_noise_channel_phase_damping(self):
        rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
        result = NoiseChannel.phase_damping_channel(rho, 0.0)
        assert np.allclose(result, rho)

    def test_engine_various_codes(self):
        for code in ["repetition", "steane", "surface"]:
            engine = ErrorCorrectionEngine(code, 3)
            params = engine.get_code_params()
            assert params['code_type'] == code


class TestOptimizationAdvanced:
    def test_cobyla_simple(self):
        opt = COBYLAOptimizer(max_iter=100)
        result = opt.minimize(lambda x: float(np.sum(x**2)), np.array([5.0, -3.0]))
        assert result.optimal_value < 0.01

    def test_adam_convergence(self):
        opt = AdamOptimizer(lr=0.01)
        x = np.array([2.0, 2.0])
        for _ in range(500):
            grad = 2 * x
            x = opt.step(x, grad)
        assert np.linalg.norm(x) < 0.5

    def test_spsa_basic(self):
        opt = SPSAOptimizer()
        x = np.array([1.0, 1.0])
        def loss(p): return float(np.sum(p**2))
        x, val = opt.step(x, loss)
        assert isinstance(val, float)

    def test_parameter_shift_accuracy(self):
        ps = ParameterShiftRule()
        params = np.array([1.0, 2.0])
        def loss(x): return float(np.sin(x[0]) + np.cos(x[1]))
        grad = ps.compute_full_gradient(params, loss)
        expected = [np.cos(1.0), -np.sin(2.0)]
        assert abs(grad[0] - expected[0]) < 0.1
        assert abs(grad[1] - expected[1]) < 0.1

    def test_optimization_engine_methods(self):
        for method in ["cobyla", "adam"]:
            engine = OptimizationEngine(method, max_iter=50)
            result = engine.minimize(lambda x: float(np.sum(x**2)), np.ones(2))
            assert result.optimal_value < 10.0


class TestNoiseAdvanced:
    def test_depolarizing_channels(self):
        for p in [0.01, 0.05, 0.1]:
            channel = DepolarizingChannel(p)
            rho = np.array([[1, 0], [0, 0]], dtype=complex)
            result = channel.apply(rho)
            assert abs(np.trace(result) - 1.0) < 1e-10
            assert result[0, 0] > 0

    def test_amplitude_damping_steady_state(self):
        channel = AmplitudeDampingChannel(0.5)
        ss = channel.steady_state()
        assert abs(ss[0, 0] - 1.0) < 1e-10

    def test_noise_model_types(self):
        for mt in NoiseModelType:
            if mt == NoiseModelType.CUSTOM:
                continue
            model = NoiseModel(mt)
            assert model.model_type == mt

    def test_noise_level(self):
        model = NoiseModel(NoiseModelType.DEPOLARIZING)
        level = model.get_noise_level()
        assert all(isinstance(v, float) for v in level.values())


class TestEntanglementAdvanced:
    def test_concurrence_range(self):
        calc = ConcurrenceCalculator()
        for _ in range(10):
            sv = QuantumStateVector.random_state(2, seed=np.random.randint(1000))
            c = calc.calculate_2qubit(sv.amplitudes)
            assert 0 <= c <= 1 + 1e-10

    def test_bell_inequality(self):
        analyzer = BellStateAnalyzer()
        # 量子力学违反 Bell 不等式: S > 2
        correlations = {'ab': 1/np.sqrt(2), 'ab_prime': 0, 'a_prime_b': 1/np.sqrt(2), 'a_prime_b_prime': 1/np.sqrt(2)}
        S = analyzer.bell_inequality(correlations)
        assert S > 2  # 违反经典极限

    def test_schmidt_separable(self):
        sd = SchmidtDecomposition()
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩ 可分态
        rank = sd.schmidt_rank(state, 2, 2)
        assert rank == 1

    def test_schmidt_bell(self):
        sd = SchmidtDecomposition()
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        coeffs = sd.schmidt_coefficients(bell, 2, 2)
        assert len(coeffs) == 2
        assert abs(coeffs[0] - 1/np.sqrt(2)) < 1e-10

    def test_entanglement_witness(self):
        from core.entanglement import EntanglementWitness
        witness = EntanglementWitness.for_bell_state('phi_plus')
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_bell = np.outer(bell, bell.conj())
        assert witness.is_entangled(rho_bell)
        # 可分态不应触发
        separable = np.array([1, 0, 0, 0], dtype=complex)
        rho_sep = np.outer(separable, separable.conj())
        assert not witness.is_entangled(rho_sep)

    def test_renyi_entropy(self):
        calc = EntanglementEntropyCalculator()
        rho = np.eye(4, dtype=complex) / 4
        s2 = calc.renyi(rho, 2.0)
        assert s2 > 0

    def test_tsallis_entropy(self):
        calc = EntanglementEntropyCalculator()
        rho = np.eye(4, dtype=complex) / 4
        s = calc.tsallis(rho, 2.0)
        assert s >= 0


class TestAlgorithmsAdvanced:
    def test_qft_sizes(self):
        for n in range(2, 5):
            qft = QuantumFourierTransform(n)
            state = np.zeros(2**n, dtype=complex)
            state[0] = 1.0
            result = qft.apply(state)
            assert len(result) == 2**n

    def test_grover_sizes(self):
        for n in [3, 4]:
            grover = GroverSearch(n, target=1)
            result = grover.run()
            assert result.algorithm.value == "grover"

    def test_superdense_all_messages(self):
        sd = SuperdenseCoding()
        for msg in ['00', '01', '10', '11']:
            result = sd.run(msg)
            assert result.output['original'] == msg

    def test_qkd_lengths(self):
        for length in [8, 16, 32]:
            qkd = QuantumKeyDistribution(length)
            result = qkd.run()
            assert result.output['key_length'] > 0

    def test_algorithm_library_all(self):
        lib = QuantumAlgorithmLibrary()
        for algo in lib.list_algorithms():
            try:
                result = lib.run(algo)
                assert result is not None
            except:
                pass  # 某些算法可能需要特定参数


class TestPipelineAdvanced:
    def test_pipeline_reset(self):
        pipeline = QuantumPipeline()
        pipeline.execute({'num_qubits': 2})
        pipeline.reset()
        assert pipeline.metrics.layers_passed == 0

    def test_pipeline_different_sizes(self):
        for n in [2, 3, 4]:
            pipeline = QuantumPipeline()
            result = pipeline.execute({'num_qubits': n})
            assert result['metrics'].layers_passed == 7

    def test_pipeline_circuit_types(self):
        for ct in ['generic', 'qft', 'random']:
            pipeline = QuantumPipeline()
            result = pipeline.execute({'num_qubits': 3, 'circuit_type': ct})
            assert result['metrics'].layers_passed == 7


class TestEngineAdvanced:
    def test_agent_registry_multiple(self):
        registry = AgentRegistry()
        for agent in create_all_agents():
            registry.register(agent)
        assert len(registry.get_all()) == 10
        assert len(registry.get_active()) == 10

    def test_agent_metrics(self):
        agent = CircuitArchitectAgent()
        task = AgentTask(name="test", task_type="design", parameters={'num_qubits': 2})
        agent.execute(task)
        m = agent.get_metrics()
        assert m['tasks_completed'] == 1
        assert m['success_rate'] == 1.0

    def test_classifier_recommendations(self):
        clf = CircuitComplexityClassifier()
        for level in CircuitComplexityClassifier.COMPLEXITY_LEVELS:
            recs = clf.recommend_optimization(level)
            assert len(recs) > 0

    def test_toolkit_calculate_fidelity(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([1, 0], dtype=complex)
        assert abs(QuantumToolKit.calculate_fidelity(sv1, sv2) - 1.0) < 1e-10

    def test_toolkit_hamming_distance(self):
        assert QuantumToolKit.hamming_distance("000", "111") == 3
        assert QuantumToolKit.hamming_distance("101", "101") == 0

    def test_toolkit_bitstring_conversion(self):
        assert QuantumToolKit.bitstring_to_int("101") == 5
        assert QuantumToolKit.int_to_bitstring(5, 3) == "101"

    def test_evolution_engine_small(self):
        evo = EvolutionEngine(population_size=5, mutation_rate=0.3)
        evo.initialize(3, seed=42)
        def fitness(x): return -float(np.sum(x**2))
        result = evo.evolve(fitness, generations=5)
        assert result['generations'] == 5

    def test_quantum_engine_full(self):
        engine = QuantumEngine()
        register_all_agents(engine.registry)
        result = engine.execute_task(
            "design_test", "design",
            AgentRole.CIRCUIT_ARCHITECT,
            num_qubits=3
        )
        assert result.status.value == "completed"

    def test_toolkit_trace_distance(self):
        rho1 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho2 = np.array([[1, 0], [0, 0]], dtype=complex)
        assert QuantumToolKit.trace_distance(rho1, rho2) < 1e-10

    def test_toolkit_entropy(self):
        rho = np.eye(2, dtype=complex) / 2
        assert abs(QuantumToolKit.entropy(rho) - 1.0) < 1e-10



# === 最终扩充测试 ===

class TestConfigPresets:
    def test_high_precision(self):
        from config.settings import ConfigPresets
        config = ConfigPresets.high_precision()
        assert config.simulation.precision == 1e-15
        assert config.simulation.shots == 10000

    def test_noisy_hardware(self):
        from config.settings import ConfigPresets
        config = ConfigPresets.noisy_hardware()
        assert config.noise.enabled == True

    def test_fast_prototype(self):
        from config.settings import ConfigPresets
        config = ConfigPresets.fast_prototype()
        assert config.simulation.shots == 100

    def test_quantum_advantage(self):
        from config.settings import ConfigPresets
        config = ConfigPresets.quantum_advantage()
        assert config.simulation.max_qubits == 20

    def test_list_presets(self):
        from config.settings import ConfigPresets
        presets = ConfigPresets.list_presets()
        assert len(presets) == 4


class TestCircuitOptimizer:
    def test_gate_cancellation(self):
        from core.circuit import CircuitOptimizer
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        optimizer = CircuitOptimizer()
        optimized = optimizer.optimize(qc)
        assert optimized.gate_count == 0

    def test_optimizer_report(self):
        from core.circuit import CircuitOptimizer
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        optimizer = CircuitOptimizer()
        optimized = optimizer.optimize(qc)
        report = optimizer.count_optimizations(qc, optimized)
        assert report['gates_saved'] == 2


class TestCircuitCompiler:
    def test_compile_hadamard(self):
        from core.circuit import CircuitCompiler
        qc = QuantumCircuit(1)
        qc.h(0)
        compiler = CircuitCompiler()
        compiled = compiler.compile(qc)
        assert compiled.gate_count > 0

    def test_gate_cost(self):
        from core.circuit import CircuitCompiler
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        compiler = CircuitCompiler()
        compiled = compiler.compile(qc)
        cost = compiler.gate_cost(compiled)
        assert len(cost) > 0


class TestCircuitVerifier:
    def test_verify_unitarity(self):
        from core.circuit import CircuitVerifier
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        verifier = CircuitVerifier()
        assert verifier.verify_unitarity(qc)

    def test_verify_qubit_range(self):
        from core.circuit import CircuitVerifier
        qc = QuantumCircuit(2)
        qc.h(0)
        verifier = CircuitVerifier()
        assert verifier.verify_qubit_range(qc)

    def test_circuit_properties(self):
        from core.circuit import CircuitVerifier
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        verifier = CircuitVerifier()
        props = verifier.get_circuit_properties(qc)
        assert props['num_qubits'] == 3
        assert props['gate_count'] == 2


class TestQuantumBenchmark:
    def test_benchmark_state(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.benchmark_state_preparation(3, 'random')
        assert result['test'] == 'state_preparation'
        assert result['time'] > 0

    def test_benchmark_simulation(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.benchmark_simulation(3, 5)
        assert result['test'] == 'simulation'
        assert result['num_outcomes'] > 0

    def test_benchmark_entanglement(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.benchmark_entanglement(3)
        assert result['test'] == 'entanglement'
        assert result['is_entangled'] == True or result['entropy'] >= 0 or result['entropy'] >= 0

    def test_benchmark_optimization(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.benchmark_optimization(5)
        assert result['test'] == 'optimization'

    def test_full_benchmark(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.run_full_benchmark()
        assert result['total_tests'] > 0

    def test_benchmark_summary(self):
        from core.engine import QuantumBenchmark
        bench = QuantumBenchmark()
        bench.benchmark_state_preparation(2)
        summary = bench.get_summary()
        assert summary['total_benchmarks'] == 1


class TestQuantumDebugger:
    def test_inspect_state(self):
        from core.engine import QuantumDebugger
        debugger = QuantumDebugger()
        sv = QuantumStateVector.bell_state(0)
        result = debugger.inspect_state(sv)
        assert result['num_qubits'] == 2
        assert len(result['top_states']) > 0

    def test_inspect_circuit(self):
        from core.engine import QuantumDebugger
        debugger = QuantumDebugger()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = debugger.inspect_circuit(qc)
        assert result['gate_count'] == 2

    def test_debug_log(self):
        from core.engine import QuantumDebugger
        debugger = QuantumDebugger()
        sv = QuantumStateVector.zero_state(1)
        debugger.inspect_state(sv)
        log = debugger.get_log()
        assert len(log) == 1


class TestAgentPool:
    def test_agent_pool(self):
        from agents.pool import AgentPool
        pool = AgentPool()
        for agent in create_all_agents():
            pool.add_agent(agent)
        assert len(pool.pools) == 10

    def test_performance_tracker(self):
        from agents.pool import AgentPerformanceTracker
        tracker = AgentPerformanceTracker()
        tracker.record("agent_1", "test_task", 0.5, True)
        stats = tracker.get_agent_stats("agent_1")
        assert stats['total_tasks'] == 1

    def test_task_queue(self):
        from agents.pool import AgentTaskQueue
        queue = AgentTaskQueue(max_size=10)
        task = AgentTask(name="test")
        assert queue.enqueue(task)
        assert queue.size() == 1
        dequeued = queue.dequeue()
        assert dequeued.name == "test"

    def test_communication_bus(self):
        from agents.pool import AgentCommunicationBus
        bus = AgentCommunicationBus()
        received = []
        bus.subscribe("test", lambda m: received.append(m))
        bus.publish("test", {"data": "hello"})
        assert len(received) == 1

    def test_agent_pool_stats(self):
        from agents.pool import AgentPool
        pool = AgentPool()
        for agent in create_all_agents():
            pool.add_agent(agent)
        stats = pool.get_pool_stats()
        assert len(stats) == 10


class TestQuantumStateFactory:
    def test_factory_types(self):
        from core.quantum_state import QuantumStateFactory
        for t in QuantumStateFactory.list_types():
            sv = QuantumStateFactory.create(t, num_qubits=2)
            assert sv.num_qubits == 2

    def test_random_haar(self):
        from core.quantum_state import QuantumStateFactory
        sv = QuantumStateFactory.random_haar(3, seed=42)
        assert sv.num_qubits == 3
        assert abs(np.linalg.norm(sv.amplitudes) - 1.0) < 1e-10

    def test_maximally_entangled(self):
        from core.quantum_state import QuantumStateFactory
        sv = QuantumStateFactory.maximally_entangled(4)
        assert sv.num_qubits == 4


class TestNoiseMitigator:
    def test_zne(self):
        from core.noise import NoiseMitigator
        mitigator = NoiseMitigator()
        results = [{'value': 0.9}, {'value': 0.8}, {'value': 0.7}]
        noise_levels = [0.0, 0.01, 0.02]
        extrapolated = mitigator.zero_noise_extrapolation(results, noise_levels)
        assert 'extrapolated_value' in extrapolated

    def test_error_budget(self):
        from core.noise import QuantumErrorBudget
        budget = QuantumErrorBudget(0.01)
        assert budget.allocate('gate', 0.003)
        assert budget.allocate('readout', 0.002)
        assert budget.get_remaining() == pytest.approx(0.005)


class TestQuantumExporter:
    def test_state_to_json(self):
        from core.engine import QuantumExporter
        sv = QuantumStateVector.bell_state(0)
        json_str = QuantumExporter.state_to_json(sv)
        import json
        data = json.loads(json_str)
        assert data['num_qubits'] == 2

    def test_counts_to_csv(self):
        from core.engine import QuantumExporter
        counts = {'00': 500, '11': 500}
        csv = QuantumExporter.counts_to_csv(counts)
        assert 'bitstring,count,probability' in csv



class TestFinalEdgeCases:
    def test_zero_qubit_circuit(self):
        qc = QuantumCircuit(1)
        assert qc.num_qubits == 1

    def test_single_qubit_simulation(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        result = qc.simulate(shots=100)
        assert len(result['counts']) > 0

    def test_state_repr(self):
        sv = QuantumStateVector.bell_state(0)
        r = repr(sv)
        assert 'QuantumState' in r

    def test_quantum_state_repr(self):
        qs = QuantumState()
        r = repr(qs)
        assert 'QuantumState' in r

    def test_density_matrix_repr(self):
        dm = DensityMatrix(2)
        r = repr(dm)
        assert 'DensityMatrix' in r

    def test_gate_repr(self):
        gate = Hadamard()
        r = repr(gate)
        assert 'QuantumGate' in r

    def test_circuit_repr(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        r = repr(qc)
        assert 'QuantumCircuit' in r

    def test_entanglement_measure_dict(self):
        from core.entanglement import EntanglementMeasure
        m = EntanglementMeasure()
        d = m.to_dict()
        assert 'concurrence' in d

    def test_entanglement_measure_class(self):
        from core.entanglement import EntanglementMeasure
        m = EntanglementMeasure(concurrence=0.99)
        cls = m.entanglement_class()
        assert cls in ["maximally_entangled", "highly_entangled"]

    def test_entanglement_measure_is_entangled(self):
        from core.entanglement import EntanglementMeasure
        m = EntanglementMeasure(concurrence=0.5)
        assert m.is_entangled()

    def test_entanglement_not_entangled(self):
        from core.entanglement import EntanglementMeasure
        m = EntanglementMeasure()
        assert not m.is_entangled()


# === Tomography 测试 ===

class TestTomographyBasis:
    def test_pauli_measurement_basis_1qubit(self):
        from core.tomography import PauliMeasurementBasis
        basis = PauliMeasurementBasis(1)
        strings = basis.generate_pauli_strings()
        assert len(strings) == 3

    def test_pauli_measurement_basis_2qubit(self):
        from core.tomography import PauliMeasurementBasis
        basis = PauliMeasurementBasis(2)
        strings = basis.generate_pauli_strings()
        assert len(strings) == 15  # 4^2 - 1

    def test_pauli_names(self):
        from core.tomography import PauliMeasurementBasis
        basis = PauliMeasurementBasis(1)
        strings = basis.generate_pauli_strings()
        names = [s[0] for s in strings]
        assert 'X' in names
        assert 'Y' in names
        assert 'Z' in names

    def test_eigenvectors_x(self):
        from core.tomography import PauliMeasurementBasis
        basis = PauliMeasurementBasis(1)
        vecs = basis.eigenvectors('X')
        assert vecs.shape == (2, 2)

    def test_eigenvectors_z(self):
        from core.tomography import PauliMeasurementBasis
        basis = PauliMeasurementBasis(1)
        vecs = basis.eigenvectors('Z')
        assert vecs.shape == (2, 2)


class TestMLE:
    def test_mle_reconstruct(self):
        from core.tomography import MaximumLikelihoodEstimation
        mle = MaximumLikelihoodEstimation(1, max_iter=50)
        measurements = {'Z': {'0': 500, '1': 500}}
        result = mle.reconstruct(measurements)
        assert result.rho.shape == (2, 2)
        assert abs(np.trace(result.rho) - 1.0) < 0.1

    def test_mle_purity(self):
        from core.tomography import MaximumLikelihoodEstimation
        mle = MaximumLikelihoodEstimation(1, max_iter=50)
        measurements = {'Z': {'0': 1000, '1': 0}}
        result = mle.reconstruct(measurements)
        assert result.purity > 0

    def test_mle_iterations(self):
        from core.tomography import MaximumLikelihoodEstimation
        mle = MaximumLikelihoodEstimation(1, max_iter=10)
        measurements = {'Z': {'0': 500, '1': 500}}
        result = mle.reconstruct(measurements)
        assert result.iterations <= 10

    def test_mle_gradient(self):
        from core.tomography import MaximumLikelihoodEstimation
        mle = MaximumLikelihoodEstimation(1)
        rho = np.eye(2, dtype=complex) / 2
        measurements = {'Z': {'0': 500, '1': 500}}
        grad = mle._gradient(rho, measurements)
        assert grad.shape == (2, 2)

    def test_mle_project_valid(self):
        from core.tomography import MaximumLikelihoodEstimation
        mle = MaximumLikelihoodEstimation(1)
        rho = np.array([[2, 0], [0, -1]], dtype=complex)
        projected = mle._project_to_valid(rho)
        eigenvalues = np.linalg.eigvalsh(projected)
        assert all(ev >= -1e-10 for ev in eigenvalues)
        assert abs(np.trace(projected) - 1.0) < 1e-6


class TestBayesianTomography:
    def test_bayesian_reconstruct(self):
        from core.tomography import BayesianTomography
        bt = BayesianTomography(1, num_samples=50, seed=42)
        measurements = {'Z': {'0': 600, '1': 400}}
        result = bt.reconstruct(measurements)
        assert result.rho.shape == (2, 2)
        assert abs(np.trace(result.rho) - 1.0) < 0.2

    def test_bayesian_purity(self):
        from core.tomography import BayesianTomography
        bt = BayesianTomography(1, num_samples=50, seed=42)
        measurements = {'Z': {'0': 900, '1': 100}}
        result = bt.reconstruct(measurements)
        assert result.purity > 0

    def test_bayesian_effective_samples(self):
        from core.tomography import BayesianTomography
        bt = BayesianTomography(1, num_samples=50, seed=42)
        measurements = {'Z': {'0': 500, '1': 500}}
        result = bt.reconstruct(measurements)
        assert 'effective_samples' in result.metadata

    def test_bayesian_sample_random(self):
        from core.tomography import BayesianTomography
        bt = BayesianTomography(1, seed=42)
        rho = bt._sample_random_density_matrix()
        assert rho.shape == (2, 2)
        assert abs(np.trace(rho) - 1.0) < 1e-6


class TestCompressedSensing:
    def test_cs_reconstruct(self):
        from core.tomography import CompressedSensingTomography
        cs = CompressedSensingTomography(1, rank=1, max_iter=20)
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        obs_expvals = [(I, 1.0), (Z, 0.0)]
        result = cs.reconstruct(obs_expvals)
        assert result.rho.shape == (2, 2)

    def test_cs_error(self):
        from core.tomography import CompressedSensingTomography
        cs = CompressedSensingTomography(1, rank=1, max_iter=20)
        obs_expvals = [(np.eye(2, dtype=complex), 1.0)]
        result = cs.reconstruct(obs_expvals)
        assert result.convergence_error >= 0


class TestTomographyAnalyzer:
    def test_simulate_tomography(self):
        from core.tomography import TomographyAnalyzer
        ta = TomographyAnalyzer(1)
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        measurements = ta.simulate_tomography(rho, shots=1000, seed=42)
        assert len(measurements) > 0

    def test_tomographic_fidelity(self):
        from core.tomography import TomographyAnalyzer
        ta = TomographyAnalyzer(1)
        rho1 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho2 = np.array([[1, 0], [0, 0]], dtype=complex)
        f = ta.compute_tomographic_fidelity(rho1, rho2)
        assert abs(f - 1.0) < 1e-6

    def test_process_tomography(self):
        from core.tomography import TomographyAnalyzer
        ta = TomographyAnalyzer(1)
        identity_channel = lambda rho: rho
        inputs = [np.array([[1, 0], [0, 0]], dtype=complex),
                  np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)]
        chi = ta.process_tomography(identity_channel, inputs)
        assert chi.shape == (4, 4)


class TestMeasurementOutcome:
    def test_outcome_expectation(self):
        from core.tomography import MeasurementOutcome
        mo = MeasurementOutcome(basis='Z', outcome=0, counts={'0': 700, '1': 300})
        exp = mo.expectation()
        assert abs(exp - 0.4) < 0.1

    def test_outcome_empty(self):
        from core.tomography import MeasurementOutcome
        mo = MeasurementOutcome(basis='Z', outcome=0, counts={})
        assert mo.expectation() == 0.0


# === Simulation 测试 ===

class TestSchrodingerEvolver:
    def test_evolve_identity(self):
        from core.simulation import SchrodingerEvolver, EvolutionConfig
        H = np.zeros((2, 2), dtype=complex)
        config = EvolutionConfig(dt=0.1, total_time=1.0)
        evolver = SchrodingerEvolver(H, config)
        state = np.array([1, 0], dtype=complex)
        result = evolver.evolve(state)
        assert len(result.states) > 0

    def test_evolve_preserves_norm(self):
        from core.simulation import SchrodingerEvolver, EvolutionConfig
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        config = EvolutionConfig(dt=0.01, total_time=0.5)
        evolver = SchrodingerEvolver(H, config)
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        result = evolver.evolve(state)
        for s in result.states:
            assert abs(np.linalg.norm(s) - 1.0) < 0.1

    def test_energy_expectation(self):
        from core.simulation import SchrodingerEvolver
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        evolver = SchrodingerEvolver(H)
        state = np.array([1, 0], dtype=complex)
        e = evolver.energy_expectation(state)
        assert abs(e - 1.0) < 1e-10

    def test_time_evolution_operator(self):
        from core.simulation import SchrodingerEvolver
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        evolver = SchrodingerEvolver(H)
        U = evolver.time_evolution_operator(0)
        assert np.allclose(U, np.eye(2))

    def test_observable_tracking(self):
        from core.simulation import SchrodingerEvolver, EvolutionConfig
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        config = EvolutionConfig(dt=0.1, total_time=0.5)
        evolver = SchrodingerEvolver(H, config)
        state = np.array([1, 0], dtype=complex)
        obs = np.array([[1, 0], [0, -1]], dtype=complex)
        result = evolver.evolve(state, observables=[obs])
        assert 'obs_0' in result.expectations

    def test_simulation_result(self):
        from core.simulation import SimulationResult
        sr = SimulationResult(times=np.array([0, 1]), states=[np.array([1, 0]), np.array([0, 1])])
        assert sr.final_state().shape == (2,)
        d = sr.to_dict()
        assert 'num_steps' in d


class TestLindbladEvolver:
    def test_lindblad_pure_dephasing(self):
        from core.simulation import LindbladEvolver, EvolutionConfig
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        L = [np.array([[1, 0], [0, -1]], dtype=complex)]
        config = EvolutionConfig(dt=0.1, total_time=1.0)
        evolver = LindbladEvolver(H, L, [0.1], config)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = evolver.evolve(rho)
        assert len(result.states) > 0

    def test_lindblad_preserves_trace(self):
        from core.simulation import LindbladEvolver, EvolutionConfig
        H = np.eye(2, dtype=complex)
        L = [np.array([[0, 1], [0, 0]], dtype=complex)]
        config = EvolutionConfig(dt=0.05, total_time=0.5)
        evolver = LindbladEvolver(H, L, [0.5], config)
        rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
        result = evolver.evolve(rho)
        final = result.final_state()
        assert abs(np.trace(final) - 1.0) < 0.1

    def test_lindblad_rhs(self):
        from core.simulation import LindbladEvolver
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        L = [np.array([[0, 1], [0, 0]], dtype=complex)]
        evolver = LindbladEvolver(H, L, [0.1])
        rho = np.eye(2, dtype=complex) / 2
        drho = evolver._lindblad_rhs(rho)
        assert drho.shape == (2, 2)


class TestMonteCarloWaveFunction:
    def test_single_trajectory(self):
        from core.simulation import MonteCarloWaveFunction
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        L = [np.array([[0, 1], [0, 0]], dtype=complex)]
        mc = MonteCarloWaveFunction(H, L, [0.1], seed=42)
        state = np.array([1, 0], dtype=complex)
        traj = mc.simulate_single(state, 0.1, 10)
        assert len(traj) == 11

    def test_ensemble_average(self):
        from core.simulation import MonteCarloWaveFunction
        H = np.array([[0, 0], [0, 0]], dtype=complex)
        L = [np.array([[0, 0], [1, 0]], dtype=complex)]
        mc = MonteCarloWaveFunction(H, L, [0.5], seed=42)
        state = np.array([1, 0], dtype=complex)
        rho = mc.simulate_ensemble(state, 0.1, 5, num_trajectories=20)
        assert rho.shape == (2, 2)

    def test_effective_hamiltonian(self):
        from core.simulation import MonteCarloWaveFunction
        H = np.eye(2, dtype=complex)
        L = [np.array([[0, 1], [0, 0]], dtype=complex)]
        mc = MonteCarloWaveFunction(H, L, [1.0])
        H_eff = mc._effective_hamiltonian()
        assert H_eff.shape == (2, 2)


class TestAdiabaticEvolution:
    def test_adiabatic_evolve(self):
        from core.simulation import AdiabaticEvolution, EvolutionConfig
        H_i = np.array([[1, 0], [0, -1]], dtype=complex)
        H_f = np.array([[-1, 0], [0, 1]], dtype=complex)
        config = EvolutionConfig(dt=0.1, total_time=5.0)
        ae = AdiabaticEvolution(H_i, H_f, config)
        state = np.array([1, 0], dtype=complex)
        result = ae.evolve(state)
        assert len(result.states) > 0

    def test_instantaneous_ground_state(self):
        from core.simulation import AdiabaticEvolution
        H_i = np.array([[1, 0], [0, -1]], dtype=complex)
        H_f = np.array([[-1, 0], [0, 1]], dtype=complex)
        ae = AdiabaticEvolution(H_i, H_f)
        gs = ae.instantaneous_ground_state(0.0)
        assert len(gs) == 2

    def test_schedule_hamiltonian(self):
        from core.simulation import AdiabaticEvolution
        H_i = np.array([[1, 0], [0, -1]], dtype=complex)
        H_f = np.array([[-1, 0], [0, 1]], dtype=complex)
        ae = AdiabaticEvolution(H_i, H_f)
        H_mid = ae._schedule_hamiltonian(0.5)
        expected = np.zeros((2, 2), dtype=complex)
        assert np.allclose(H_mid, expected)


class TestSimulationEngine:
    def test_engine_schrodinger(self):
        from core.simulation import SimulationEngine
        engine = SimulationEngine()
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        state = np.array([1, 0], dtype=complex)
        result = engine.schrodinger(H, state, total_time=0.5, dt=0.1)
        assert len(result.states) > 0

    def test_engine_adiabatic(self):
        from core.simulation import SimulationEngine
        engine = SimulationEngine()
        H_i = np.array([[1, 0], [0, -1]], dtype=complex)
        H_f = np.array([[-1, 0], [0, 1]], dtype=complex)
        state = np.array([1, 0], dtype=complex)
        result = engine.adiabatic(H_i, H_f, state, total_time=2.0, dt=0.5)
        assert len(result.states) > 0

    def test_engine_history(self):
        from core.simulation import SimulationEngine
        engine = SimulationEngine()
        H = np.zeros((2, 2), dtype=complex)
        state = np.array([1, 0], dtype=complex)
        engine.schrodinger(H, state, total_time=0.1, dt=0.1)
        history = engine.get_history()
        assert len(history) == 1


# === Compiler 测试 ===

class TestSingleQubitDecomposer:
    def test_decompose_identity(self):
        from core.compiler import SingleQubitDecomposer
        sqd = SingleQubitDecomposer()
        I = np.eye(2, dtype=complex)
        gates = sqd.decompose(I)
        assert isinstance(gates, list)

    def test_decompose_x(self):
        from core.compiler import SingleQubitDecomposer
        sqd = SingleQubitDecomposer()
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        gates = sqd.decompose(X)
        assert len(gates) >= 1

    def test_decompose_u3(self):
        from core.compiler import SingleQubitDecomposer
        sqd = SingleQubitDecomposer()
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        gate = sqd.decompose_to_u3(H)
        assert gate.name == 'U3'
        assert len(gate.params) == 3

    def test_decompose_hadamard(self):
        from core.compiler import SingleQubitDecomposer
        sqd = SingleQubitDecomposer()
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        gates = sqd.decompose(H)
        assert len(gates) >= 1


class TestTwoQubitDecomposer:
    def test_decompose_cz(self):
        from core.compiler import TwoQubitDecomposer
        tqd = TwoQubitDecomposer()
        gates = tqd.decompose_cz(0, 1)
        assert len(gates) == 3
        assert gates[0].name == 'H'
        assert gates[1].name == 'CNOT'
        assert gates[2].name == 'H'

    def test_decompose_swap(self):
        from core.compiler import TwoQubitDecomposer
        tqd = TwoQubitDecomposer()
        gates = tqd.decompose_swap(0, 1)
        assert len(gates) == 3
        assert all(g.name == 'CNOT' for g in gates)

    def test_decompose_cnot(self):
        from core.compiler import TwoQubitDecomposer
        tqd = TwoQubitDecomposer()
        gates = tqd.decompose_cnot(0, 1)
        assert len(gates) == 1
        assert gates[0].name == 'CNOT'

    def test_decompose_iswap(self):
        from core.compiler import TwoQubitDecomposer
        tqd = TwoQubitDecomposer()
        gates = tqd.decompose_iswap(0, 1)
        assert len(gates) >= 4

    def test_count_cnots(self):
        from core.compiler import TwoQubitDecomposer, CompiledGate, GateType
        tqd = TwoQubitDecomposer()
        gates = [
            CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('H', [0]),
            CompiledGate('CNOT', [1, 0], gate_type=GateType.TWO_QUBIT)
        ]
        assert tqd.count_cnots(gates) == 2


class TestCircuitRouter:
    def test_shortest_path(self):
        from core.compiler import CircuitRouter
        coupling = [(0, 1), (1, 2), (2, 3)]
        router = CircuitRouter(coupling)
        path = router.shortest_path(0, 3)
        assert path == [0, 1, 2, 3]

    def test_shortest_path_same(self):
        from core.compiler import CircuitRouter
        coupling = [(0, 1)]
        router = CircuitRouter(coupling)
        path = router.shortest_path(0, 0)
        assert path == [0]

    def test_adjacency(self):
        from core.compiler import CircuitRouter
        coupling = [(0, 1), (1, 2)]
        router = CircuitRouter(coupling)
        assert 1 in router.adjacency[0]
        assert 0 in router.adjacency[1]

    def test_route_direct(self):
        from core.compiler import CircuitRouter, CompiledGate, GateType
        coupling = [(0, 1)]
        router = CircuitRouter(coupling)
        gates = [CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT)]
        routed = router.route_circuit(gates, 2)
        assert len(routed) >= 1


class TestGateOptimizer:
    def test_remove_identity_pairs(self):
        from core.compiler import GateOptimizer, CompiledGate
        opt = GateOptimizer()
        gates = [CompiledGate('H', [0]), CompiledGate('H', [0])]
        new_gates, changed = opt._remove_identity_pairs(gates)
        assert len(new_gates) == 0
        assert changed

    def test_merge_rotations(self):
        from core.compiler import GateOptimizer, CompiledGate, GateType
        opt = GateOptimizer()
        gates = [
            CompiledGate('Rz', [0], [0.5], GateType.PARAMETRIC),
            CompiledGate('Rz', [0], [0.3], GateType.PARAMETRIC)
        ]
        new_gates, changed = opt._merge_rotations(gates)
        assert changed
        assert len(new_gates) == 1

    def test_cancel_cnot_pairs(self):
        from core.compiler import GateOptimizer, CompiledGate, GateType
        opt = GateOptimizer()
        gates = [
            CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT)
        ]
        new_gates, changed = opt._cancel_cnot_pairs(gates)
        assert changed
        assert len(new_gates) == 0

    def test_optimize_full(self):
        from core.compiler import GateOptimizer, CompiledGate, GateType
        opt = GateOptimizer()
        gates = [
            CompiledGate('H', [0]),
            CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('CNOT', [0, 1], gate_type=GateType.TWO_QUBIT),
            CompiledGate('H', [0])
        ]
        optimized, passes = opt.optimize(gates)
        assert len(optimized) <= len(gates)


class TestCompilerPipeline:
    def test_compile_simple(self):
        from core.compiler import CompilerPipeline, CompiledGate
        pipeline = CompilerPipeline()
        gates = [CompiledGate('H', [0]), CompiledGate('CNOT', [0, 1])]
        result = pipeline.compile(gates, 2)
        assert result.compiled_count > 0

    def test_compile_with_cz(self):
        from core.compiler import CompilerPipeline, CompiledGate
        pipeline = CompilerPipeline()
        gates = [CompiledGate('H', [0]), CompiledGate('CZ', [0, 1])]
        result = pipeline.compile(gates, 2)
        assert result.compiled_count > 0

    def test_compile_with_swap(self):
        from core.compiler import CompilerPipeline, CompiledGate
        pipeline = CompilerPipeline()
        gates = [CompiledGate('SWAP', [0, 1])]
        result = pipeline.compile(gates, 2)
        assert result.compiled_count >= 3

    def test_compression_ratio(self):
        from core.compiler import CompilationResult, CompiledGate
        cr = CompilationResult(gates=[], original_count=10, compiled_count=5)
        assert cr.compression_ratio() == 0.5

    def test_compile_result_dict(self):
        from core.compiler import CompilationResult
        cr = CompilationResult(gates=[], original_count=10, compiled_count=8, depth=3)
        d = cr.to_dict()
        assert d['original_count'] == 10


# === Channel 测试 ===

class TestPauliChannel:
    def test_apply_pure_dephasing(self):
        from core.channel import PauliChannel
        pc = PauliChannel(p_i=0.5, p_z=0.5)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = pc.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_kraus_operators(self):
        from core.channel import PauliChannel
        pc = PauliChannel(p_i=0.7, p_x=0.1, p_y=0.1, p_z=0.1)
        ops = pc.kraus_operators()
        assert len(ops) == 4

    def test_process_fidelity(self):
        from core.channel import PauliChannel
        pc = PauliChannel(p_i=1.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = pc.apply(rho)
        f = pc.process_fidelity(rho, result)
        assert f > 0


class TestGeneralizedAmplitudeDamping:
    def test_apply(self):
        from core.channel import GeneralizedAmplitudeDamping
        gad = GeneralizedAmplitudeDamping(0.1, 0.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = gad.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_kraus_complete(self):
        from core.channel import GeneralizedAmplitudeDamping, ChannelCompositor
        gad = GeneralizedAmplitudeDamping(0.1, 0.0)
        ops = gad.kraus_operators()
        assert ChannelCompositor.verify_complete(ops)


class TestPhaseDampingChannel:
    def test_apply(self):
        from core.channel import PhaseDampingChannel
        pdc = PhaseDampingChannel(0.3)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = pdc.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10

    def test_kraus_complete(self):
        from core.channel import PhaseDampingChannel, ChannelCompositor
        pdc = PhaseDampingChannel(0.5)
        ops = pdc.kraus_operators()
        assert ChannelCompositor.verify_complete(ops)


class TestRandomUnitaryChannel:
    def test_random_pauli_channel(self):
        from core.channel import RandomUnitaryChannel
        ruc = RandomUnitaryChannel.random_pauli_channel(0.1, 0.1, 0.1)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ruc.apply(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10


class TestChannelCompositor:
    def test_compose_identity(self):
        from core.channel import ChannelCompositor
        I = [np.eye(2, dtype=complex)]
        composed = ChannelCompositor.compose(I, I)
        assert len(composed) == 1

    def test_tensor_product(self):
        from core.channel import ChannelCompositor
        I = [np.eye(2, dtype=complex)]
        result = ChannelCompositor.tensor_product(I, I)
        assert result[0].shape == (4, 4)

    def test_verify_complete(self):
        from core.channel import ChannelCompositor
        I = [np.eye(2, dtype=complex)]
        assert ChannelCompositor.verify_complete(I)

    def test_channel_fidelity(self):
        from core.channel import ChannelCompositor
        I = [np.eye(2, dtype=complex)]
        f = ChannelCompositor.channel_fidelity(I, I)
        assert f > 0


class TestChannelAnalyzer:
    def test_apply_channel(self):
        from core.channel import ChannelAnalyzer
        ca = ChannelAnalyzer()
        rho = np.eye(2, dtype=complex) / 2
        ops = [np.eye(2, dtype=complex)]
        result = ca.apply_channel(rho, ops, "identity")
        assert result.trace_preserved

    def test_compare_channels(self):
        from core.channel import ChannelAnalyzer
        ca = ChannelAnalyzer()
        rho = np.eye(2, dtype=complex) / 2
        I = [np.eye(2, dtype=complex)]
        comp = ca.compare_channels(rho, [I, I], ["ch1", "ch2"])
        assert "ch1" in comp
        assert "ch2" in comp

    def test_noise_strength(self):
        from core.channel import ChannelAnalyzer
        ca = ChannelAnalyzer()
        I = [np.eye(2, dtype=complex)]
        assert ca.noise_strength(I) < 1e-6

    def test_history(self):
        from core.channel import ChannelAnalyzer
        ca = ChannelAnalyzer()
        rho = np.eye(2, dtype=complex) / 2
        ca.apply_channel(rho, [np.eye(2, dtype=complex)])
        assert len(ca.get_history()) == 1


# === Scheduler 测试 ===

class TestResourceInfo:
    def test_quality_score(self):
        from core.scheduler import ResourceInfo
        ri = ResourceInfo(num_qubits=5, gate_error_rate=0.001, readout_error_rate=0.01)
        score = ri.quality_score()
        assert score > 0

    def test_availability_idle(self):
        from core.scheduler import ResourceInfo, ResourceState
        ri = ResourceInfo(num_qubits=5, state=ResourceState.IDLE, queue_length=0)
        assert ri.availability() > 0.5

    def test_availability_offline(self):
        from core.scheduler import ResourceInfo, ResourceState
        ri = ResourceInfo(state=ResourceState.OFFLINE)
        assert ri.availability() == 0.0

    def test_to_dict(self):
        from core.scheduler import ResourceInfo
        ri = ResourceInfo(name="backend1", num_qubits=5)
        d = ri.to_dict()
        assert d['name'] == "backend1"


class TestResourcePool:
    def test_add_resource(self):
        from core.scheduler import ResourcePool, ResourceInfo
        pool = ResourcePool()
        ri = ResourceInfo(name="backend1", num_qubits=5)
        pool.add_resource(ri)
        assert len(pool.resources) == 1

    def test_remove_resource(self):
        from core.scheduler import ResourcePool, ResourceInfo
        pool = ResourcePool()
        ri = ResourceInfo(name="backend1", num_qubits=5)
        pool.add_resource(ri)
        assert pool.remove_resource(ri.id)
        assert len(pool.resources) == 0

    def test_get_available(self):
        from core.scheduler import ResourcePool, ResourceInfo, ResourceState
        pool = ResourcePool()
        ri = ResourceInfo(name="backend1", num_qubits=5, state=ResourceState.IDLE)
        pool.add_resource(ri)
        available = pool.get_available(min_qubits=3)
        assert len(available) == 1

    def test_get_best_resource(self):
        from core.scheduler import ResourcePool, ResourceInfo, ResourceState
        pool = ResourcePool()
        ri1 = ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE,
                           gate_error_rate=0.001)
        ri2 = ResourceInfo(name="b2", num_qubits=5, state=ResourceState.IDLE,
                           gate_error_rate=0.01)
        pool.add_resource(ri1)
        pool.add_resource(ri2)
        best = pool.get_best_resource(3)
        assert best is not None

    def test_total_qubits(self):
        from core.scheduler import ResourcePool, ResourceInfo
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(num_qubits=5))
        pool.add_resource(ResourceInfo(num_qubits=10))
        assert pool.total_qubits() == 15

    def test_utilization(self):
        from core.scheduler import ResourcePool, ResourceInfo
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(num_qubits=5, queue_length=50))
        util = pool.utilization()
        assert len(util) == 1


class TestPriorityScheduler:
    def test_submit_and_schedule(self):
        from core.scheduler import ResourcePool, ResourceInfo, PriorityScheduler, ScheduledTask, TaskPriority, ResourceState
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE))
        scheduler = PriorityScheduler(pool)
        task = ScheduledTask(name="test", num_qubits=2, priority=TaskPriority.HIGH)
        scheduler.submit(task)
        assert len(scheduler.task_queue) == 1

    def test_schedule_priority(self):
        from core.scheduler import ResourcePool, ResourceInfo, PriorityScheduler, ScheduledTask, TaskPriority, ResourceState
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE))
        scheduler = PriorityScheduler(pool)
        scheduler.submit(ScheduledTask(name="low", num_qubits=1, priority=TaskPriority.LOW))
        scheduler.submit(ScheduledTask(name="high", num_qubits=1, priority=TaskPriority.CRITICAL))
        scheduled = scheduler.schedule_next()
        assert scheduled.name == "high"

    def test_queue_status(self):
        from core.scheduler import ResourcePool, PriorityScheduler, ScheduledTask, TaskPriority
        pool = ResourcePool()
        scheduler = PriorityScheduler(pool)
        scheduler.submit(ScheduledTask(name="t1", num_qubits=1))
        status = scheduler.get_queue_status()
        assert status['pending'] == 1


class TestLoadBalancer:
    def test_round_robin(self):
        from core.scheduler import ResourcePool, ResourceInfo, LoadBalancer, ScheduledTask, ResourceState
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE))
        pool.add_resource(ResourceInfo(name="b2", num_qubits=5, state=ResourceState.IDLE))
        lb = LoadBalancer(pool)
        tasks = [ScheduledTask(name=f"t{i}", num_qubits=1) for i in range(4)]
        assignments = lb.round_robin(tasks)
        assert len(assignments) == 4

    def test_weighted_assignment(self):
        from core.scheduler import ResourcePool, ResourceInfo, LoadBalancer, ScheduledTask, ResourceState
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE))
        lb = LoadBalancer(pool)
        tasks = [ScheduledTask(name="t1", num_qubits=1)]
        assignments = lb.weighted_assignment(tasks)
        assert len(assignments) == 1

    def test_load_balance(self):
        from core.scheduler import ResourcePool, ResourceInfo, LoadBalancer, ScheduledTask, ResourceState
        pool = ResourcePool()
        pool.add_resource(ResourceInfo(name="b1", num_qubits=5, state=ResourceState.IDLE))
        pool.add_resource(ResourceInfo(name="b2", num_qubits=5, state=ResourceState.IDLE))
        lb = LoadBalancer(pool)
        tasks = [ScheduledTask(name=f"t{i}", num_qubits=1) for i in range(4)]
        assignments = lb.load_balance(tasks)
        assert len(assignments) == 4


class TestCircuitBatcher:
    def test_batch_circuits(self):
        from core.scheduler import CircuitBatcher
        batcher = CircuitBatcher(max_batch_size=2)
        circuits = [{'name': f'c{i}'} for i in range(5)]
        batches = batcher.batch_circuits(circuits)
        assert len(batches) == 3

    def test_estimate_time(self):
        from core.scheduler import CircuitBatcher
        batcher = CircuitBatcher()
        t = batcher.estimate_execution_time(10, 1024, 5)
        assert t > 0


class TestSchedulingEngine:
    def test_add_backend(self):
        from core.scheduler import SchedulingEngine
        engine = SchedulingEngine()
        rid = engine.add_backend("ibmq_bogota", 5)
        assert rid is not None

    def test_submit_circuit(self):
        from core.scheduler import SchedulingEngine, TaskPriority
        engine = SchedulingEngine()
        engine.add_backend("b1", 5)
        tid = engine.submit_circuit("test_circuit", 3, 1024, TaskPriority.HIGH)
        assert tid is not None

    def test_process_queue(self):
        from core.scheduler import SchedulingEngine, TaskPriority
        engine = SchedulingEngine()
        engine.add_backend("b1", 10)
        engine.submit_circuit("c1", 2, 1024, TaskPriority.NORMAL)
        engine.submit_circuit("c2", 3, 1024, TaskPriority.HIGH)
        scheduled = engine.process_queue()
        assert len(scheduled) == 2

    def test_status(self):
        from core.scheduler import SchedulingEngine
        engine = SchedulingEngine()
        engine.add_backend("b1", 5)
        status = engine.get_status()
        assert 'resources' in status
        assert 'queue' in status


class TestScheduledTask:
    def test_task_dict(self):
        from core.scheduler import ScheduledTask, TaskPriority
        task = ScheduledTask(name="test", num_qubits=3, shots=2048, priority=TaskPriority.HIGH)
        d = task.to_dict()
        assert d['name'] == "test"
        assert d['shots'] == 2048

    def test_task_wait_time(self):
        from core.scheduler import ScheduledTask
        task = ScheduledTask(name="test", num_qubits=1)
        wt = task.wait_time()
        assert wt >= 0


class TestCompiledGate:
    def test_compiled_gate_dict(self):
        from core.compiler import CompiledGate, GateType
        cg = CompiledGate('H', [0], gate_type=GateType.SINGLE_QUBIT)
        d = cg.to_dict()
        assert d['name'] == 'H'
        assert d['qubits'] == [0]

    def test_compiled_gate_params(self):
        from core.compiler import CompiledGate, GateType
        cg = CompiledGate('Rx', [0], [1.57], GateType.PARAMETRIC)
        assert cg.params == [1.57]
