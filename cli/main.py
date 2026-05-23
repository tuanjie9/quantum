"""CLI 工具 - 12 命令"""
import click
import json
import numpy as np
import time


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Quantum - AI 量子计算模拟与优化蜂群引擎"""
    pass


@cli.command()
@click.option('--num-qubits', '-n', default=3, help='量子比特数')
@click.option('--shots', '-s', default=1024, help='测量次数')
@click.option('--state-type', '-t', default='zero', type=click.Choice(['zero', 'plus', 'ghz', 'w', 'bell', 'random']))
def simulate(num_qubits, shots, state_type):
    """模拟量子电路"""
    from core.quantum_state import QuantumStateVector
    from core.circuit import QuantumCircuit
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.h(i)
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    result = circuit.simulate(shots=shots)
    click.echo(f"电路: {num_qubits} 量子比特, {circuit.gate_count} 门, 深度 {circuit.depth}")
    click.echo(f"测量结果 (前5):")
    sorted_counts = sorted(result['counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    for bitstring, count in sorted_counts:
        click.echo(f"  |{bitstring}⟩: {count} ({count/shots*100:.1f}%)")


@cli.command()
@click.option('--n', default=4, help='量子比特数')
@click.option('--target', default=0, help='目标状态')
def grover(n, target):
    """运行 Grover 搜索"""
    from core.algorithms import GroverSearch
    grover = GroverSearch(n, target)
    result = grover.run()
    click.echo(f"Grover 搜索: {n} 量子比特, 目标 {format(target, f'0{n}b')}")
    click.echo(f"最优迭代次数: {grover.optimal_iterations()}")
    click.echo(f"找到: {result.output}, 成功概率: {result.probability:.4f}")


@cli.command()
@click.option('--n', default=4, help='量子比特数')
def qft(n):
    """量子傅里叶变换"""
    from core.algorithms import QuantumFourierTransform
    qft = QuantumFourierTransform(n)
    click.echo(f"QFT: {n} 量子比特")
    click.echo(f"门数量: {qft.gate_count()}")
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    result = qft.apply(state)
    probs = np.abs(result)**2
    click.echo(f"输出态概率分布 (前5):")
    top = np.argsort(probs)[-5:][::-1]
    for idx in top:
        click.echo(f"  |{format(idx, f'0{n}b')}⟩: {probs[idx]:.4f}")


@cli.command()
@click.option('--code-type', '-c', default='repetition', type=click.Choice(['repetition', 'steane', 'shor', 'surface']))
@click.option('--distance', '-d', default=3, help='码距离')
@click.option('--physical-rate', '-p', default=0.01, help='物理错误率')
def error_correction(code_type, distance, physical_rate):
    """量子纠错分析"""
    from core.error_correction import ErrorCorrectionEngine
    engine = ErrorCorrectionEngine(code_type, distance)
    params = engine.get_code_params()
    logical_rate = engine.logical_error_rate(physical_rate)
    click.echo(f"纠错码: {code_type}, 距离: {distance}")
    click.echo(f"数据比特: {params['data_qubits']}, 总比特: {params['total_qubits']}")
    click.echo(f"物理错误率: {physical_rate:.6f}")
    click.echo(f"逻辑错误率: {logical_rate:.10f}")
    click.echo(f"改善因子: {physical_rate / max(logical_rate, 1e-15):.2f}x")


@cli.command()
@click.option('--method', '-m', default='cobyla', type=click.Choice(['cobyla', 'adam', 'spsa']))
@click.option('--num-params', '-n', default=4, help='参数数量')
@click.option('--max-iter', default=100, help='最大迭代次数')
def optimize(method, num_params, max_iter):
    """运行优化"""
    from core.optimization import OptimizationEngine
    opt = OptimizationEngine(method, max_iter)
    target = np.ones(num_params) * 0.5
    def loss(x): return float(np.linalg.norm(x - target))
    x0 = np.random.randn(num_params)
    start = time.time()
    result = opt.minimize(loss, x0)
    elapsed = time.time() - start
    click.echo(f"优化器: {method}, 参数: {num_params}")
    click.echo(f"最优值: {result.optimal_value:.8f}")
    click.echo(f"迭代次数: {result.iterations}")
    click.echo(f"收敛: {result.converged}")
    click.echo(f"耗时: {elapsed:.3f}s")


@cli.command()
@click.option('--model-type', '-t', default='depolarizing')
@click.option('--single-error', default=0.001)
@click.option('--two-error', default=0.01)
def noise(model_type, single_error, two_error):
    """噪声分析"""
    from core.noise import NoiseModel, NoiseModelType, NoiseParameters
    params = NoiseParameters(single_qubit_error=single_error, two_qubit_error=two_error)
    try:
        model = NoiseModel(NoiseModelType(model_type), params)
    except ValueError:
        click.echo(f"无效模型: {model_type}")
        return
    click.echo(f"噪声模型: {model_type}")
    for k, v in model.get_noise_level().items():
        click.echo(f"  {k}: {v:.6f}")


@cli.command()
@click.option('--state', '-s', default='bell', type=click.Choice(['bell', 'ghz', 'w', 'random']))
@click.option('--num-qubits', '-n', default=2)
def entangle(state, num_qubits):
    """纠缠分析"""
    from core.quantum_state import QuantumStateVector
    from core.entanglement import EntanglementAnalyzer
    if state == 'bell':
        sv = QuantumStateVector.bell_state(0)
    elif state == 'ghz':
        sv = QuantumStateVector.ghz_state(num_qubits)
    elif state == 'w':
        sv = QuantumStateVector.w_state(num_qubits)
    else:
        sv = QuantumStateVector.random_state(num_qubits)
    analyzer = EntanglementAnalyzer()
    measure = analyzer.analyze_state_vector(sv.amplitudes, sv.num_qubits)
    click.echo(f"状态类型: {state}, 量子比特: {sv.num_qubits}")
    click.echo(f"并发度: {measure.concurrence:.4f}")
    click.echo(f"负性: {measure.negativity:.4f}")
    click.echo(f"纠缠熵: {measure.entanglement_entropy:.4f}")
    click.echo(f"Bell 保真度: {measure.bell_fidelity:.4f}")
    click.echo(f"纠缠分类: {measure.entanglement_class()}")


@cli.command()
@click.option('--key-length', '-k', default=16)
@click.option('--eavesdrop/--no-eavesdrop', default=False)
def qkd(key_length, eavesdrop):
    """BB84 量子密钥分发"""
    from core.algorithms import QuantumKeyDistribution
    protocol = QuantumKeyDistribution(key_length)
    result = protocol.run(eavesdrop=eavesdrop)
    data = result.output
    click.echo(f"BB84 QKD: 密钥长度 {key_length}")
    click.echo(f"筛选后密钥长度: {data['key_length']}")
    click.echo(f"QBER: {data['qber']:.4f}")
    click.echo(f"窃听: {'是' if eavesdrop else '否'}")
    click.echo(f"安全: {'是' if result.success else '否'}")
    if data['key_length'] <= 8:
        click.echo(f"密钥: {''.join(map(str, data['key']))}")


@cli.command()
@click.option('--message', '-m', default='01', type=click.Choice(['00', '01', '10', '11']))
def superdense(message):
    """超密编码"""
    from core.algorithms import SuperdenseCoding
    sd = SuperdenseCoding()
    result = sd.run(message)
    data = result.output
    click.echo(f"超密编码: 发送 {data['original']}, 解码 {data['decoded']}")
    click.echo(f"成功: {result.success}")


@cli.command()
@click.option('--steps', '-s', default=10)
def walk(steps):
    """量子随机游走"""
    from core.algorithms import QuantumWalk
    qw = QuantumWalk(steps)
    result = qw.run_line_walk()
    data = result.output
    positions = data['positions']
    probs = data['probabilities']
    max_prob_idx = np.argmax(probs)
    click.echo(f"量子游走: {steps} 步")
    click.echo(f"最高概率位置: {positions[max_prob_idx]}, 概率: {probs[max_prob_idx]:.4f}")


@cli.command()
def agents():
    """列出所有 Agent"""
    from agents.pool import create_all_agents
    for agent in create_all_agents():
        m = agent.get_metrics()
        click.echo(f"  {m['name']} ({m['role']}): tasks={m['tasks_completed']}, fitness={m['fitness']:.2f}")


@cli.command()
@click.option('--num-qubits', '-n', default=3)
@click.option('--circuit-type', '-t', default='generic')
@click.option('--shots', '-s', default=1024)
def benchmark(num_qubits, circuit_type, shots):
    """基准测试"""
    from core.pipeline import QuantumPipeline
    pipeline = QuantumPipeline()
    start = time.time()
    result = pipeline.execute({'num_qubits': num_qubits, 'circuit_type': circuit_type, 'shots': shots})
    elapsed = time.time() - start
    click.echo(f"基准测试: {num_qubits} 量子比特, {circuit_type}")
    click.echo(f"通过层: {result['metrics'].layers_passed}/7")
    click.echo(f"总耗时: {elapsed:.4f}s")
    click.echo(f"管道指标: {json.dumps(pipeline.get_metrics(), indent=2)}")



@cli.command()
def benchmark_all():
    """运行完整基准测试"""
    from core.engine import QuantumBenchmark
    bench = QuantumBenchmark()
    result = bench.run_full_benchmark()
    click.echo(f"基准测试完成: {result['total_tests']} 项测试")
    click.echo(f"总耗时: {result['total_time']:.4f}s")
    for t in result['tests']:
        click.echo(f"  {t['test']}: {t.get('num_qubits', '?')} qubits - {t['time']:.4f}s")


@cli.command()
@click.option('--num-qubits', '-n', default=3)
@click.option('--state-type', '-t', default='zero')
def state_info(num_qubits, state_type):
    """显示量子态信息"""
    from core.quantum_state import QuantumStateVector, QuantumStateFactory
    sv = QuantumStateFactory.create(state_type, num_qubits=num_qubits)
    click.echo(f"量子态: {state_type}, {num_qubits} 量子比特")
    click.echo(f"维度: {2**num_qubits}")
    click.echo(f"熵: {sv.entropy():.4f}")
    click.echo(f"范数: {np.linalg.norm(sv.amplitudes):.6f}")
    probs = sv.probabilities()
    top5 = np.argsort(probs)[-5:][::-1]
    click.echo("概率分布 (前5):")
    for idx in top5:
        if probs[idx] > 1e-6:
            click.echo(f"  |{format(idx, f'0{num_qubits}b')}⟩: {probs[idx]:.4f}")


@cli.command()
@click.option('--code-type', '-c', default='steane')
@click.option('--distance', '-d', default=3)
def error_budget(code_type, distance):
    """误差预算分析"""
    from core.noise import QuantumErrorBudget
    budget = QuantumErrorBudget(total_budget=0.01)
    budget.allocate('gate_error', 0.003)
    budget.allocate('readout_error', 0.002)
    budget.allocate('decoherence', 0.002)
    budget.allocate('crosstalk', 0.001)
    report = budget.get_report()
    click.echo(f"误差预算报告:")
    click.echo(f"  总预算: {report['total_budget']:.4f}")
    click.echo(f"  已分配: {report['allocated']:.4f}")
    click.echo(f"  剩余: {report['remaining']:.4f}")
    click.echo(f"  利用率: {report['utilization']:.1%}")
    for comp, val in report['components'].items():
        click.echo(f"    {comp}: {val:.4f}")


@cli.command()
def version():
    """显示版本信息"""
    click.echo("Quantum Engine v0.1.0")
    click.echo("AI 量子计算模拟与优化蜂群引擎")
    click.echo("10 Agent × 7 DAG × 10 维量子态")
