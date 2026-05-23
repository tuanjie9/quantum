# 🌀 Quantum — 量子计算模拟与优化蜂群

> 10 Agent × 7 层 DAG × 量子态空间 | 面向量子计算研究与优化的蜂群智能系统

## 📊 项目概览

| 指标 | 数值 |
|------|:---|
| **代码行数** | 10,400+ |
| **测试用例** | 337 全绿 ✓ |
| **Agent 数量** | 10 |
| **DAG 层数** | 7 |
| **Python 文件** | 28 |
| **核心模块** | 16 |

## 🏗️ 架构设计

```
感知 → 建图 → 识别 → 定位 → 规划 → 执行 → 评估
  ↓      ↓      ↓      ↓      ↓      ↓      ↓
量子态  纠缠   门操作  算法   VQE    编译   层析
测量   建图   通道   优化   QML    调度   评估
```

### 10 Agent 蜂群

| Agent | 职责 |
|-------|------|
| State Simulator | 量子态模拟与演化 |
| Gate Operator | 量子门操作与组合 |
| Circuit Builder | 量子电路构建与优化 |
| Algorithm Runner | 量子算法执行 |
| Error Corrector | 量子纠错码实现 |
| Noise Analyzer | 噪声通道建模与分析 |
| VQE Optimizer | 变分量子特征求解器 |
| QML Trainer | 量子机器学习训练 |
| Compiler | 量子编译与路由 |
| Fusion Coordinator | 多任务融合调度 |

## 🔬 核心模块

### 量子基础 (`core/`)

| 模块 | 功能 | 亮点 |
|------|------|------|
| `quantum_state.py` | 量子态表示 | 密度矩阵、Bloch球、态向量 |
| `quantum_gates.py` | 量子门库 | 20+ 标准门 + 自定义门 |
| `circuit.py` | 量子电路 | 构建、优化、模拟 |
| `entanglement.py` | 纠缠理论 | 并发度、负性、Schmidt分解 |
| `noise.py` | 噪声模型 | 退相干、去极化、振幅阻尼 |
| `error_correction.py` | 纠错码 | Shor、Steane、Surface码 |

### 量子算法 (`core/`)

| 模块 | 功能 | 算法 |
|------|------|------|
| `algorithms.py` | 量子算法库 | Grover、Shor、QFT、QPE、HHL |
| `optimization.py` | 变分优化 | VQE、QAOA、VQLS |
| `simulation.py` | 系统仿真 | 薛定谔方程、Lindblad、绝热演化 |
| `tomography.py` | 量子态层析 | MLE、贝叶斯、压缩感知 |

### 工程模块 (`core/`)

| 模块 | 功能 |
|------|------|
| `channel.py` | 量子通道 — Pauli、振幅阻尼、相位阻尼 |
| `compiler.py` | 量子编译器 — 门分解、路由、优化 |
| `scheduler.py` | 任务调度 — 资源池、优先级、负载均衡 |
| `pipeline.py` | 7层DAG管道 |
| `engine.py` | Agent引擎核心 |

## 🚀 快速开始

```bash
# 安装依赖
pip install numpy

# 运行测试
python3 -m pytest tests/ -v

# CLI 使用
python3 -m cli.main status
python3 -m cli.main agents
python3 -m cli.main circuit --qubits 3
python3 -m cli.main benchmark
```

## 📐 项目结构

```
quantum/
├── meta.py              # 项目元数据
├── config/
│   └── settings.py      # 量子配置（精度、噪声、优化器）
├── core/
│   ├── quantum_state.py # 量子态 — 密度矩阵/态向量/Bloch球
│   ├── quantum_gates.py # 量子门 — 20+标准门 + 自定义
│   ├── circuit.py       # 量子电路构建与优化
│   ├── algorithms.py    # 量子算法（Grover/Shor/QFT/QPE）
│   ├── entanglement.py  # 纠缠理论（并发度/负性/Schmidt）
│   ├── noise.py         # 噪声通道（退相干/去极化）
│   ├── error_correction.py # 量子纠错（Shor/Steane/Surface）
│   ├── optimization.py  # 变分优化（VQE/QAOA/VQLS）
│   ├── simulation.py    # 系统仿真（薛定谔/Lindblad/绝热）
│   ├── tomography.py    # 量子态层析（MLE/贝叶斯）
│   ├── channel.py       # 量子通道（Pauli/振幅阻尼）
│   ├── compiler.py      # 量子编译器（门分解/路由）
│   ├── scheduler.py     # 任务调度（资源池/负载均衡）
│   ├── pipeline.py      # 7层DAG管道
│   └── engine.py        # Agent引擎核心
├── agents/
│   └── pool.py          # 10个Agent实现
├── api/
│   └── server.py        # REST API（25+端点）
├── cli/
│   └── main.py          # CLI工具（15+命令）
├── dashboard/
│   └── router.py        # Web仪表板（12页）
└── tests/
    └── test_core.py     # 337个测试用例
```

## 🧪 测试覆盖

```bash
$ python3 -m pytest tests/ -v
# 337 passed in 0.50s ✓
```

覆盖模块：量子态、量子门、电路、算法、纠缠、噪声、纠错、VQE、QML、层析、编译、调度、API、CLI、Dashboard、集成测试

## 📜 License

MIT License
