"""Web 仪表板 - 8 页"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import time

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _page(title: str, content: str) -> HTMLResponse:
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
    <style>body{{font-family:system-ui;background:#0a0a0f;color:#e0e0e0;padding:20px}}
    h1{{color:#8b5cf6}}h2{{color:#a78bfa}}.card{{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;margin:10px 0}}
    .stat{{font-size:32px;font-weight:700;color:#8b5cf6}}table{{width:100%;border-collapse:collapse}}
    th,td{{padding:8px;text-align:left;border-bottom:1px solid #333}}th{{color:#a78bfa}}
    .success{{color:#4ade80}}.error{{color:#f85149}}nav a{{color:#a78bfa;margin:0 10px}}</style></head>
    <body><nav><a href="/dashboard/">概览</a><a href="/dashboard/circuits">电路</a>
    <a href="/dashboard/simulation">模拟</a><a href="/dashboard/optimization">优化</a>
    <a href="/dashboard/agents">Agent</a><a href="/dashboard/noise">噪声</a>
    <a href="/dashboard/entanglement">纠缠</a><a href="/dashboard/metrics">指标</a></nav>
    <hr>{content}</body></html>"""
    return HTMLResponse(html)


@router.get("/", response_class=HTMLResponse)
async def overview():
    content = """<h1>🐝 Quantum Engine</h1><p>AI 量子计算模拟与优化蜂群引擎</p>
    <div class="card"><span class="stat">10</span><p>Agent 蜂群</p></div>
    <div class="card"><span class="stat">7</span><p>DAG 管道层</p></div>
    <div class="card"><span class="stat">10</span><p>量子维度</p></div>
    <div class="card"><span class="stat">20+</span><p>量子门</p></div>
    <div class="card"><span class="stat">10</span><p>量子算法</p></div>"""
    return _page("Quantum - 概览", content)


@router.get("/circuits", response_class=HTMLResponse)
async def circuits():
    content = """<h1>量子电路</h1>
    <div class="card"><h2>电路设计器</h2><p>支持 H, X, Y, Z, CNOT, CZ, SWAP, Toffoli, Fredkin 等 20+ 量子门</p>
    <p>电路深度、宽度、门数量自动计算</p></div>
    <div class="card"><h2>电路优化</h2><p>门消除、门合并、电路重写</p></div>"""
    return _page("Quantum - 电路", content)


@router.get("/simulation", response_class=HTMLResponse)
async def simulation():
    content = """<h1>量子模拟</h1>
    <div class="card"><h2>状态矢量模拟</h2><p>精确量子态演化，支持 20+ 量子比特</p></div>
    <div class="card"><h2>密度矩阵模拟</h2><p>混合态模拟，支持噪声通道</p></div>
    <div class="card"><h2>量子算法</h2>
    <table><tr><th>算法</th><th>描述</th><th>复杂度</th></tr>
    <tr><td>Grover</td><td>无序搜索</td><td>O(√N)</td></tr>
    <tr><td>QFT</td><td>量子傅里叶变换</td><td>O(n²)</td></tr>
    <tr><td>QPE</td><td>相位估计</td><td>O(n)</td></tr>
    <tr><td>Teleportation</td><td>量子隐形传态</td><td>O(1)</td></tr>
    <tr><td>BB84</td><td>量子密钥分发</td><td>O(n)</td></tr></table></div>"""
    return _page("Quantum - 模拟", content)


@router.get("/optimization", response_class=HTMLResponse)
async def optimization():
    content = """<h1>量子优化</h1>
    <div class="card"><h2>VQE</h2><p>变分量子特征值求解器 - 求解基态能量</p></div>
    <div class="card"><h2>QAOA</h2><p>量子近似优化算法 - 组合优化</p></div>
    <div class="card"><h2>VQLS</h2><p>变分量子线性方程组求解器</p></div>
    <div class="card"><h2>优化器</h2>
    <table><tr><th>优化器</th><th>类型</th><th>特点</th></tr>
    <tr><td>COBYLA</td><td>无梯度</td><td>稳健</td></tr>
    <tr><td>SPSA</td><td>随机梯度</td><td>噪声友好</td></tr>
    <tr><td>Adam</td><td>自适应梯度</td><td>快速收敛</td></tr>
    <tr><td>自然梯度</td><td>量子梯度</td><td>几何最优</td></tr></table></div>"""
    return _page("Quantum - 优化", content)


@router.get("/agents", response_class=HTMLResponse)
async def agents_page():
    content = """<h1>Agent 蜂群</h1>
    <table><tr><th>Agent</th><th>角色</th><th>职责</th></tr>
    <tr><td>CircuitArchitect</td><td>电路架构师</td><td>设计量子电路</td></tr>
    <tr><td>GateSmith</td><td>门工匠</td><td>合成最优门序列</td></tr>
    <tr><td>StatePreparer</td><td>态制备师</td><td>准备目标量子态</td></tr>
    <tr><td>QuantumEvolver</td><td>量子演化师</td><td>时间演化模拟</td></tr>
    <tr><td>ErrorCorrector</td><td>纠错师</td><td>量子纠错编码</td></tr>
    <tr><td>MeasurementEngine</td><td>测量引擎</td><td>测量与层析</td></tr>
    <tr><td>OptimizationPilot</td><td>优化导航</td><td>变分优化</td></tr>
    <tr><td>NoiseModeler</td><td>噪声建模</td><td>噪声特征化</td></tr>
    <tr><td>EntanglementAnalyzer</td><td>纠缠分析</td><td>纠缠度量</td></tr>
    <tr><td>QuantumOracle</td><td>量子预言机</td><td>算法Oracle构造</td></tr></table>"""
    return _page("Quantum - Agent", content)


@router.get("/noise", response_class=HTMLResponse)
async def noise_page():
    content = """<h1>噪声模型</h1>
    <div class="card"><h2>噪声通道</h2>
    <table><tr><th>通道</th><th>参数</th><th>描述</th></tr>
    <tr><td>去极化</td><td>p</td><td>等概率施加 Pauli 错误</td></tr>
    <tr><td>振幅阻尼</td><td>γ</td><td>能量弛豫到基态</td></tr>
    <tr><td>相位阻尼</td><td>γ</td><td>相干性损失</td></tr>
    <tr><td>热弛豫</td><td>T1, T2</td><td>有限温度弛豫</td></tr>
    <tr><td>读出误差</td><td>p0→1, p1→0</td><td>测量误判</td></tr>
    <tr><td>串扰</td><td>强度</td><td>相邻量子比特干扰</td></tr>
    <tr><td>相干误差</td><td>角度</td><td>过/欠旋转</td></tr></table></div>"""
    return _page("Quantum - 噪声", content)


@router.get("/entanglement", response_class=HTMLResponse)
async def entanglement_page():
    content = """<h1>纠缠分析</h1>
    <div class="card"><h2>纠缠度量</h2>
    <table><tr><th>度量</th><th>范围</th><th>描述</th></tr>
    <tr><td>并发度</td><td>[0, 1]</td><td>两量子比特纠缠度</td></tr>
    <tr><td>负性</td><td>[0, ∞)</td><td>部分转置负特征值</td></tr>
    <tr><td>纠缠熵</td><td>[0, n]</td><td>子系统冯诺依曼熵</td></tr>
    <tr><td>Bell保真度</td><td>[0, 1]</td><td>与Bell态的保真度</td></tr>
    <tr><td>互信息</td><td>[0, 2n]</td><td>总关联度量</td></tr>
    <tr><td>Schmidt秩</td><td>[1, d]</td><td>纠缠自由度数</td></tr></table></div>
    <div class="card"><h2>Bell 不等式</h2><p>CHSH 不等式: S ≤ 2 (经典) vs S ≤ 2√2 (量子)</p></div>"""
    return _page("Quantum - 纠缠", content)


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    content = """<h1>系统指标</h1>
    <div class="card"><h2>项目信息</h2>
    <table><tr><td>版本</td><td>0.1.0</td></tr>
    <tr><td>Agent 数量</td><td>10</td></tr>
    <tr><td>DAG 层数</td><td>7</td></tr>
    <tr><td>量子维度</td><td>10</td></tr>
    <tr><td>量子门</td><td>20+</td></tr>
    <tr><td>量子算法</td><td>10</td></tr>
    <tr><td>噪声通道</td><td>7</td></tr>
    <tr><td>纠错码</td><td>4</td></tr>
    <tr><td>优化器</td><td>7</td></tr></table></div>"""
    return _page("Quantum - 指标", content)



@router.get("/algorithms", response_class=HTMLResponse)
async def algorithms_page():
    content = """<h1>量子算法库</h1>
    <div class="card"><h2>搜索算法</h2>
    <table><tr><th>算法</th><th>复杂度</th><th>描述</th></tr>
    <tr><td>Grover</td><td>O(√N)</td><td>无序数据库搜索</td></tr>
    <tr><td>Deutsch-Jozsa</td><td>O(1)</td><td>判断函数是否恒定</td></tr>
    <tr><td>Bernstein-Vazirani</td><td>O(1)</td><td>查找隐藏比特串</td></tr>
    <tr><td>Simon</td><td>O(n)</td><td>查找隐藏周期</td></tr></table></div>
    <div class="card"><h2>变换算法</h2>
    <table><tr><th>算法</th><th>复杂度</th><th>描述</th></tr>
    <tr><td>QFT</td><td>O(n²)</td><td>量子傅里叶变换</td></tr>
    <tr><td>QPE</td><td>O(n)</td><td>量子相位估计</td></tr>
    <tr><td>量子游走</td><td>O(t)</td><td>量子随机游走</td></tr></table></div>
    <div class="card"><h2>通信协议</h2>
    <table><tr><th>协议</th><th>描述</th></tr>
    <tr><td>量子隐形传态</td><td>传输量子态</td></tr>
    <tr><td>超密编码</td><td>1量子比特传2经典比特</td></tr>
    <tr><td>BB84 QKD</td><td>量子密钥分发</td></tr></table></div>"""
    return _page("Quantum - 算法库", content)


@router.get("/error-correction", response_class=HTMLResponse)
async def error_correction_page():
    content = """<h1>量子纠错</h1>
    <div class="card"><h2>纠错码</h2>
    <table><tr><th>码</th><th>距离</th><th>数据比特</th><th>描述</th></tr>
    <tr><td>重复码</td><td>3,5,7</td><td>d</td><td>最简单的纠错码</td></tr>
    <tr><td>Steane码</td><td>3</td><td>7</td><td>[[7,1,3]] CSS码</td></tr>
    <tr><td>Shor码</td><td>3</td><td>9</td><td>[[9,1,3]] 码</td></tr>
    <tr><td>Surface码</td><td>3,5,7</td><td>d²</td><td>拓扑纠错码</td></tr></table></div>
    <div class="card"><h2>噪声通道</h2>
    <table><tr><th>通道</th><th>Kraus算子</th><th>描述</th></tr>
    <tr><td>去极化</td><td>4</td><td>等概率Pauli错误</td></tr>
    <tr><td>振幅阻尼</td><td>2</td><td>能量弛豫</td></tr>
    <tr><td>相位阻尼</td><td>2</td><td>相干性损失</td></tr>
    <tr><td>热弛豫</td><td>-</td><td>有限温度效应</td></tr></table></div>"""
    return _page("Quantum - 纠错", content)


@router.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page():
    content = """<h1>7层 DAG 管道</h1>
    <div class="card"><h2>管道层</h2>
    <table><tr><th>层</th><th>名称</th><th>描述</th></tr>
    <tr><td>1</td><td>电路设计</td><td>根据问题设计量子电路</td></tr>
    <tr><td>2</td><td>门合成</td><td>合成最优门序列</td></tr>
    <tr><td>3</td><td>态制备</td><td>制备初始量子态</td></tr>
    <tr><td>4</td><td>量子演化</td><td>模拟量子态时间演化</td></tr>
    <tr><td>5</td><td>纠错</td><td>应用量子纠错编码</td></tr>
    <tr><td>6</td><td>测量</td><td>量子测量与结果提取</td></tr>
    <tr><td>7</td><td>优化</td><td>变分优化与参数调优</td></tr></table></div>
    <div class="card"><h2>依赖关系</h2>
    <pre>Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → Layer 6 → Layer 7</pre>
    <p>每层依赖前一层的输出，形成有向无环图 (DAG)</p></div>"""
    return _page("Quantum - 管道", content)


@router.get("/config", response_class=HTMLResponse)
async def config_page():
    content = """<h1>系统配置</h1>
    <div class="card"><h2>配置预设</h2>
    <table><tr><th>预设</th><th>描述</th><th>适用场景</th></tr>
    <tr><td>high_precision</td><td>高精度模拟</td><td>精确计算</td></tr>
    <tr><td>noisy_hardware</td><td>真实硬件噪声</td><td>NISQ设备</td></tr>
    <tr><td>fast_prototype</td><td>快速原型</td><td>开发调试</td></tr>
    <tr><td>quantum_advantage</td><td>量子优势</td><td>大规模计算</td></tr></table></div>
    <div class="card"><h2>配置参数</h2>
    <table><tr><th>参数</th><th>默认值</th><th>范围</th></tr>
    <tr><td>max_qubits</td><td>20</td><td>1-30</td></tr>
    <tr><td>shots</td><td>1024</td><td>1-∞</td></tr>
    <tr><td>precision</td><td>1e-10</td><td>0-1</td></tr>
    <tr><td>t1_time</td><td>50μs</td><td>0-∞</td></tr>
    <tr><td>t2_time</td><td>70μs</td><td>0-∞</td></tr></table></div>"""
    return _page("Quantum - 配置", content)
