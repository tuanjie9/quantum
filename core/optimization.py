"""变分量子优化器 - VQE/QAOA/VQLS"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple, Dict, Any
from enum import Enum
import math


class OptimizerMethod(Enum):
    """优化方法"""
    COBYLA = "cobyla"
    SPSA = "spsa"
    ADAM = "adam"
    NATURAL_GRADIENT = "natural_gradient"
    L_BFGS = "l_bfgs"
    NELDER_MEAD = "nelder_mead"
    POWELL = "powell"
    GRADIENT_DESCENT = "gradient_descent"
    PARAMETER_SHIFT = "parameter_shift"


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_params: np.ndarray
    optimal_value: float
    iterations: int
    history: List[float]
    converged: bool
    execution_time: float = 0.0
    gradient_norms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimal_params': self.optimal_params.tolist(),
            'optimal_value': float(self.optimal_value),
            'iterations': self.iterations,
            'converged': self.converged,
            'final_loss': float(self.history[-1]) if self.history else None,
            'history_length': len(self.history)
        }


class AdamOptimizer:
    """Adam 优化器"""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Adam 更新步骤"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class SPSAOptimizer:
    """同步扰动随机逼近优化器"""

    def __init__(self, a: float = 0.1, c: float = 0.1, alpha: float = 0.602, gamma: float = 0.101, A: float = 10):
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.k = 0

    def step(self, params: np.ndarray, loss_fn: Callable) -> Tuple[np.ndarray, float]:
        """SPSA 更新步骤"""
        self.k += 1
        ak = self.a / (self.k + self.A)**self.alpha
        ck = self.c / self.k**self.gamma
        delta = 2 * np.random.randint(0, 2, size=params.shape) - 1
        loss_plus = loss_fn(params + ck * delta)
        loss_minus = loss_fn(params - ck * delta)
        g_hat = (loss_plus - loss_minus) / (2 * ck * delta)
        new_params = params - ak * g_hat
        return new_params, (loss_plus + loss_minus) / 2


class COBYLAOptimizer:
    """COBYLA 优化器（无梯度）"""

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8, rho_end: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.rho_end = rho_end

    def minimize(self, loss_fn: Callable, x0: np.ndarray) -> OptimizationResult:
        """COBYLA 最小化"""
        n = len(x0)
        x = x0.copy()
        rho = 0.5
        history = []
        gradient_norms = []
        for iteration in range(self.max_iter):
            loss = loss_fn(x)
            history.append(float(loss))
            if rho < self.rho_end:
                return OptimizationResult(x, float(loss), iteration + 1, history, True)
            # 简化 COBYLA: 使用坐标下降
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += rho
                x_minus = x.copy()
                x_minus[i] -= rho
                loss_plus = loss_fn(x_plus)
                loss_minus = loss_fn(x_minus)
                grad_i = (loss_plus - loss_minus) / (2 * rho)
                gradient_norms.append(abs(float(grad_i)))
                if loss_plus < loss:
                    x = x_plus
                    loss = loss_plus
                elif loss_minus < loss:
                    x = x_minus
                    loss = loss_minus
            rho *= 0.95
        return OptimizationResult(x, float(loss), self.max_iter, history, False, gradient_norms=gradient_norms)


class ParameterShiftRule:
    """参数偏移规则 - 量子梯度计算"""

    def __init__(self, shift: float = np.pi / 2):
        self.shift = shift

    def compute_gradient(self, params: np.ndarray, loss_fn: Callable, param_index: int) -> float:
        """计算单个参数的梯度"""
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[param_index] += self.shift
        params_minus[param_index] -= self.shift
        return (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * np.sin(self.shift))

    def compute_full_gradient(self, params: np.ndarray, loss_fn: Callable) -> np.ndarray:
        """计算完整梯度向量"""
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            gradient[i] = self.compute_gradient(params, loss_fn, i)
        return gradient


class VariationalQuantumEigensolver:
    """变分量子特征值求解器 (VQE)"""

    def __init__(self, ansatz: Callable, hamiltonian: np.ndarray, optimizer: str = "cobyla"):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer_name = optimizer
        self.history: List[float] = []

    def cost_function(self, params: np.ndarray) -> float:
        """代价函数: ⟨ψ(θ)|H|ψ(θ)⟩"""
        psi = self.ansatz(params)
        return float(np.real(psi.conj() @ self.hamiltonian @ psi))

    def solve(self, initial_params: np.ndarray, max_iter: int = 1000) -> OptimizationResult:
        """求解基态能量"""
        optimizer = COBYLAOptimizer(max_iter=max_iter)
        result = optimizer.minimize(self.cost_function, initial_params)
        result.history = [self.cost_function(p) for p in [result.optimal_params]]
        # 精确计算最优值
        result.optimal_value = self.cost_function(result.optimal_params)
        return result

    def ground_state_energy(self, params: np.ndarray) -> float:
        """计算基态能量"""
        return self.cost_function(params)

    def excited_state_energy(self, params: np.ndarray, ground_energy: float, penalty: float = 100.0) -> float:
        """计算激发态能量（带惩罚项）"""
        energy = self.cost_function(params)
        penalty_term = penalty * max(0, ground_energy - energy + 0.1)**2
        return energy + penalty_term


class QAOA:
    """量子近似优化算法 (QAOA)"""

    def __init__(self, cost_hamiltonian: np.ndarray, mixer_hamiltonian: np.ndarray, p: int = 1):
        self.cost_hamiltonian = cost_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian
        self.p = p  # QAOA 深度
        self.n_qubits = int(np.log2(cost_hamiltonian.shape[0]))

    def ansatz(self, params: np.ndarray) -> np.ndarray:
        """QAOA 线路"""
        betas = params[:self.p]
        gammas = params[self.p:]
        n = 2**self.n_qubits
        state = np.ones(n, dtype=complex) / np.sqrt(n)  # 均匀叠加态
        for i in range(self.p):
            # 应用成本哈密顿量
            phase_matrix = np.diag(np.exp(-1j * gammas[i] * np.diag(self.cost_hamiltonian)))
            state = phase_matrix @ state
            # 应用混合哈密顿量
            mixer_exp = self._matrix_exp(-1j * betas[i] * self.mixer_hamiltonian)
            state = mixer_exp @ state
        return state

    def cost_function(self, params: np.ndarray) -> float:
        """QAOA 代价函数"""
        psi = self.ansatz(params)
        return float(np.real(psi.conj() @ self.cost_hamiltonian @ psi))

    def solve(self, max_iter: int = 500) -> OptimizationResult:
        """求解 QAOA"""
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * self.p)
        optimizer = COBYLAOptimizer(max_iter=max_iter)
        return optimizer.minimize(self.cost_function, initial_params)

    def _matrix_exp(self, matrix: np.ndarray) -> np.ndarray:
        """矩阵指数"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.conj().T


class VariationalQuantumLinearSolver:
    """变分量子线性方程组求解器 (VQLS)"""

    def __init__(self, matrix_a: np.ndarray, vector_b: np.ndarray):
        self.A = matrix_a
        self.b = vector_b
        self.n = int(np.log2(matrix_a.shape[0]))

    def cost_function(self, params: np.ndarray, ansatz: Callable) -> float:
        """代价函数: ||A|x(θ)⟩ - |b⟩||²"""
        psi = ansatz(params)
        residual = self.A @ psi - self.b
        return float(np.real(residual.conj() @ residual))

    def solve(self, ansatz: Callable, initial_params: np.ndarray, max_iter: int = 1000) -> OptimizationResult:
        """求解线性方程组"""
        optimizer = COBYLAOptimizer(max_iter=max_iter)
        return optimizer.minimize(lambda p: self.cost_function(p, ansatz), initial_params)


class NaturalGradientOptimizer:
    """自然梯度优化器"""

    def __init__(self, lr: float = 0.01, regularization: float = 1e-4):
        self.lr = lr
        self.reg = regularization

    def compute_fubini_metric(self, params: np.ndarray, gradient_fn: Callable) -> np.ndarray:
        """计算 Fubini-Study 度量张量"""
        n = len(params)
        metric = np.zeros((n, n))
        grads = gradient_fn(params)
        for i in range(n):
            for j in range(n):
                metric[i, j] = grads[i] * grads[j]
        return metric + self.reg * np.eye(n)

    def step(self, params: np.ndarray, gradient: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """自然梯度更新"""
        natural_grad = np.linalg.solve(metric, gradient)
        return params - self.lr * natural_grad


class QuantumNaturalGradient:
    """量子自然梯度"""

    def __init__(self, shift: float = np.pi / 2):
        self.shift = shift
        self.ps_rule = ParameterShiftRule(shift)

    def compute_metric_tensor(self, params: np.ndarray, ansatz: Callable) -> np.ndarray:
        """计算量子 Fisher 信息矩阵"""
        n = len(params)
        metric = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                # 参数偏移计算
                params_pp = params.copy(); params_pp[i] += self.shift; params_pp[j] += self.shift
                params_pm = params.copy(); params_pm[i] += self.shift; params_pm[j] -= self.shift
                params_mp = params.copy(); params_mp[i] -= self.shift; params_mp[j] += self.shift
                params_mm = params.copy(); params_mm[i] -= self.shift; params_mm[j] -= self.shift
                psi_pp = ansatz(params_pp)
                psi_pm = ansatz(params_pm)
                psi_mp = ansatz(params_mp)
                psi_mm = ansatz(params_mm)
                val = (np.abs(np.vdot(psi_pp, psi_pm)) - np.abs(np.vdot(psi_mp, psi_mm))) / 4
                metric[i, j] = val
                metric[j, i] = val
        return metric + 1e-6 * np.eye(n)


class OptimizationEngine:
    """优化引擎 - 统一接口"""

    def __init__(self, method: str = "cobyla", max_iter: int = 1000, tol: float = 1e-8):
        self.method = method
        self.max_iter = max_iter
        self.tol = tol

    def minimize(self, loss_fn: Callable, x0: np.ndarray) -> OptimizationResult:
        """最小化"""
        if self.method == "cobyla":
            opt = COBYLAOptimizer(self.max_iter, self.tol)
            return opt.minimize(loss_fn, x0)
        elif self.method == "adam":
            opt = AdamOptimizer()
            x = x0.copy()
            history = []
            for i in range(self.max_iter):
                loss = loss_fn(x)
                history.append(float(loss))
                # 数值梯度
                grad = np.zeros_like(x)
                for j in range(len(x)):
                    x_p = x.copy(); x_p[j] += 1e-6
                    grad[j] = (loss_fn(x_p) - loss) / 1e-6
                x = opt.step(x, grad)
                if np.linalg.norm(grad) < self.tol:
                    return OptimizationResult(x, float(loss), i+1, history, True)
            return OptimizationResult(x, float(loss), self.max_iter, history, False)
        elif self.method == "spsa":
            opt = SPSAOptimizer()
            x = x0.copy()
            history = []
            for i in range(self.max_iter):
                x, loss = opt.step(x, loss_fn)
                history.append(float(loss))
            return OptimizationResult(x, float(loss), self.max_iter, history, False)
        else:
            opt = COBYLAOptimizer(self.max_iter, self.tol)
            return opt.minimize(loss_fn, x0)

    def vqe(self, ansatz: Callable, hamiltonian: np.ndarray, initial_params: np.ndarray) -> OptimizationResult:
        """VQE 求解"""
        vqe = VariationalQuantumEigensolver(ansatz, hamiltonian, self.method)
        return vqe.solve(initial_params, self.max_iter)

    def qaoa(self, cost_h: np.ndarray, mixer_h: np.ndarray, p: int = 1) -> OptimizationResult:
        """QAOA 求解"""
        qaoa = QAOA(cost_h, mixer_h, p)
        return qaoa.solve(self.max_iter)

    def parameter_shift_gradient(self, params: np.ndarray, loss_fn: Callable) -> np.ndarray:
        """参数偏移规则梯度"""
        ps = ParameterShiftRule()
        return ps.compute_full_gradient(params, loss_fn)
