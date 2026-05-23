"""量子任务调度器 - 资源分配/优先级/并行化/负载均衡"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import math
import time
import uuid


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ResourceState(Enum):
    """资源状态"""
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class ResourceInfo:
    """量子计算资源信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    num_qubits: int = 0
    state: ResourceState = ResourceState.IDLE
    gate_error_rate: float = 0.001
    readout_error_rate: float = 0.01
    t1_time: float = 50e-6
    t2_time: float = 70e-6
    queue_length: int = 0
    max_circuits: int = 300

    def quality_score(self) -> float:
        """资源质量评分"""
        return (1.0 - self.gate_error_rate) * (1.0 - self.readout_error_rate) * \
               min(1.0, self.t1_time / 100e-6) * min(1.0, self.t2_time / 100e-6)

    def availability(self) -> float:
        """可用性评估"""
        if self.state == ResourceState.OFFLINE:
            return 0.0
        if self.state == ResourceState.MAINTENANCE:
            return 0.3
        return max(0, 1.0 - self.queue_length / self.max_circuits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'name': self.name,
            'num_qubits': self.num_qubits, 'state': self.state.value,
            'quality_score': self.quality_score(),
            'availability': self.availability()
        }


@dataclass
class ScheduledTask:
    """已调度任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    num_qubits: int = 1
    shots: int = 1024
    priority: TaskPriority = TaskPriority.NORMAL
    resource_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

    def wait_time(self) -> float:
        """等待时间"""
        start = self.started_at if self.started_at > 0 else time.time()
        return start - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'name': self.name,
            'num_qubits': self.num_qubits, 'shots': self.shots,
            'priority': self.priority.value, 'status': self.status,
            'resource_id': self.resource_id
        }


class ResourcePool:
    """资源池管理器"""

    def __init__(self):
        self.resources: Dict[str, ResourceInfo] = {}

    def add_resource(self, resource: ResourceInfo):
        """添加资源"""
        self.resources[resource.id] = resource

    def remove_resource(self, resource_id: str) -> bool:
        """移除资源"""
        return self.resources.pop(resource_id, None) is not None

    def get_available(self, min_qubits: int = 1) -> List[ResourceInfo]:
        """获取可用资源"""
        return [r for r in self.resources.values()
                if r.state in (ResourceState.IDLE, ResourceState.BUSY) and r.num_qubits >= min_qubits]

    def get_best_resource(self, num_qubits: int) -> Optional[ResourceInfo]:
        """选择最佳资源"""
        available = self.get_available(num_qubits)
        if not available:
            return None
        return max(available, key=lambda r: r.quality_score() * r.availability())

    def update_state(self, resource_id: str, state: ResourceState):
        """更新资源状态"""
        if resource_id in self.resources:
            self.resources[resource_id].state = state

    def utilization(self) -> Dict[str, float]:
        """资源利用率"""
        result = {}
        for rid, r in self.resources.items():
            result[rid] = 1.0 - r.availability()
        return result

    def total_qubits(self) -> int:
        """总可用量子比特数"""
        return sum(r.num_qubits for r in self.resources.values()
                   if r.state != ResourceState.OFFLINE)


class PriorityScheduler:
    """优先级调度器"""

    def __init__(self, resource_pool: ResourcePool):
        self.resource_pool = resource_pool
        self.task_queue: List[ScheduledTask] = []
        self.completed: List[ScheduledTask] = []

    def submit(self, task: ScheduledTask):
        """提交任务"""
        task.status = "queued"
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)

    def schedule_next(self) -> Optional[ScheduledTask]:
        """调度下一个任务"""
        for i, task in enumerate(self.task_queue):
            resource = self.resource_pool.get_best_resource(task.num_qubits)
            if resource:
                self.task_queue.pop(i)
                task.resource_id = resource.id
                task.started_at = time.time()
                task.status = "running"
                resource.queue_length += 1
                return task
        return None

    def complete_task(self, task_id: str, result: Optional[Dict] = None):
        """完成任务"""
        for task in self.task_queue + self.completed:
            if task.id == task_id:
                task.status = "completed"
                task.completed_at = time.time()
                task.result = result
                if task.resource_id:
                    res = self.resource_pool.resources.get(task.resource_id)
                    if res:
                        res.queue_length = max(0, res.queue_length - 1)
                return
        # 查找运行中的
        self.completed = [t for t in self.completed if t.id != task_id]

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'pending': len(self.task_queue),
            'completed': len(self.completed),
            'by_priority': {
                p.name: sum(1 for t in self.task_queue if t.priority == p)
                for p in TaskPriority
            }
        }


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, resource_pool: ResourcePool):
        self.resource_pool = resource_pool
        self.assignment_history: List[Dict[str, Any]] = []

    def round_robin(self, tasks: List[ScheduledTask]) -> Dict[str, str]:
        """轮询分配"""
        available = self.resource_pool.get_available()
        if not available:
            return {}
        assignments = {}
        for i, task in enumerate(tasks):
            resource = available[i % len(available)]
            assignments[task.id] = resource.id
        return assignments

    def weighted_assignment(self, tasks: List[ScheduledTask]) -> Dict[str, str]:
        """基于权重的分配"""
        available = self.resource_pool.get_available()
        if not available:
            return {}
        weights = np.array([r.quality_score() * r.availability() for r in available])
        if np.sum(weights) < 1e-15:
            weights = np.ones(len(available))
        weights /= np.sum(weights)
        assignments = {}
        rng = np.random.RandomState(42)
        for task in tasks:
            idx = rng.choice(len(available), p=weights)
            assignments[task.id] = available[idx].id
        return assignments

    def load_balance(self, tasks: List[ScheduledTask]) -> Dict[str, str]:
        """智能负载均衡"""
        available = self.resource_pool.get_available()
        if not available:
            return {}
        assignments = {}
        loads = {r.id: 0 for r in available}
        for task in tasks:
            best = min(available, key=lambda r: loads[r.id] / max(r.num_qubits, 1))
            assignments[task.id] = best.id
            loads[best.id] += 1
        self.assignment_history.append(assignments)
        return assignments


class CircuitBatcher:
    """电路批处理器"""

    def __init__(self, max_batch_size: int = 300, max_qubits: int = 20):
        self.max_batch_size = max_batch_size
        self.max_qubits = max_qubits

    def batch_circuits(self, circuits: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """将电路分批"""
        batches = []
        current_batch = []
        for circuit in circuits:
            if len(current_batch) >= self.max_batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(circuit)
        if current_batch:
            batches.append(current_batch)
        return batches

    def estimate_execution_time(self, num_circuits: int, shots: int,
                                num_qubits: int, overhead: float = 0.1) -> float:
        """估计执行时间"""
        base_time_per_shot = 0.001 * num_qubits
        circuit_time = num_circuits * shots * base_time_per_shot
        return circuit_time * (1 + overhead)


class SchedulingEngine:
    """调度引擎 - 统一接口"""

    def __init__(self):
        self.resource_pool = ResourcePool()
        self.scheduler = PriorityScheduler(self.resource_pool)
        self.load_balancer = LoadBalancer(self.resource_pool)
        self.batcher = CircuitBatcher()

    def add_backend(self, name: str, num_qubits: int, gate_error: float = 0.001,
                    readout_error: float = 0.01) -> str:
        """添加后端"""
        resource = ResourceInfo(
            name=name, num_qubits=num_qubits,
            gate_error_rate=gate_error, readout_error_rate=readout_error
        )
        self.resource_pool.add_resource(resource)
        return resource.id

    def submit_circuit(self, name: str, num_qubits: int, shots: int = 1024,
                       priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交电路"""
        task = ScheduledTask(name=name, num_qubits=num_qubits, shots=shots, priority=priority)
        self.scheduler.submit(task)
        return task.id

    def process_queue(self) -> List[ScheduledTask]:
        """处理队列中的所有可调度任务"""
        scheduled = []
        while True:
            task = self.scheduler.schedule_next()
            if task is None:
                break
            scheduled.append(task)
        return scheduled

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'resources': {rid: r.to_dict() for rid, r in self.resource_pool.resources.items()},
            'queue': self.scheduler.get_queue_status(),
            'total_qubits': self.resource_pool.total_qubits()
        }
