"""用实际观察检查Q网络是否已退化；不参与训练，不消耗经验采样随机数。"""
import torch
from network import QNetwork


def inspect_network(network: QNetwork, observations: torch.Tensor) -> dict[str, float | bool | int]:
    """输入float32[B,N]真实观察；全批次隐藏层全零才标记失活。"""
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError('健康检查需要非空观察批次')
    with torch.no_grad():
        first = network.layers[:2](observations)
        second = network.layers[2:4](first)
        q = network.layers[4](second)
        return dict(samples=observations.shape[0],
                    finite=bool(torch.isfinite(q).all()),
                    first_active_units=int((first > 0).any(dim=0).sum()),
                    second_active_units=int((second > 0).any(dim=0).sum()),
                    collapsed=bool((first == 0).all() or (second == 0).all()),
                    q_max_abs=float(q.abs().max()),
                    q_state_span=float((q.max(dim=0).values - q.min(dim=0).values).max()))
