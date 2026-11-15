"""fin-proxy-optimizer: 三维单肋导热求解 + RBF 代理模型（P95 <= 5%）。"""
__version__ = "1.0.0"


def get_surrogate(path="surrogate.npz"):
    """惰性构造代理模型，避免在 import 阶段强依赖 numpy。"""
    from .predict import Surrogate
    return Surrogate(path)


__all__ = ["__version__", "get_surrogate"]
