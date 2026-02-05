import lightning as L


def seed_everything(seed: int, deterministic: bool = True):
    L.seed_everything(seed, workers=True)
    # Lightning 的 deterministic 会交给 Trainer(deterministic=...) 控制
    # 这里只负责统一入口
    return seed
