class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name=None):
        """作为装饰器使用"""
        def decorator(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise ValueError(f"{key} already registered in {self._name}")
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, name, *args, **kwargs):
        """根据名字实例化对象"""
        if name not in self._registry:
            raise KeyError(f"{name} not found. Available: {list(self._registry.keys())}")
        return self._registry[name](*args, **kwargs)

    def __contains__(self, name):
        return name in self._registry

TRAINER_REGISTRY  = Registry("trainer")
LOADER_REGISTRY   = Registry("loader")
