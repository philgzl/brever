class Registry:
    def __init__(self, tag):
        self.tag = tag
        self._registry = {}

    def register(self, name):

        def inner_wrapper(wrapped_class):
            if name in self._registry:
                raise ValueError(f'"{name}" already in {self.tag} registry')
            self._registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get(self, name):
        if name in self._registry:
            return self._registry[name]
        else:
            raise KeyError(f'"{name}" not in {self.tag} registry')

    def keys(self):
        return self._registry.keys()
