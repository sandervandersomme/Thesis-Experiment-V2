from methods_diversity import diversity_methods
from methods_privacy import privacy_methods
from methods_similarity import similarity_methods
from methods_utility import utility_methods

from typing import List

class Metric:
    def __init__(self, name, methods: List[function]):
        self.name = name
        self.methods = methods
        self.results = {}

    def evaluate(self, data):
        for method in self.methods:
            self.results[method.__name__] = method(data)

class Privacy(Metric):
    def __init__(self):
        super().__init__("privacy", privacy_methods)

class Similarity(Metric):
    def __init__(self):
        super().__init__("similarity", similarity_methods)

class Utility(Metric):
    def __init__(self):
        super().__init__("utility", utility_methods)

class Diversity(Metric):
    def __init__(self):
        super().__init__("diversity", diversity_methods)



