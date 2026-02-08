

# src/my_ml_project/service/router.py

import random

class CanaryRouter:
    def __init__(self, canary_pct: float):
        self.canary_pct = canary_pct

    def route(self) -> str:
        return "challenger" if random.random() < self.canary_pct else "champion"


# Canary Router