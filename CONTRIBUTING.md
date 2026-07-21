# Contributing To FedGB

New algorithms must:

1. keep client and server logic in separate files;
2. place algorithm-specific hyperparameters in a dedicated configuration module;
3. preserve the method's original core algorithm rather than replacing it with a simplified approximation;
4. declare supported scenarios in the public registry;
5. add focused unit tests and at least one end-to-end smoke case;
6. use the shared task, metric, logging, and dataset interfaces;
7. avoid writing caches or results inside dataset directories.

Before submitting a change, run:

```bash
PYTHONPATH=. python -m compileall -q fedgb examples scripts tests
PYTHONPATH=. pytest tests -q
PYTHONPATH=. python scripts/verify/run_smoke_matrix.py --cpu
```

