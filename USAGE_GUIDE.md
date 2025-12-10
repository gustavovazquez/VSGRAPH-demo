## Guía de Uso - VS-Graph con Optimizaciones Multi-Core

### Inicio Rápido

#### 1. Usar Configuración por Defecto (Optimizada)

```python
from vsgraph.encoder import VSGraphEncoder
from vsgraph.data_loader import load_tudataset

# Cargar dataset
graphs, labels = load_tudataset("MUTAG")

# Encoder optimizado (default: vectorización + todos los cores)
encoder = VSGraphEncoder(dimension=8192)
embeddings = encoder.encode_graphs(graphs, verbose=True)
```

#### 2. Volver a la Implementación Original

```python
# Configuración exactamente igual a la implementación original
encoder_original = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,                    # Sin paralelización
    use_vectorization=False      # Sin vectorización
)
embeddings = encoder_original.encode_graphs(graphs, verbose=True)
```

#### 3. Comparar Rendimiento

```bash
# Comparación rápida
python compare_performance.py --n-graphs 100 --dimension 2048

# Comparación completa con dataset real
python compare_performance.py --dataset MUTAG --n-repeats 5

# Comparación solo de encoding (más rápido)
python compare_performance.py --skip-cv --skip-detail
```

### Cuándo Usar Cada Configuración

#### Configuración A: Original (Baseline)
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,
    use_vectorization=False
)
```
**Cuándo usar:**
- Verificar que las optimizaciones dan los mismos resultados
- Benchmarking y medición de mejoras
- Debugging de problemas numéricos

**NO usar para producción** - Es la más lenta

#### Configuración B: Solo Vectorización
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,
    use_vectorization=True
)
```
**Cuándo usar:**
- Grafos medianos a grandes (>50 nodos)
- Datasets pequeños (<50 grafos)
- Ambientes con pocos cores disponibles

**Ventajas:**
- 4-8× más rápido que original en grafos grandes
- No consume recursos de CPU adicionales
- Predecible y estable

**Desventajas:**
- Puede ser más lento en grafos muy pequeños (<20 nodos)

#### Configuración C: Vectorización + Paralelismo (Recomendada)
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=-1,  # Usa todos los cores
    use_vectorization=True
)
```
**Cuándo usar:**
- Datasets medianos a grandes (>100 grafos)
- Grafos medianos a grandes (>50 nodos)
- Sistema con múltiples cores (4+)
- **Esta es la configuración por defecto**

**Ventajas:**
- 10-30× más rápido que original en datasets grandes
- Aprovecha todo el hardware disponible
- Mejor rendimiento general

**Desventajas:**
- Mayor uso de memoria
- Overhead en datasets muy pequeños

#### Configuración D: Paralelización Controlada
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=4,  # Usa exactamente 4 cores
    use_vectorization=True
)
```
**Cuándo usar:**
- Limitar uso de CPU (entorno compartido)
- Controlar uso de memoria
- Balancear con otros procesos

### Configuración para Cross-Validation

#### CV Original
```python
encoder = VSGraphEncoder(n_jobs=1, use_vectorization=False)
evaluator = VSGraphEvaluator(encoder, n_folds=10, n_jobs=1)

results = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=False
)
```

#### CV con Folds Paralelos (Recomendado)
```python
encoder = VSGraphEncoder(n_jobs=1, use_vectorization=True)
evaluator = VSGraphEvaluator(encoder, n_folds=10, n_jobs=-1)

results = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=True
)
```

#### CV Máxima Optimización
```python
encoder = VSGraphEncoder(n_jobs=2, use_vectorization=True)
evaluator = VSGraphEvaluator(encoder, n_folds=5, n_jobs=5)

results = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=True
)
```
**Nota:** 2 cores × 5 folds = 10 cores totales en uso

### Medición de Tiempos

#### Método 1: Timing Simple
```python
import time

start = time.time()
embeddings = encoder.encode_graphs(graphs)
elapsed = time.time() - start

print(f"Tiempo: {elapsed:.2f}s")
print(f"Throughput: {len(graphs)/elapsed:.2f} grafos/seg")
```

#### Método 2: Con Context Manager
```python
from vsgraph.timing_utils import time_operation

with time_operation("Encoding"):
    embeddings = encoder.encode_graphs(graphs)
```

#### Método 3: Comparación Detallada
```python
from vsgraph.timing_utils import ComparisonTimer

timer = ComparisonTimer()

# Configuración 1
with timer.time_version("original"):
    enc1 = VSGraphEncoder(use_vectorization=False, n_jobs=1)
    emb1 = enc1.encode_graphs(graphs)

# Configuración 2
with timer.time_version("optimized"):
    enc2 = VSGraphEncoder(use_vectorization=True, n_jobs=-1)
    emb2 = enc2.encode_graphs(graphs)

timer.print_comparison()
```

#### Método 4: Estadísticas Múltiples Corridas
```python
from vsgraph.timing_utils import PerformanceTimer

timer = PerformanceTimer("Encoder Test")
encoder = VSGraphEncoder(dimension=4096)

# Ejecutar 10 veces
for i in range(10):
    with timer.time("encoding"):
        embeddings = encoder.encode_graphs(graphs)

# Ver estadísticas
timer.print_summary()
# Muestra: mean, std, min, max, median
```

### Ejemplos Completos

#### Ejemplo 1: Pipeline Completo con Timing
```python
from vsgraph.data_loader import load_tudataset
from vsgraph.encoder import VSGraphEncoder
from vsgraph.evaluator import VSGraphEvaluator
from vsgraph.timing_utils import PerformanceTimer
import numpy as np

# Configurar timer
timer = PerformanceTimer("VS-Graph Pipeline")

# Cargar datos
with timer.time("data_loading"):
    graphs, labels = load_tudataset("MUTAG")
    num_classes = len(np.unique(labels))

print(f"Loaded {len(graphs)} graphs")

# Crear encoder optimizado
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=-1,
    use_vectorization=True
)

# Crear evaluator
evaluator = VSGraphEvaluator(
    encoder,
    n_folds=10,
    n_repeats=3,
    n_jobs=-1
)

# Evaluar
with timer.time("evaluation"):
    results = evaluator.evaluate(
        graphs, labels, num_classes,
        verbose=True,
        parallel_folds=True
    )

# Mostrar resultados
print(f"\nAccuracy: {results['accuracy_mean']:.2f} ± {results['accuracy_std']:.2f}%")
timer.print_summary()
```

#### Ejemplo 2: Comparación Lado a Lado
```python
from vsgraph.data_loader import load_tudataset
from vsgraph.encoder import VSGraphEncoder
import time

graphs, labels = load_tudataset("PROTEINS")
print(f"Testing on {len(graphs)} graphs\n")

configs = [
    ("Original", {"use_vectorization": False, "n_jobs": 1}),
    ("Vectorized", {"use_vectorization": True, "n_jobs": 1}),
    ("Parallel-4", {"use_vectorization": True, "n_jobs": 4}),
    ("Parallel-All", {"use_vectorization": True, "n_jobs": -1}),
]

results = {}

for name, params in configs:
    encoder = VSGraphEncoder(dimension=2048, **params)

    start = time.time()
    embeddings = encoder.encode_graphs(graphs, verbose=False)
    elapsed = time.time() - start

    results[name] = elapsed
    print(f"{name:15s}: {elapsed:6.2f}s  ({len(graphs)/elapsed:6.1f} grafos/s)")

# Calcular speedups
baseline = results["Original"]
print(f"\nSpeedups vs Original:")
for name, elapsed in results.items():
    if name != "Original":
        print(f"  {name:15s}: {baseline/elapsed:5.2f}×")
```

#### Ejemplo 3: Encontrar Configuración Óptima
```python
from multiprocessing import cpu_count
from vsgraph.timing_utils import PerformanceTimer
import numpy as np

graphs, labels = load_tudataset("ENZYMES")
timer = PerformanceTimer()

# Probar diferentes números de workers
n_jobs_values = [1, 2, 4, 8, cpu_count()]

for n_jobs in n_jobs_values:
    encoder = VSGraphEncoder(dimension=4096, n_jobs=n_jobs)

    # Medir 3 veces
    for _ in range(3):
        with timer.time(f"n_jobs={n_jobs}"):
            embeddings = encoder.encode_graphs(graphs, verbose=False)

# Encontrar mejor configuración
summary = timer.get_summary()
best_config = min(summary.items(), key=lambda x: x[1]['mean'])

print(f"\nMejor configuración: {best_config[0]}")
print(f"Tiempo promedio: {best_config[1]['mean']:.2f}s")

timer.print_summary()
```

### Consejos de Rendimiento

#### 1. Dataset Pequeño (<50 grafos)
```python
# Mejor: solo vectorización
encoder = VSGraphEncoder(n_jobs=1, use_vectorization=True)
```
**Razón:** Overhead de paralelización > beneficio

#### 2. Dataset Mediano (50-500 grafos)
```python
# Mejor: paralelización moderada
encoder = VSGraphEncoder(n_jobs=4, use_vectorization=True)
```
**Razón:** Buen balance entre speedup y overhead

#### 3. Dataset Grande (>500 grafos)
```python
# Mejor: paralelización máxima
encoder = VSGraphEncoder(n_jobs=-1, use_vectorization=True)
```
**Razón:** Beneficio >> overhead

#### 4. Grafos Pequeños (<20 nodos)
```python
# Mejor: sin vectorización
encoder = VSGraphEncoder(n_jobs=-1, use_vectorization=False)
```
**Razón:** Overhead de matrices sparse > beneficio

#### 5. Grafos Grandes (>100 nodos)
```python
# Mejor: con vectorización
encoder = VSGraphEncoder(n_jobs=-1, use_vectorization=True)
```
**Razón:** Vectorización muy eficiente en grafos grandes

### Troubleshooting

#### Problema: "Vectorización más lenta que original"
**Causa:** Grafos muy pequeños (<20 nodos)
**Solución:**
```python
encoder = VSGraphEncoder(use_vectorization=False, n_jobs=-1)
```

#### Problema: "Paralelización no mejora rendimiento"
**Causa:** Dataset muy pequeño
**Solución:**
```python
encoder = VSGraphEncoder(n_jobs=1, use_vectorization=True)
```

#### Problema: "Uso excesivo de memoria"
**Causa:** Demasiados workers paralelos
**Solución:**
```python
encoder = VSGraphEncoder(n_jobs=2, use_vectorization=True)
```

#### Problema: "Resultados ligeramente diferentes"
**Causa:** Diferentes basis vectors por semillas aleatorias
**Solución:**
```python
encoder = VSGraphEncoder(seed=42, use_vectorization=True)
```

### Scripts Disponibles

1. **`compare_performance.py`** - Comparación completa de rendimiento
2. **`benchmark_parallel.py`** - Benchmark de paralelización
3. **`test_parallel.py`** - Tests de funcionalidad

Ejecutar:
```bash
python compare_performance.py --help
python benchmark_parallel.py
python test_parallel.py
```

### Documentación Adicional

- **[PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md)** - Resumen de modificaciones
- **[PARALLEL_OPTIMIZATIONS.md](PARALLEL_OPTIMIZATIONS.md)** - Detalles técnicos
- **[FEATURE_FLAGS.md](FEATURE_FLAGS.md)** - Guía completa de feature flags
