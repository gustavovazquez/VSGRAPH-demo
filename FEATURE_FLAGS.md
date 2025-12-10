# Feature Flags y Comparación de Rendimiento

Este documento explica cómo usar los feature flags para comparar el rendimiento entre la implementación original y las optimizaciones.

## Feature Flags Disponibles

### 1. `use_vectorization` (VSGraphEncoder)

Controla si se usa la implementación vectorizada optimizada o la implementación original con bucles.

```python
from vsgraph.encoder import VSGraphEncoder

# Implementación original (bucles Python)
encoder_original = VSGraphEncoder(
    dimension=8192,
    use_vectorization=False  # Desactiva vectorización
)

# Implementación optimizada (operaciones vectorizadas)
encoder_optimized = VSGraphEncoder(
    dimension=8192,
    use_vectorization=True  # Activa vectorización (default)
)
```

**Qué afecta:**
- `spike_diffusion()`: Usa bucles vs. multiplicación de matrices sparse
- `associative_message_passing()`: Usa diccionarios vs. operaciones con matrices sparse

### 2. `n_jobs` (VSGraphEncoder y VSGraphEvaluator)

Controla el número de workers para procesamiento paralelo.

```python
# Sin paralelización (original)
encoder_sequential = VSGraphEncoder(
    dimension=8192,
    n_jobs=1  # Procesamiento secuencial
)

# Con paralelización
encoder_parallel = VSGraphEncoder(
    dimension=8192,
    n_jobs=-1  # Usa todos los cores disponibles
)

# Número específico de workers
encoder_custom = VSGraphEncoder(
    dimension=8192,
    n_jobs=4  # Usa 4 cores
)
```

### 3. `parallel_folds` (VSGraphEvaluator.evaluate)

Controla si los folds de cross-validation se procesan en paralelo.

```python
from vsgraph.evaluator import VSGraphEvaluator

encoder = VSGraphEncoder(dimension=8192, n_jobs=1)
evaluator = VSGraphEvaluator(encoder, n_folds=10, n_jobs=-1)

# Sin paralelización de folds
results_seq = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=False
)

# Con paralelización de folds
results_par = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=True
)
```

## Configuraciones de Comparación

### Configuración 1: Completamente Original
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,
    use_vectorization=False
)
evaluator = VSGraphEvaluator(encoder, n_jobs=1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=False)
```

**Características:**
- Implementación original sin modificaciones
- Bucles Python para todas las operaciones
- Sin paralelización
- **Uso:** Baseline para mediciones

### Configuración 2: Solo Vectorización
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,
    use_vectorization=True
)
evaluator = VSGraphEvaluator(encoder, n_jobs=1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=False)
```

**Características:**
- Operaciones vectorizadas con NumPy/SciPy
- Sin paralelización multi-core
- **Mejora esperada:** 4-8× más rápido que original
- **Uso:** Medir beneficio de vectorización

### Configuración 3: Encoding Paralelo
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=-1,
    use_vectorization=True
)
evaluator = VSGraphEvaluator(encoder, n_jobs=1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=False)
```

**Características:**
- Vectorización + paralelización de encoding de grafos
- Folds de CV secuenciales
- **Mejora esperada:** 4-16× más rápido que original
- **Uso:** Datasets con muchos grafos

### Configuración 4: CV Paralela
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=1,
    use_vectorization=True
)
evaluator = VSGraphEvaluator(encoder, n_jobs=-1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=True)
```

**Características:**
- Vectorización + paralelización de folds de CV
- Encoding secuencial dentro de cada fold
- **Mejora esperada:** Hasta 10× para 10-fold CV
- **Uso:** Evaluación intensiva en CV

### Configuración 5: Máxima Optimización (Recomendada)
```python
encoder = VSGraphEncoder(
    dimension=8192,
    n_jobs=-1,
    use_vectorization=True
)
evaluator = VSGraphEvaluator(encoder, n_jobs=-1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=True)
```

**Características:**
- Todas las optimizaciones habilitadas
- **Mejora esperada:** 10-30× más rápido que original
- **Uso:** Producción con datasets grandes

## Scripts de Comparación

### Script 1: Comparación Completa

Compara todas las configuraciones y mide tiempos detallados:

```bash
# Comparación básica con datos sintéticos
python compare_performance.py

# Comparación con dataset real
python compare_performance.py --dataset MUTAG

# Comparación con parámetros personalizados
python compare_performance.py --n-graphs 200 --dimension 4096 --n-repeats 5

# Comparación rápida (sin CV ni detalles)
python compare_performance.py --skip-cv --skip-detail
```

**Opciones:**
- `--dataset NAME`: Usar dataset de TUDataset (MUTAG, PROTEINS, etc.)
- `--n-graphs N`: Número de grafos sintéticos a crear (default: 100)
- `--dimension D`: Dimensión de hypervectores (default: 2048)
- `--n-repeats R`: Repeticiones por test (default: 3)
- `--skip-cv`: Omitir comparación de cross-validation
- `--skip-detail`: Omitir comparación detallada de operaciones

**Salida:**
```
================================================================================
VS-GRAPH PERFORMANCE COMPARISON
================================================================================
System: 16 CPU cores available
Dimension: 2048
Repeats per test: 3

Creating 100 synthetic graphs...

================================================================================
COMPARISON 1: Graph Encoding Performance
================================================================================

Configuration                                 Mean Time (s)   Speedup    Graphs/sec
--------------------------------------------------------------------------------
Original (no vectorization, sequential)           2.5000       1.00×       40.00
Vectorized (with vectorization, sequential)       0.6250       4.00×      160.00
Parallel 2-cores (vectorized)                     0.3500       7.14×      285.71
Parallel 4-cores (vectorized)                     0.2000      12.50×      500.00
Parallel all-cores (vectorized, 16 cores)         0.1500      16.67×      666.67
================================================================================
```

### Script 2: Utilidades de Timing

Usar las utilidades de timing en tu propio código:

```python
from vsgraph.timing_utils import ComparisonTimer, PerformanceTimer, time_operation

# Timing simple
with time_operation("my_encoding"):
    embeddings = encoder.encode_graphs(graphs)

# Timer con estadísticas
timer = PerformanceTimer("My Experiment")

for i in range(10):
    with timer.time("encoding"):
        embeddings = encoder.encode_graphs(graphs)

timer.print_summary()

# Comparación entre versiones
comp_timer = ComparisonTimer("Original vs Optimized")

with comp_timer.time_version("original", "encoding"):
    encoder_orig = VSGraphEncoder(use_vectorization=False, n_jobs=1)
    emb1 = encoder_orig.encode_graphs(graphs)

with comp_timer.time_version("optimized", "encoding"):
    encoder_opt = VSGraphEncoder(use_vectorization=True, n_jobs=-1)
    emb2 = encoder_opt.encode_graphs(graphs)

comp_timer.print_comparison("encoding")
```

## Ejemplos de Uso Práctico

### Ejemplo 1: Comparar Vectorización en un Grafo Grande

```python
import networkx as nx
from vsgraph.encoder import VSGraphEncoder
from vsgraph.timing_utils import ComparisonTimer

# Crear grafo grande
G = nx.barabasi_albert_graph(1000, 5)
G = nx.convert_node_labels_to_integers(G)

timer = ComparisonTimer("Vectorization Impact")

# Original
with timer.time_version("original"):
    encoder1 = VSGraphEncoder(use_vectorization=False)
    emb1 = encoder1.encode_graph(G)

# Vectorizado
with timer.time_version("vectorized"):
    encoder2 = VSGraphEncoder(use_vectorization=True)
    emb2 = encoder2.encode_graph(G)

timer.print_comparison()
```

### Ejemplo 2: Encontrar el Número Óptimo de Workers

```python
from multiprocessing import cpu_count
from vsgraph.timing_utils import PerformanceTimer

timer = PerformanceTimer("Scaling Test")
graphs, labels = load_tudataset("MUTAG")

for n_jobs in [1, 2, 4, 8, cpu_count()]:
    encoder = VSGraphEncoder(dimension=4096, n_jobs=n_jobs)

    with timer.time(f"n_jobs={n_jobs}"):
        embeddings = encoder.encode_graphs(graphs)

timer.print_summary()
```

### Ejemplo 3: Medir Mejora Total en Pipeline Completo

```python
import time
from vsgraph.data_loader import load_tudataset
from vsgraph.encoder import VSGraphEncoder
from vsgraph.evaluator import VSGraphEvaluator
import numpy as np

graphs, labels = load_tudataset("PROTEINS")
num_classes = len(np.unique(labels))

# Configuración original
print("Testing ORIGINAL implementation...")
start = time.time()
encoder_orig = VSGraphEncoder(dimension=2048, n_jobs=1, use_vectorization=False)
evaluator_orig = VSGraphEvaluator(encoder_orig, n_folds=5, n_repeats=1, n_jobs=1)
results_orig = evaluator_orig.evaluate(graphs, labels, num_classes,
                                       verbose=False, parallel_folds=False)
time_orig = time.time() - start

# Configuración optimizada
print("Testing OPTIMIZED implementation...")
start = time.time()
encoder_opt = VSGraphEncoder(dimension=2048, n_jobs=-1, use_vectorization=True)
evaluator_opt = VSGraphEvaluator(encoder_opt, n_folds=5, n_repeats=1, n_jobs=-1)
results_opt = evaluator_opt.evaluate(graphs, labels, num_classes,
                                     verbose=False, parallel_folds=True)
time_opt = time.time() - start

# Comparar
print(f"\n{'='*60}")
print("PIPELINE COMPARISON")
print(f"{'='*60}")
print(f"Original:  {time_orig:.2f}s - Accuracy: {results_orig['accuracy_mean']:.2f}%")
print(f"Optimized: {time_opt:.2f}s - Accuracy: {results_opt['accuracy_mean']:.2f}%")
print(f"Speedup:   {time_orig/time_opt:.2f}×")
print(f"{'='*60}")
```

## Verificación de Resultados

Las optimizaciones NO deben cambiar los resultados numéricos:

```python
# Verificar que las optimizaciones dan resultados equivalentes
encoder1 = VSGraphEncoder(use_vectorization=False, seed=42)
encoder2 = VSGraphEncoder(use_vectorization=True, seed=42)

G = nx.karate_club_graph()
emb1 = encoder1.encode_graph(G)
emb2 = encoder2.encode_graph(G)

# Las semillas aleatorias deben producir los mismos basis vectors
# pero las operaciones numéricas pueden tener diferencias de punto flotante pequeñas
print(f"Difference: {np.max(np.abs(emb1 - emb2)):.10f}")
# Debería ser muy pequeño (< 1e-5)
```

## Recomendaciones por Tamaño de Dataset

| Tamaño Dataset | Configuración Recomendada | Speedup Esperado |
|----------------|---------------------------|------------------|
| < 50 grafos | Vectorización, n_jobs=1 | 4-8× |
| 50-500 grafos | Vectorización, n_jobs=-1 | 8-16× |
| 500-5000 grafos | Encoding paralelo | 10-20× |
| > 5000 grafos | CV + Encoding paralelo | 15-30× |

## Troubleshooting

### "No veo mejora de rendimiento"
- **Problema:** Dataset muy pequeño (<10 grafos)
- **Solución:** Usar configuración secuencial, overhead de paralelización es mayor que beneficio

### "Más lento con paralelización"
- **Problema:** Grafos muy pequeños (< 10 nodos)
- **Solución:** Usar solo vectorización sin paralelización

### "Uso excesivo de memoria"
- **Problema:** Demasiados workers paralelos
- **Solución:** Reducir `n_jobs` a 2-4

### "Resultados difieren entre versiones"
- **Problema:** Semillas aleatorias no están fijadas
- **Solución:** Establecer `seed` en el encoder

## Resumen

Los feature flags te permiten:

1. **Medir** el impacto de cada optimización individualmente
2. **Comparar** rendimiento entre configuraciones
3. **Elegir** la mejor configuración para tu hardware y dataset
4. **Verificar** que las optimizaciones no cambian los resultados

Usa `compare_performance.py` para obtener un análisis completo del rendimiento en tu sistema.
