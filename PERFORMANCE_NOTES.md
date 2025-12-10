# Notas Importantes sobre Rendimiento

## Comportamiento Observado: Vectorización en Grafos Pequeños

### Problema

En datasets con grafos pequeños (< 30 nodos promedio), la **vectorización puede ser más lenta** que la implementación original con bucles Python.

### Ejemplo: MUTAG Dataset

```
Dataset MUTAG: 188 grafos, 17.93 nodos promedio, 19.79 edges promedio

Configuration                                 Mean Time (s)   Speedup
--------------------------------------------------------------------------------
Original (no vectorization, sequential)           0.14s        1.00×     (MÁS RÁPIDO)
Vectorized (with vectorization, sequential)       1.51s        0.09×     (10× MÁS LENTO)
Parallel 2-cores (vectorized)                     3.01s        0.05×     (20× MÁS LENTO)
```

### ¿Por Qué Ocurre Esto?

#### Overhead de Matrices Sparse

La implementación vectorizada usa matrices sparse de SciPy:

```python
# Vectorizado: crea matriz sparse
adj_matrix = nx.adjacency_matrix(graph, nodelist=range(n))  # OVERHEAD
for hop in range(K):
    spikes = adj_matrix @ spikes  # Operación rápida

# Original: usa diccionarios
adj_list = {i: list(graph.neighbors(i)) for i in range(n)}  # MUY RÁPIDO para grafos pequeños
for hop in range(K):
    for i in range(n):
        for j in adj_list[i]:
            new_spikes[i] += spikes[j]  # Simple acceso a array
```

**En grafos pequeños:**
- Crear `adj_matrix` toma ~0.5-1ms
- La operación `adj_matrix @ spikes` toma ~0.1ms
- Total: ~1.5ms por grafo

**Vs. bucles Python:**
- Crear `adj_list` toma ~0.1ms
- Los bucles toman ~0.3ms
- Total: ~0.7ms por grafo

**Resultado:** La versión "optimizada" es 2× más lenta en grafos pequeños.

#### Overhead de Multiprocessing

El overhead de crear procesos paralelos es significativo:

```python
# Overhead por usar multiprocessing:
- Crear pool de procesos: ~50-100ms
- Serializar cada grafo: ~0.1ms/grafo
- Comunicación inter-proceso: ~0.1ms/grafo
- Total overhead: ~100ms + 0.2ms × n_graphs
```

Para datasets pequeños (< 500 grafos), este overhead domina el tiempo total.

## Reglas de Oro

### Cuándo Usar Cada Configuración

#### 1. Grafos Muy Pequeños (< 20 nodos)

**Mejor configuración:**
```python
encoder = VSGraphEncoder(
    use_vectorization=False,  # Sin vectorización
    n_jobs=1                   # Sin paralelización
)
```

**Speedup esperado:** 1× (baseline)
**Cuándo:** MUTAG, pequeños grafos sintéticos

#### 2. Grafos Pequeños (20-50 nodos), Dataset Pequeño (< 200 grafos)

**Mejor configuración:**
```python
encoder = VSGraphEncoder(
    use_vectorization=False,  # Sin vectorización
    n_jobs=1                   # Sin paralelización
)
```

**Speedup esperado:** 1× (baseline)
**Cuándo:** MUTAG, PTC_FM, datasets custom pequeños

#### 3. Grafos Medianos (50-100 nodos), Dataset Pequeño

**Mejor configuración:**
```python
encoder = VSGraphEncoder(
    use_vectorization=True,   # Con vectorización
    n_jobs=1                   # Sin paralelización
)
```

**Speedup esperado:** 2-4×
**Cuándo:** PROTEINS (39 nodos promedio), datasets custom medianos

#### 4. Grafos Grandes (> 100 nodos), Dataset Pequeño

**Mejor configuración:**
```python
encoder = VSGraphEncoder(
    use_vectorization=True,   # Con vectorización
    n_jobs=1                   # Sin paralelización
)
```

**Speedup esperado:** 4-8×
**Cuándo:** DD (284 nodos promedio), grafos grandes custom

#### 5. Cualquier Tamaño de Grafo, Dataset Grande (> 500 grafos)

**Mejor configuración:**
```python
encoder = VSGraphEncoder(
    use_vectorization=True,   # Con vectorización
    n_jobs=-1                  # Paralelización total
)
```

**Speedup esperado:** 5-20×
**Cuándo:** NCI1 (4110 grafos), ENZYMES (600 grafos), datasets grandes

#### 6. Cross-Validation Intensivo

**Mejor configuración:**
```python
encoder = VSGraphEncoder(use_vectorization=True, n_jobs=1)
evaluator = VSGraphEvaluator(encoder, n_folds=10, n_jobs=-1)
results = evaluator.evaluate(graphs, labels, num_classes, parallel_folds=True)
```

**Speedup esperado:** 8-10× (para 10 folds)
**Cuándo:** Cualquier dataset con evaluación exhaustiva

## Tabla de Decisión Rápida

| Tamaño Grafo | Num Grafos | `use_vectorization` | `n_jobs` | Speedup |
|--------------|------------|---------------------|----------|---------|
| < 20 nodos | < 200 | `False` | `1` | 1× (baseline) |
| < 20 nodos | > 200 | `False` | `-1` | 2-4× |
| 20-50 nodos | < 200 | `False` | `1` | 1× |
| 20-50 nodos | > 200 | `True` | `4` | 3-6× |
| 50-100 nodos | < 200 | `True` | `1` | 2-4× |
| 50-100 nodos | > 200 | `True` | `-1` | 8-15× |
| > 100 nodos | < 200 | `True` | `1` | 4-8× |
| > 100 nodos | > 200 | `True` | `-1` | 15-30× |

## Benchmarks en Datasets Reales

### MUTAG (188 grafos, 17.93 nodos promedio)

```python
# RECOMENDADO
encoder = VSGraphEncoder(use_vectorization=False, n_jobs=1)
# Tiempo: ~0.14s
# Throughput: ~1370 grafos/sec
```

### PTC_FM (349 grafos, 14.11 nodos promedio)

```python
# RECOMENDADO
encoder = VSGraphEncoder(use_vectorization=False, n_jobs=1)
# Tiempo esperado: ~0.25s
# Throughput esperado: ~1400 grafos/sec
```

### PROTEINS (1113 grafos, 39.06 nodos promedio)

```python
# RECOMENDADO
encoder = VSGraphEncoder(use_vectorization=True, n_jobs=4)
# Tiempo esperado: ~2-3s
# Speedup vs original: ~3-5×
```

### DD (1178 grafos, 284.32 nodos promedio)

```python
# RECOMENDADO
encoder = VSGraphEncoder(use_vectorization=True, n_jobs=-1)
# Tiempo esperado: ~15-20s
# Speedup vs original: ~15-25×
```

### NCI1 (4110 grafos, 29.87 nodos promedio)

```python
# RECOMENDADO
encoder = VSGraphEncoder(use_vectorization=True, n_jobs=-1)
# Tiempo esperado: ~8-12s
# Speedup vs original: ~10-20×
```

## Cómo Determinar la Mejor Configuración

### Método 1: Usar el Script de Comparación

```bash
# Probar automáticamente todas las configuraciones
python compare_performance.py --dataset MUTAG --n-repeats 3

# Ver solo encoding (más rápido)
python compare_performance.py --dataset MUTAG --skip-cv --skip-detail
```

El script mostrará qué configuración es más rápida para tu dataset específico.

### Método 2: Prueba Manual Rápida

```python
from vsgraph.data_loader import load_tudataset
from vsgraph.encoder import VSGraphEncoder
import time

graphs, labels, num_classes = load_tudataset("MUTAG")

# Probar configuración original
encoder1 = VSGraphEncoder(dimension=2048, use_vectorization=False, n_jobs=1)
start = time.time()
emb1 = encoder1.encode_graphs(graphs)
time1 = time.time() - start

# Probar configuración optimizada
encoder2 = VSGraphEncoder(dimension=2048, use_vectorization=True, n_jobs=-1)
start = time.time()
emb2 = encoder2.encode_graphs(graphs)
time2 = time.time() - start

print(f"Original: {time1:.3f}s")
print(f"Optimized: {time2:.3f}s")
print(f"Speedup: {time1/time2:.2f}×")

if time2 < time1:
    print("✓ Usar configuración optimizada")
else:
    print("✓ Usar configuración original")
```

### Método 3: Heurística Simple

```python
import numpy as np

# Obtener estadísticas del dataset
avg_nodes = np.mean([g.number_of_nodes() for g in graphs])
num_graphs = len(graphs)

# Regla simple
if avg_nodes < 20 and num_graphs < 200:
    print("Recomendación: use_vectorization=False, n_jobs=1")
elif avg_nodes < 50 and num_graphs < 500:
    print("Recomendación: use_vectorization=True, n_jobs=1")
else:
    print("Recomendación: use_vectorization=True, n_jobs=-1")
```

## Configuración Conservadora

Si no estás seguro, usa esta configuración que funciona bien en la mayoría de casos:

```python
encoder = VSGraphEncoder(
    dimension=8192,
    use_vectorization=True,
    n_jobs=4  # Usar 4 cores (buen balance)
)
```

Esta configuración:
- ✓ Funciona bien en grafos medianos a grandes
- ✓ No satura todos los cores (bueno para multitasking)
- ✓ Overhead aceptable incluso en datasets pequeños
- ✓ Escalará bien si el dataset crece

## Configuración Agresiva (Máximo Rendimiento)

Para obtener el máximo rendimiento sin importar el overhead:

```python
encoder = VSGraphEncoder(
    dimension=8192,
    use_vectorization=True,
    n_jobs=-1  # Todos los cores
)

evaluator = VSGraphEvaluator(
    encoder,
    n_folds=10,
    n_jobs=-1  # Paralelizar folds también
)

results = evaluator.evaluate(
    graphs, labels, num_classes,
    parallel_folds=True
)
```

**Advertencia:** Esta configuración puede:
- Usar 100% CPU
- Consumir mucha memoria (cada worker duplica el encoder)
- Ser más lenta en datasets muy pequeños

## Resumen Ejecutivo

### Para MUTAG y Grafos Pequeños

```python
# LA VERSIÓN "NO OPTIMIZADA" ES MÁS RÁPIDA
encoder = VSGraphEncoder(
    use_vectorization=False,
    n_jobs=1
)
```

### Para Datasets Grandes

```python
# LA VERSIÓN OPTIMIZADA ES MUCHO MÁS RÁPIDA
encoder = VSGraphEncoder(
    use_vectorization=True,
    n_jobs=-1
)
```

### En Duda

```python
# Usar el script de comparación
python compare_performance.py --dataset TU_DATASET
```

El script te dirá exactamente qué configuración es mejor para tu caso específico.
