# 🚀 Cómo Instalar VS-Graph en Otra PC

## Instalación Rápida

### 1. Clonar el Repositorio

```bash
# Navega a donde quieras el proyecto
cd C:\Projects  # o la carpeta que prefieras

# Clona desde GitHub
git clone https://github.com/gustavovazquez/VSGRAPH-demo.git

# Entra al directorio
cd VSGRAPH-demo
```

### 2. Instalar Dependencias

```bash
# Instala las dependencias de Python
pip install -r requirements.txt
```

**Dependencias requeridas:**
- numpy >= 1.21.0
- scipy >= 1.7.0
- networkx >= 2.6.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

### 3. Verificar Instalación

```bash
# Ejecuta el test rápido
python test_quick.py
```

**Salida esperada:**
```
============================================================
ALL TESTS PASSED ✓
============================================================
```

### 4. Ejecutar Experimentos

```bash
# Test en MUTAG (rápido, ~30 segundos)
python experiments/run_experiments.py --datasets MUTAG --n-repeats 1

# Todos los datasets (más tiempo)
python experiments/run_experiments.py --datasets all
```

---

## Instalación con Entorno Virtual (Recomendado)

```bash
# Clonar
git clone https://github.com/gustavovazquez/VSGRAPH-demo.git
cd VSGRAPH-demo

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar
python test_quick.py
```

---

## Estructura de Archivos

Después de clonar tendrás:

```
VSGRAPH-demo/
├── vsgraph/              # Paquete principal
├── experiments/          # Scripts de experimentos
├── test_quick.py        # Test de verificación
├── README.md            # Documentación completa
├── requirements.txt     # Dependencias
└── setup.py            # Instalación del paquete
```

**Nota:** Los datasets se descargan automáticamente la primera vez que ejecutas un experimento.

---

## Comandos Útiles

```bash
# Ver commits recientes
git log --oneline

# Actualizar desde GitHub (si hay cambios)
git pull origin main

# Ver estado de archivos
git status
```

---

## Troubleshooting

### Si falta alguna librería:

```bash
pip install numpy scipy networkx scikit-learn matplotlib pandas
```

### Si hay errores con matplotlib en Windows:

```bash
pip install --upgrade matplotlib
```

### Para reinstalar todo:

```bash
pip install -r requirements.txt --force-reinstall
```

---

## Uso Rápido

```python
from vsgraph import VSGraphEncoder, PrototypeClassifier, load_tudataset

# Cargar dataset
graphs, labels, num_classes = load_tudataset('MUTAG')

# Crear encoder
encoder = VSGraphEncoder(dimension=8192)

# Codificar
embeddings = encoder.encode_graphs(graphs, verbose=True)

# Clasificar
classifier = PrototypeClassifier(num_classes=num_classes)
classifier.fit(embeddings, labels)
predictions = classifier.predict(embeddings)
```

---

## URL del Repositorio

**GitHub**: https://github.com/gustavovazquez/VSGRAPH-demo

¡Listo para usar en cualquier PC con Python 3.7+!
