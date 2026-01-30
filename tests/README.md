# 🧪 Tests Suite

Suite completa de tests unitarios para el proyecto Agentic MLOps.

## 📋 Estructura

```
tests/
├── __init__.py
├── test_temporal_generator.py    # Tests para generación temporal
├── test_drift_monitor.py          # Tests para drift detection
└── README.md                       # Este archivo
```

## 🚀 Ejecutar Tests

### Todos los Tests
```bash
pytest tests/ -v
```

### Tests Específicos
```bash
# Solo TemporalHRGenerator
pytest tests/test_temporal_generator.py -v

# Solo DriftMonitor
pytest tests/test_drift_monitor.py -v
```

### Con Coverage
```bash
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
```

### Tests por Marker
```bash
# Solo tests unitarios
pytest -m unit

# Excluir tests lentos
pytest -m "not slow"
```

## 📊 Cobertura de Tests

### TemporalHRGenerator (test_temporal_generator.py)
- ✅ Inicialización básica y con fecha personalizada
- ✅ Generación de un mes
- ✅ Tasa de retención
- ✅ Tamaño del dataset
- ✅ Secuencia temporal
- ✅ Envejecimiento de empleados
- ✅ Cálculo de attrition
- ✅ Generación de nuevos empleados
- ✅ Columna de período
- ✅ Escenarios (normal, recession, etc.)
- ✅ Preservación de tipos de datos
- ✅ Validación de valores no negativos
- ✅ Independencia de secuencias
- ✅ Casos extremos (dataset pequeño, alta rotación, 0 meses)

**Total: 20 tests**

### DriftMonitor (test_drift_monitor.py)
- ✅ Inicialización básica
- ✅ Identificación de tipos de columnas
- ✅ Cálculo de PSI (sin drift y con drift)
- ✅ Detección de covariate shift
- ✅ Generación de reportes de drift
- ✅ Identificación de top features con drift
- ✅ Clasificación de severidad
- ✅ Casos extremos (DataFrame vacío, una columna, solo numéricas, solo categóricas)

**Total: 25 tests**

## 🎯 Fixtures Disponibles

### test_temporal_generator.py
- `sample_hr_data`: Dataset de 100 empleados con todas las features

### test_drift_monitor.py
- `sample_reference_data`: Datos de referencia (200 registros)
- `sample_current_data_no_drift`: Datos sin drift
- `sample_current_data_with_drift`: Datos con drift significativo

## 📝 Convenciones

### Naming
- Archivos: `test_*.py`
- Clases: `Test*`
- Funciones: `test_*`

### Estructura de Tests
```python
class TestComponentName:
    """Suite de tests para ComponentName."""
    
    def test_feature_basic(self, fixture):
        """Test: Descripción breve."""
        # Arrange
        component = Component()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected
```

### Markers
```python
@pytest.mark.slow
def test_long_running():
    """Test que toma mucho tiempo."""
    pass

@pytest.mark.integration
def test_with_external_service():
    """Test de integración."""
    pass
```

## 🔧 Configuración

### pytest.ini
```ini
[pytest]
testpaths = tests
addopts = -v --tb=short --strict-markers
markers =
    slow: tests lentos
    integration: tests de integración
    unit: tests unitarios
```

### requirements.txt
```
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
```

## 📈 Métricas de Calidad

### Coverage Target
- **Objetivo**: >80% de cobertura
- **Actual**: ~85% (componentes principales)

### Test Execution Time
- **Total**: ~5 segundos
- **Por archivo**: <3 segundos

## 🐛 Debugging Tests

### Ejecutar un test específico
```bash
pytest tests/test_temporal_generator.py::TestTemporalHRGenerator::test_initialization -v
```

### Ver output completo
```bash
pytest tests/ -v -s
```

### Modo debug con pdb
```bash
pytest tests/ --pdb
```

### Ver warnings
```bash
pytest tests/ -v -W all
```

## 🚧 Tests Pendientes

### Alta Prioridad
- [ ] Tests para TemporalValidator
- [ ] Tests de integración end-to-end
- [ ] Tests para train_pipeline_temporal.py

### Media Prioridad
- [ ] Tests de performance
- [ ] Tests de carga
- [ ] Tests de regresión

### Baja Prioridad
- [ ] Tests de UI (Streamlit)
- [ ] Tests de API (FastAPI)

## 💡 Best Practices

### ✅ DO
- Usar fixtures para datos de prueba
- Mantener tests independientes
- Usar nombres descriptivos
- Probar casos extremos
- Mantener tests rápidos

### ❌ DON'T
- Depender de orden de ejecución
- Usar datos de producción
- Hacer tests demasiado complejos
- Ignorar tests fallidos
- Hardcodear valores mágicos

## 🔗 Referencias

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py](https://coverage.readthedocs.io/)

## 📞 Soporte

Para problemas con tests:
1. Verificar que todas las dependencias están instaladas: `pip install -r requirements.txt`
2. Verificar que PYTHONPATH incluye el directorio raíz
3. Revisar logs de ejecución con `-v -s`

---

*Tests mantienen la calidad del código y previenen regresiones*