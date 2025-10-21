# 🎓 Demo Educativa: Reconocimiento Facial por Landmarks

## 📋 Descripción

Aplicación web educativa para demostrar conceptos de **reconocimiento facial basado en landmarks** utilizando MediaPipe FaceMesh. Diseñada para sesiones de clase de 50 minutos, permite a los estudiantes comprender cómo funciona el reconocimiento facial sin comprometer la privacidad.

## 🔒 Privacidad Garantizada

**IMPORTANTE:** Esta aplicación:
- ✅ **NO almacena imágenes** de ningún tipo
- ✅ **NO envía datos** a servidores externos
- ✅ Solo procesa **vectores numéricos** (coordenadas x, y de puntos faciales)
- ✅ Todos los datos permanecen **en la memoria del navegador**
- ✅ El JSON exportado contiene únicamente **números** (landmarks normalizados)

## 🎯 Objetivos Pedagógicos

1. **Captura de Landmarks:** Detectar 468 puntos faciales en tiempo real
2. **Normalización:** Aprender a eliminar variaciones de posición y escala
3. **Prototipos:** Calcular representaciones promedio de cada persona
4. **Clasificación k-NN:** Implementar el algoritmo de k vecinos más cercanos
5. **Umbralización:** Distinguir entre reconocido y desconocido

## 🚀 Inicio Rápido

### Requisitos
- Navegador moderno (Chrome, Firefox, Edge)
- Cámara web funcional
- Conexión a internet (para CDN de MediaPipe)

### Instalación
```bash
# Clonar o descargar los archivos
# No requiere instalación de dependencias

# Abrir directamente en navegador
open index.html
# o usar un servidor local:
python -m http.server 8000
# Luego abrir: http://localhost:8000
```

## 📖 Guía de Uso

### 1. Registro de Alumnos
1. Escribir el nombre del alumno
2. Clic en "Capturar 3 segundos"
3. Mantener el rostro visible ante la cámara
4. El sistema captura ~45 frames y calcula el prototipo

### 2. Configuración k-NN
- **k (vecinos):** Número de prototipos más cercanos a considerar (recomendado: 3)
- **Umbral τ:** Distancia máxima para considerar "reconocido" (recomendado: 3.0)
- **Métrica:** Euclídea (geométrica) o Coseno (angular)

### 3. Predicción en Vivo
1. Clic en "Iniciar Predicción"
2. El sistema identifica en tiempo real
3. Muestra nombre y distancia al prototipo más cercano

### 4. Gestión de Datos
- **Exportar JSON:** Guarda prototipos como archivo .json
- **Importar JSON:** Carga prototipos previamente guardados
- **Borrar Todo:** Elimina todos los registros de memoria

## 🧮 Fundamentos Técnicos

### Normalización de Landmarks

```javascript
// 1. Calcular centro (punto medio entre ojos)
centerX = (leftEye.x + rightEye.x) / 2
centerY = (leftEye.y + rightEye.y) / 2

// 2. Calcular escala (distancia interpupilar)
IPD = sqrt((rightEye.x - leftEye.x)² + (rightEye.y - leftEye.y)²)

// 3. Normalizar cada punto
x_norm = (x - centerX) / IPD
y_norm = (y - centerY) / IPD
```

**Beneficios:**
- ✅ Invariante a **traslación** (posición en la imagen)
- ✅ Invariante a **escala** (distancia a la cámara)
- ✅ Robusta ante **movimientos** de cabeza

### Algoritmo k-NN

```javascript
// 1. Calcular distancias a todos los prototipos
distances = []
for cada prototipo:
    dist = euclideanDistance(vector_actual, prototipo)
    distances.push({ nombre, dist })

// 2. Ordenar por distancia (menor a mayor)
distances.sort()

// 3. Tomar k vecinos más cercanos
vecinos = distances[0:k]

// 4. Verificar umbral
if vecinos[0].dist > threshold:
    return "Desconocido"
else:
    return vecinos[0].nombre
```

### Distancia Euclídea vs Coseno

**Distancia Euclídea (L2) - Recomendada:**
```javascript
d = sqrt(Σ(x₁ᵢ - x₂ᵢ)²)
```
- Mide diferencia **geométrica directa**
- Intuitiva para geometría facial
- Efectiva tras normalización

**Distancia Coseno:**
```javascript
d = 1 - (v₁ · v₂) / (||v₁|| ||v₂||)
```
- Mide diferencia **angular**
- Útil cuando magnitud varía
- Menos necesaria tras normalización

## 📊 Valores Recomendados

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **k** | 3 | Balance entre ruido y precisión |
| **Umbral τ** | 3.0 | Testado para landmarks normalizados |
| **Métrica** | Euclídea | Mejor para geometría facial |
| **Duración captura** | 3 seg | ~45 frames = prototipo robusto |
| **FPS** | 15 | Balance rendimiento/precisión |

## 🎨 Estructura del Proyecto

```
CBTis/
├── index.html          # Estructura HTML
├── styles.css          # Estilos y diseño responsivo
├── app.js              # Lógica de la aplicación
└── README.md           # Esta documentación
```

## 🔧 Personalización

### Modificar duración de captura
```javascript
// En app.js, línea ~44
CAPTURE_DURATION: 5000,  // Cambiar a 5 segundos
```

### Ajustar FPS
```javascript
// En app.js, línea ~45
TARGET_FPS: 20,          // Aumentar a 20 FPS
```

### Cambiar landmarks de referencia
```javascript
// En app.js, líneas ~49-50
LEFT_EYE_INDEX: 33,      // Usar otro punto
RIGHT_EYE_INDEX: 263,    // Usar otro punto
```

## 🐛 Solución de Problemas

### La cámara no funciona
- Verificar permisos del navegador
- Probar en Chrome/Firefox (mejor compatibilidad)
- Usar HTTPS o localhost

### Predicciones incorrectas
- Aumentar duración de captura (más frames)
- Ajustar umbral τ
- Recapturar prototipos con mejor iluminación

### Bajo rendimiento
- Reducir FPS a 10-12
- Usar navegador actualizado
- Cerrar otras pestañas/aplicaciones

## 📚 Conceptos para Clase

### Temas a discutir (50 min)
1. **¿Por qué landmarks y no imágenes?** (5 min)
   - Privacidad, eficiencia, interpretabilidad
   
2. **Importancia de normalización** (10 min)
   - Demostrar con/sin normalización
   
3. **k-NN: teoría y práctica** (15 min)
   - Explicar k, distancias, umbral
   
4. **Limitaciones del sistema** (10 min)
   - Iluminación, oclusiones, gemelos
   
5. **Ética y privacidad** (10 min)
   - Debate sobre uso de reconocimiento facial

### Actividades sugeridas
- 🎭 Probar con/sin accesorios (gafas, gorra)
- 👥 Comparar hermanos/amigos parecidos
- 📊 Graficar distribución de distancias
- 🔬 Experimentar con diferentes k y τ

## 🌐 Recursos Adicionales

- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [k-NN Algorithm](https://es.wikipedia.org/wiki/K-vecinos_m%C3%A1s_pr%C3%B3ximos)
- [Face Recognition Ethics](https://www.eff.org/es/deeplinks/2018/06/facing-facts-about-facial-recognition)

## 📄 Licencia

Este proyecto es de uso educativo libre. Puedes modificarlo y distribuirlo con fines pedagógicos.

## 👨‍🏫 Créditos

Desarrollado como material educativo para CBTis.
Basado en MediaPipe de Google y algoritmos clásicos de machine learning.

---

**¿Preguntas o mejoras?** Este es un proyecto vivo. Siéntete libre de experimentar y adaptarlo a tus necesidades educativas.
