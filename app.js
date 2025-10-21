/*
 * ========================================================================
 * DEMO EDUCATIVA: RECONOCIMIENTO FACIAL POR LANDMARKS
 * ========================================================================
 * 
 * PRIVACIDAD GARANTIZADA:
 * - Este sistema NO almacena, guarda ni transmite imágenes de ningún tipo.
 * - Solo procesa vectores numéricos (coordenadas x, y de puntos faciales).
 * - Todos los datos permanecen en la memoria del navegador.
 * - El JSON exportado contiene únicamente números (landmarks normalizados).
 * 
 * FUNCIONAMIENTO:
 * 1. MediaPipe FaceMesh detecta 468 puntos faciales (landmarks) en tiempo real.
 * 2. Normalizamos cada vector centrando por el punto medio entre ojos y
 *    escalando por la distancia interpupilar (invariancia a posición/escala).
 * 3. Capturamos ~45 frames durante 3 segundos y calculamos la media como prototipo.
 * 4. Clasificamos con k-NN (k vecinos más cercanos) usando distancia euclídea.
 * 5. Si la distancia mínima supera el umbral τ, predicción = "Desconocido".
 * 
 * JUSTIFICACIÓN DE DISTANCIA EUCLÍDEA:
 * - La distancia euclídea (L2) mide la diferencia geométrica directa entre
 *   puntos correspondientes de dos rostros.
 * - Es intuitiva para geometría facial: cuanto más diferentes sean las posiciones
 *   relativas de ojos, nariz, boca, mayor será la distancia.
 * - Tras normalizar (centrar + escalar), la métrica euclídea es robusta y 
 *   suficiente para esta demo educativa.
 * - Alternativa: distancia coseno mide ángulo (similaridad direccional),
 *   útil cuando la magnitud varía mucho, pero aquí ya normalizamos la escala.
 * 
 * ========================================================================
 */

// ========================================================================
// CONFIGURACIÓN GLOBAL Y ESTADO
// ========================================================================

const CONFIG = {
    CAPTURE_DURATION: 3000,      // 3 segundos de captura
    TARGET_FPS: 15,              // FPS objetivo para captura y predicción
    FRAME_INTERVAL: 1000 / 15,   // ~66ms entre frames
    MAX_LOG_ENTRIES: 5,          // Máximo de entradas en el log
    
    // Índices de landmarks clave en MediaPipe FaceMesh (468 puntos)
    LEFT_EYE_INDEX: 33,          // Ojo izquierdo (del usuario)
    RIGHT_EYE_INDEX: 263,        // Ojo derecho (del usuario)
};

// Estado de la aplicación
const state = {
    faceMesh: null,
    camera: null,
    prototypes: {},              // { "NombreAlumno": [x1, y1, x2, y2, ...] }
    isCapturing: false,
    isPredicting: false,
    captureFrames: [],
    lastFrameTime: 0,
    lastInferenceTime: 0,
};

// ========================================================================
// INICIALIZACIÓN
// ========================================================================

document.addEventListener('DOMContentLoaded', async () => {
    logEvent('Iniciando aplicación...', 'info');
    await initializeMediaPipe();
    initializeEventListeners();
    updateStudentList();
    logEvent('Sistema listo. Configura k-NN y registra alumnos.', 'success');
});

/**
 * Inicializa MediaPipe FaceMesh y la cámara
 */
async function initializeMediaPipe() {
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');

    // Configurar FaceMesh
    state.faceMesh = new FaceMesh({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }
    });

    state.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    state.faceMesh.onResults(onFaceMeshResults);

    // Iniciar cámara
    state.camera = new Camera(videoElement, {
        onFrame: async () => {
            const now = performance.now();
            // Throttling: procesar solo si ha pasado suficiente tiempo
            if (now - state.lastFrameTime >= CONFIG.FRAME_INTERVAL) {
                state.lastFrameTime = now;
                await state.faceMesh.send({ image: videoElement });
            }
        },
        width: 640,
        height: 480
    });

    await state.camera.start();
    logEvent('Cámara y FaceMesh inicializados', 'success');
}

/**
 * Callback cuando MediaPipe detecta landmarks
 */
function onFaceMeshResults(results) {
    const startTime = performance.now();
    
    const canvasElement = document.getElementById('canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const videoElement = document.getElementById('webcam');

    // Ajustar tamaño del canvas
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    // Limpiar canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        // Dibujar malla facial
        drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {
            color: '#C0C0C070',
            lineWidth: 1
        });
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {
            color: '#FF3030',
            lineWidth: 1
        });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {
            color: '#30FF30',
            lineWidth: 1
        });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {
            color: '#E0E0E0',
            lineWidth: 1
        });

        // Procesar según el modo
        if (state.isCapturing) {
            handleCapture(landmarks);
        } else if (state.isPredicting) {
            handlePrediction(landmarks);
        }
    } else {
        // No se detectó rostro
        if (state.isPredicting) {
            updatePredictionDisplay('--', '--');
        }
    }

    // Actualizar tiempo de inferencia
    const inferenceTime = (performance.now() - startTime).toFixed(1);
    state.lastInferenceTime = inferenceTime;
    document.getElementById('inference-time').textContent = `Tiempo: ${inferenceTime} ms`;
}

// ========================================================================
// NORMALIZACIÓN DE LANDMARKS
// ========================================================================

/**
 * Normaliza landmarks faciales para invariancia a traslación y escala.
 * 
 * Proceso:
 * 1. Calcular punto medio entre los ojos (centro de referencia).
 * 2. Trasladar todos los puntos para que el centro esté en (0, 0).
 * 3. Calcular distancia interpupilar (IPD) como escala de referencia.
 * 4. Dividir todas las coordenadas por IPD para normalizar la escala.
 * 
 * @param {Array} landmarks - Array de {x, y, z} de MediaPipe
 * @returns {Array} - Vector plano [x1, y1, x2, y2, ..., xN, yN] normalizado
 */
function normalizeLandmarks(landmarks) {
    // Obtener puntos de referencia (ojos)
    const leftEye = landmarks[CONFIG.LEFT_EYE_INDEX];
    const rightEye = landmarks[CONFIG.RIGHT_EYE_INDEX];

    // Centro: punto medio entre ojos
    const centerX = (leftEye.x + rightEye.x) / 2;
    const centerY = (leftEye.y + rightEye.y) / 2;

    // Distancia interpupilar (IPD) como escala
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const ipd = Math.sqrt(dx * dx + dy * dy);

    // Evitar división por cero
    if (ipd < 1e-6) {
        console.warn('IPD muy pequeña, usando escala por defecto');
        return landmarks.flatMap(lm => [lm.x, lm.y]);
    }

    // Normalizar cada landmark
    const normalized = [];
    for (let i = 0; i < landmarks.length; i++) {
        const x = (landmarks[i].x - centerX) / ipd;
        const y = (landmarks[i].y - centerY) / ipd;
        normalized.push(x, y);
        
        // Opcional: incluir z si se desea (actualmente solo x, y)
        // normalized.push((landmarks[i].z || 0) / ipd);
    }

    return normalized;
}

// ========================================================================
// CAPTURA DE PROTOTIPOS
// ========================================================================

/**
 * Maneja la captura de frames durante el período de 3 segundos
 */
function handleCapture(landmarks) {
    const normalized = normalizeLandmarks(landmarks);
    state.captureFrames.push(normalized);

    // Actualizar progreso
    const progress = state.captureFrames.length;
    document.getElementById('capture-progress').textContent = 
        `📸 Capturando: ${progress} frames`;
}

/**
 * Inicia el proceso de captura de 3 segundos
 */
function startCapture() {
    const nameInput = document.getElementById('student-name');
    const studentName = nameInput.value.trim();

    if (!studentName) {
        alert('Por favor, ingresa el nombre del alumno');
        return;
    }

    // Resetear estado de captura
    state.captureFrames = [];
    state.isCapturing = true;

    // Deshabilitar botón
    const btnCapture = document.getElementById('btn-capture');
    btnCapture.disabled = true;
    btnCapture.textContent = '⏳ Capturando...';

    logEvent(`Iniciando captura para: ${studentName}`, 'info');

    // Finalizar captura después de 3 segundos
    setTimeout(() => {
        finishCapture(studentName);
        btnCapture.disabled = false;
        btnCapture.textContent = '📸 Capturar 3 segundos';
        nameInput.value = '';
    }, CONFIG.CAPTURE_DURATION);
}

/**
 * Finaliza la captura y calcula el prototipo
 */
function finishCapture(studentName) {
    state.isCapturing = false;
    document.getElementById('capture-progress').textContent = '';

    if (state.captureFrames.length === 0) {
        logEvent('⚠️ No se capturaron frames', 'warning');
        alert('No se detectó ningún rostro durante la captura');
        return;
    }

    // Calcular prototipo (media de todos los frames)
    const prototype = meanVector(state.captureFrames);

    // Si el alumno ya existe, actualizar con media ponderada (70% nuevo, 30% viejo)
    if (state.prototypes[studentName]) {
        const oldProto = state.prototypes[studentName];
        const updatedProto = [];
        for (let i = 0; i < prototype.length; i++) {
            updatedProto.push(0.7 * prototype[i] + 0.3 * oldProto[i]);
        }
        state.prototypes[studentName] = updatedProto;
        logEvent(`✅ Actualizado: ${studentName} (${state.captureFrames.length} frames)`, 'success');
    } else {
        state.prototypes[studentName] = prototype;
        logEvent(`✅ Registrado: ${studentName} (${state.captureFrames.length} frames)`, 'success');
    }

    updateStudentList();
    state.captureFrames = [];
}

/**
 * Calcula el vector promedio de un conjunto de vectores
 * @param {Array<Array>} vectors - Array de vectores numéricos
 * @returns {Array} - Vector promedio
 */
function meanVector(vectors) {
    if (vectors.length === 0) return [];

    const dimension = vectors[0].length;
    const mean = new Array(dimension).fill(0);

    for (const vector of vectors) {
        for (let i = 0; i < dimension; i++) {
            mean[i] += vector[i];
        }
    }

    for (let i = 0; i < dimension; i++) {
        mean[i] /= vectors.length;
    }

    return mean;
}

// ========================================================================
// PREDICCIÓN CON k-NN
// ========================================================================

/**
 * Maneja la predicción en tiempo real
 */
function handlePrediction(landmarks) {
    if (Object.keys(state.prototypes).length === 0) {
        updatePredictionDisplay('Sin alumnos', '--');
        return;
    }

    const normalized = normalizeLandmarks(landmarks);
    const prediction = classifyKNN(normalized);

    updatePredictionDisplay(prediction.name, prediction.distance.toFixed(3));
}

/**
 * Clasifica un vector usando k-NN (k vecinos más cercanos)
 * 
 * Algoritmo:
 * 1. Calcular distancia entre el vector de entrada y cada prototipo.
 * 2. Ordenar distancias de menor a mayor.
 * 3. Tomar los k vecinos más cercanos.
 * 4. Si la distancia mínima supera el umbral τ, predecir "Desconocido".
 * 5. De lo contrario, el vecino más cercano es la predicción (k=1 efectivo).
 * 
 * Nota: En este contexto educativo, cada alumno tiene un solo prototipo,
 * por lo que k-NN se reduce a buscar el prototipo más cercano. Sin embargo,
 * k se puede usar para filtrar outliers en versiones más avanzadas.
 * 
 * @param {Array} vector - Vector normalizado a clasificar
 * @returns {Object} - {name: string, distance: number}
 */
function classifyKNN(vector) {
    const k = parseInt(document.getElementById('k-value').value);
    const threshold = parseFloat(document.getElementById('threshold').value);
    const metric = document.getElementById('distance-metric').value;

    // Calcular distancias a todos los prototipos
    const distances = [];
    for (const [name, prototype] of Object.entries(state.prototypes)) {
        const dist = metric === 'euclidean' 
            ? euclideanDistance(vector, prototype)
            : cosineDistance(vector, prototype);
        distances.push({ name, distance: dist });
    }

    // Ordenar por distancia (menor a mayor)
    distances.sort((a, b) => a.distance - b.distance);

    // Tomar los k vecinos más cercanos
    const neighbors = distances.slice(0, k);

    // Verificar umbral con el vecino más cercano
    if (neighbors[0].distance > threshold) {
        return { name: '❓ Desconocido', distance: neighbors[0].distance };
    }

    // Retornar el vecino más cercano (clasificación simple)
    // En una versión más avanzada, se podría votar entre los k vecinos
    return neighbors[0];
}

/**
 * Calcula la distancia euclídea (L2) entre dos vectores
 * @param {Array} v1 - Vector 1
 * @param {Array} v2 - Vector 2
 * @returns {number} - Distancia euclídea
 */
function euclideanDistance(v1, v2) {
    if (v1.length !== v2.length) {
        console.error('Vectores de diferentes dimensiones');
        return Infinity;
    }

    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const diff = v1[i] - v2[i];
        sum += diff * diff;
    }

    return Math.sqrt(sum);
}

/**
 * Calcula la distancia coseno entre dos vectores
 * Nota: Retorna 1 - similaridad para que menor distancia = más similar
 * @param {Array} v1 - Vector 1
 * @param {Array} v2 - Vector 2
 * @returns {number} - Distancia coseno (0 = idénticos, 2 = opuestos)
 */
function cosineDistance(v1, v2) {
    if (v1.length !== v2.length) {
        console.error('Vectores de diferentes dimensiones');
        return Infinity;
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < v1.length; i++) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return Infinity;

    const cosineSimilarity = dotProduct / (norm1 * norm2);
    return 1 - cosineSimilarity; // Convertir a distancia
}

/**
 * Actualiza la visualización de la predicción
 */
function updatePredictionDisplay(name, distance) {
    document.querySelector('.prediction-name').textContent = name;
    document.querySelector('.prediction-distance').textContent = 
        `Distancia: ${distance}`;
}

// ========================================================================
// GESTIÓN DE DATOS
// ========================================================================

/**
 * Actualiza la lista visual de alumnos registrados
 */
function updateStudentList() {
    const listElement = document.getElementById('student-list');
    const countElement = document.getElementById('student-count');

    const students = Object.keys(state.prototypes);
    countElement.textContent = students.length;

    if (students.length === 0) {
        listElement.innerHTML = '<li style="text-align: center; color: #6c757d;">Ningún alumno registrado</li>';
        return;
    }

    listElement.innerHTML = students.map(name => `
        <li>
            <span>👤 ${name}</span>
            <button onclick="deleteStudent('${name}')">Eliminar</button>
        </li>
    `).join('');
}

/**
 * Elimina un alumno de la base de datos
 */
function deleteStudent(name) {
    if (confirm(`¿Eliminar a ${name}?`)) {
        delete state.prototypes[name];
        updateStudentList();
        logEvent(`🗑️ Eliminado: ${name}`, 'warning');
    }
}

/**
 * Exporta los prototipos como JSON
 */
function exportData() {
    if (Object.keys(state.prototypes).length === 0) {
        alert('No hay datos para exportar');
        return;
    }

    const dataStr = JSON.stringify(state.prototypes, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `alumnos_${new Date().toISOString().split('T')[0]}.json`;
    link.click();

    URL.revokeObjectURL(url);
    logEvent('📤 Datos exportados correctamente', 'success');
}

/**
 * Importa prototipos desde JSON
 */
function importData(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const imported = JSON.parse(e.target.result);
            
            // Validar estructura básica
            if (typeof imported !== 'object') {
                throw new Error('Formato JSON inválido');
            }

            // Mezclar con datos existentes
            const overwritten = [];
            for (const [name, vector] of Object.entries(imported)) {
                if (state.prototypes[name]) {
                    overwritten.push(name);
                }
                state.prototypes[name] = vector;
            }

            updateStudentList();
            
            if (overwritten.length > 0) {
                logEvent(`📥 Importado (sobrescrito: ${overwritten.join(', ')})`, 'warning');
            } else {
                logEvent(`📥 Importados ${Object.keys(imported).length} alumnos`, 'success');
            }
        } catch (error) {
            alert('Error al importar archivo: ' + error.message);
            logEvent('❌ Error en importación', 'error');
        }
    };

    reader.readAsText(file);
    event.target.value = ''; // Limpiar input
}

/**
 * Borra todos los prototipos
 */
function clearAllData() {
    if (Object.keys(state.prototypes).length === 0) {
        alert('No hay datos para borrar');
        return;
    }

    if (confirm('⚠️ ¿Borrar TODOS los alumnos registrados? Esta acción no se puede deshacer.')) {
        state.prototypes = {};
        updateStudentList();
        updatePredictionDisplay('--', '--');
        logEvent('🗑️ Todos los datos borrados', 'warning');
        document.getElementById('event-log').innerHTML = '';
    }
}

// ========================================================================
// EVENT LISTENERS
// ========================================================================

function initializeEventListeners() {
    // Captura
    document.getElementById('btn-capture').addEventListener('click', startCapture);

    // Predicción
    document.getElementById('btn-predict').addEventListener('click', togglePrediction);

    // Controles de configuración
    document.getElementById('k-value').addEventListener('input', (e) => {
        document.getElementById('k-display').textContent = e.target.value;
    });

    document.getElementById('threshold').addEventListener('input', (e) => {
        document.getElementById('threshold-display').textContent = 
            parseFloat(e.target.value).toFixed(1);
    });

    // Gestión de datos
    document.getElementById('btn-export').addEventListener('click', exportData);
    document.getElementById('file-import').addEventListener('change', importData);
    document.getElementById('btn-clear').addEventListener('click', clearAllData);

    // Enter en campo de nombre = capturar
    document.getElementById('student-name').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            startCapture();
        }
    });
}

/**
 * Activa/desactiva el modo de predicción en vivo
 */
function togglePrediction() {
    const btn = document.getElementById('btn-predict');
    
    if (Object.keys(state.prototypes).length === 0) {
        alert('Registra al menos un alumno antes de predecir');
        return;
    }

    state.isPredicting = !state.isPredicting;

    if (state.isPredicting) {
        btn.textContent = '⏸️ Detener Predicción';
        btn.classList.add('active');
        logEvent('🔍 Predicción en vivo activada', 'info');
    } else {
        btn.textContent = '▶️ Iniciar Predicción';
        btn.classList.remove('active');
        updatePredictionDisplay('--', '--');
        logEvent('⏸️ Predicción detenida', 'info');
    }
}

// ========================================================================
// UTILIDADES
// ========================================================================

/**
 * Agrega un evento al log (mantiene máximo 5 entradas)
 */
function logEvent(message, type = 'info') {
    const logElement = document.getElementById('event-log');
    const timestamp = new Date().toLocaleTimeString('es-ES');
    
    const entry = document.createElement('div');
    entry.className = `event-${type}`;
    entry.textContent = `[${timestamp}] ${message}`;

    logElement.insertBefore(entry, logElement.firstChild);

    // Mantener solo las últimas N entradas
    while (logElement.children.length > CONFIG.MAX_LOG_ENTRIES) {
        logElement.removeChild(logElement.lastChild);
    }
}

// Hacer función deleteStudent global para el onclick
window.deleteStudent = deleteStudent;
