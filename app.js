/*
 * DEMO EDUCATIVA: RECONOCIMIENTO FACIAL POR LANDMARKS
 * Captura individual, inferencia múltiple con zonas faciales configurables.
 */

const CONFIG = {
    CAPTURE_DURATION: 5000,
    TARGET_FPS: 15,
    FRAME_INTERVAL: 1000 / 15,
    MAX_LOG_ENTRIES: 5,
    LEFT_EYE_INDEX: 33,
    RIGHT_EYE_INDEX: 263,
    FACIAL_ZONES: {
        eyes: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        nose: [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 141, 125, 237, 44, 274, 354, 461],
        mouth: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
        eyebrows: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        contour: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    }
};

const state = {
    faceMesh: null,
    camera: null,
    prototypes: {},
    isCapturing: false,
    isPredicting: false,
    captureFrames: [],
    lastFrameTime: 0,
    lastInferenceTime: 0,
    activeZones: {eyes: true, nose: true, mouth: true, eyebrows: true, contour: true},
    zoneWeights: {eyes: 2.0, nose: 1.5, mouth: 1.0, eyebrows: 1.2, contour: 0.8},
    currentMode: 'capture', // 'capture' o 'inference'
    fpsFrames: [],
    fpsLastUpdate: 0
};

document.addEventListener('DOMContentLoaded', async () => {
    logEvent('Iniciando aplicación...', 'info');
    await initializeMediaPipe();
    initializeEventListeners();
    initializeModeSelector();
    updateStudentList();
    logEvent('Sistema listo. Captura individual, inferencia múltiple.', 'success');
});

async function initializeMediaPipe() {
    const videoElement = document.getElementById('webcam');
    state.faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    state.faceMesh.setOptions({
        maxNumFaces: state.isCapturing ? 1 : 5,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    state.faceMesh.onResults(onFaceMeshResults);
    state.camera = new Camera(videoElement, {
        onFrame: async () => {
            const now = performance.now();
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

function onFaceMeshResults(results) {
    const startTime = performance.now();
    const canvasElement = document.getElementById('canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const videoElement = document.getElementById('webcam');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        if (state.isCapturing) {
            const landmarks = results.multiFaceLandmarks[0];
            drawFaceMesh(canvasCtx, landmarks);
            handleCapture(landmarks);
        } else if (state.isPredicting) {
            for (let i = 0; i < results.multiFaceLandmarks.length; i++) {
                const landmarks = results.multiFaceLandmarks[i];
                drawFaceMesh(canvasCtx, landmarks);
                handleMultiplePredictions(canvasCtx, landmarks, i);
            }
        } else {
            for (const landmarks of results.multiFaceLandmarks) {
                drawFaceMesh(canvasCtx, landmarks);
            }
        }
    } else if (state.isPredicting) {
        updatePredictionDisplay('--', '--');
    }

    const inferenceTime = (performance.now() - startTime).toFixed(1);
    state.lastInferenceTime = inferenceTime;
    document.getElementById('inference-time').textContent = `${inferenceTime} ms`;
    
    // Calcular FPS
    const now = performance.now();
    state.fpsFrames.push(now);
    state.fpsFrames = state.fpsFrames.filter(t => now - t < 1000);
    
    if (now - state.fpsLastUpdate > 500) {
        const fps = state.fpsFrames.length;
        document.getElementById('fps-display').textContent = fps;
        state.fpsLastUpdate = now;
    }
}

function drawFaceMesh(canvasCtx, landmarks) {
    drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
    // Colores para cada zona
    // Colores armoniosos y notables para cada zona
    const zoneColors = {
        eyes: '#43C6E3',      // Azul celeste
        nose: '#FFB347',      // Amarillo suave
        mouth: '#FF5E7E',     // Rosa intenso
        eyebrows: '#8D5EFF',  // Violeta vibrante
        contour: '#222222'    // Negro profundo para contorno
    };
    // Dibuja landmarks de cada zona con su color
    for (const [zone, indices] of Object.entries(CONFIG.FACIAL_ZONES)) {
        if (!state.activeZones[zone]) continue;
        canvasCtx.save();
        // El contorno se dibuja con mayor grosor y opacidad
        if (zone === 'contour') {
            canvasCtx.strokeStyle = zoneColors[zone];
            canvasCtx.lineWidth = 3.5;
            canvasCtx.globalAlpha = 0.85;
            canvasCtx.beginPath();
            for (const idx of indices) {
                const lm = landmarks[idx];
                if (!lm) continue;
                const x = lm.x * canvasCtx.canvas.width;
                const y = lm.y * canvasCtx.canvas.height;
                canvasCtx.lineTo(x, y);
            }
            canvasCtx.closePath();
            canvasCtx.stroke();
        } else {
            canvasCtx.fillStyle = zoneColors[zone];
            canvasCtx.globalAlpha = 0.95;
            for (const idx of indices) {
                const lm = landmarks[idx];
                if (!lm) continue;
                const x = lm.x * canvasCtx.canvas.width;
                const y = lm.y * canvasCtx.canvas.height;
                canvasCtx.beginPath();
                canvasCtx.arc(x, y, 3.2, 0, 2 * Math.PI);
                canvasCtx.fill();
            }
        }
        canvasCtx.restore();
    }
}

function normalizeLandmarks(landmarks, forCapture = false) {
    const leftEye = landmarks[CONFIG.LEFT_EYE_INDEX];
    const rightEye = landmarks[CONFIG.RIGHT_EYE_INDEX];
    const centerX = (leftEye.x + rightEye.x) / 2;
    const centerY = (leftEye.y + rightEye.y) / 2;
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const ipd = Math.sqrt(dx * dx + dy * dy);

    if (ipd < 1e-6) {
        console.warn('IPD muy pequeña');
        return { vector: landmarks.flatMap(lm => [lm.x, lm.y]), weights: [] };
    }

    // CAPTURA: Siempre guardar TODOS los landmarks, sin filtrar
    if (forCapture) {
        const vector = [], weights = [];
        for (let i = 0; i < landmarks.length; i++) {
            const x = (landmarks[i].x - centerX) / ipd;
            const y = (landmarks[i].y - centerY) / ipd;
            vector.push(x, y);
            weights.push(1.0, 1.0);
        }
        return { vector, weights };
    }

    // INFERENCIA: Generar vector completo pero con pesos según zonas activas
    const indexWeights = new Map();
    
    // Inicializar todos los índices con peso 0
    for (let i = 0; i < landmarks.length; i++) {
        indexWeights.set(i, 0.0);
    }

    // Asignar pesos a las zonas activas
    for (const [zone, isActive] of Object.entries(state.activeZones)) {
        if (isActive && CONFIG.FACIAL_ZONES[zone]) {
            const weight = state.zoneWeights[zone];
            for (const idx of CONFIG.FACIAL_ZONES[zone]) {
                if (!indexWeights.has(idx) || indexWeights.get(idx) < weight) {
                    indexWeights.set(idx, weight);
                }
            }
        }
    }

    // Generar vector completo con todos los landmarks
    const vector = [], weights = [];
    for (let i = 0; i < landmarks.length; i++) {
        const x = (landmarks[i].x - centerX) / ipd;
        const y = (landmarks[i].y - centerY) / ipd;
        const weight = indexWeights.get(i);
        vector.push(x, y);
        weights.push(weight, weight);
    }
    
    return { vector, weights };
}

function handleCapture(landmarks) {
    // Usar forCapture=true para guardar TODOS los landmarks
    const { vector } = normalizeLandmarks(landmarks, true);
    state.captureFrames.push(vector);
    const progress = state.captureFrames.length;
    document.getElementById('capture-progress').textContent = `📸 ${progress} frames`;
}

function startCapture() {
    const nameInput = document.getElementById('student-name');
    const studentName = nameInput.value.trim();
    if (!studentName) {
        alert('Por favor, ingresa el nombre del alumno');
        return;
    }
    state.captureFrames = [];
    state.isCapturing = true;
    state.faceMesh.setOptions({ maxNumFaces: 1 });
    const btnCapture = document.getElementById('btn-capture');
    btnCapture.disabled = true;
    btnCapture.textContent = '⏳ Capturando...';
    logEvent(`Iniciando captura para: ${studentName}`, 'info');
    setTimeout(() => {
        finishCapture(studentName);
        btnCapture.disabled = false;
        btnCapture.textContent = '📸 Capturar 5 segundos';
        nameInput.value = '';
        if (state.isPredicting) state.faceMesh.setOptions({ maxNumFaces: 5 });
    }, CONFIG.CAPTURE_DURATION);
}

function finishCapture(studentName) {
    state.isCapturing = false;
    document.getElementById('capture-progress').textContent = '';
    if (state.captureFrames.length === 0) {
        logEvent('⚠️ No se capturaron frames', 'warning');
        alert('No se detectó ningún rostro durante la captura');
        return;
    }
    const newPrototype = meanVector(state.captureFrames);
    const newSampleCount = state.captureFrames.length;
    if (state.prototypes[studentName]) {
        const oldData = state.prototypes[studentName];
        const oldProto = oldData.vector;
        const oldCount = oldData.sampleCount || 1;
        const totalCount = oldCount + newSampleCount;
        const updatedProto = [];
        for (let i = 0; i < newPrototype.length; i++) {
            updatedProto.push((oldProto[i] * oldCount + newPrototype[i] * newSampleCount) / totalCount);
        }
        state.prototypes[studentName] = { vector: updatedProto, sampleCount: totalCount };
        logEvent(`✅ Actualizado: ${studentName} (${newSampleCount} frames nuevos, total: ${totalCount})`, 'success');
    } else {
        state.prototypes[studentName] = { vector: newPrototype, sampleCount: newSampleCount };
        logEvent(`✅ Registrado: ${studentName} (${newSampleCount} frames)`, 'success');
    }
    updateStudentList();
    state.captureFrames = [];
}

function meanVector(vectors) {
    if (vectors.length === 0) return [];
    const dimension = vectors[0].length;
    const mean = new Array(dimension).fill(0);
    for (const vector of vectors) {
        for (let i = 0; i < dimension; i++) mean[i] += vector[i];
    }
    for (let i = 0; i < dimension; i++) mean[i] /= vectors.length;
    return mean;
}

function handleMultiplePredictions(canvasCtx, landmarks, faceIndex) {
    if (Object.keys(state.prototypes).length === 0) {
        if (faceIndex === 0) updatePredictionDisplay('Sin alumnos', '--');
        return;
    }
    const { vector, weights } = normalizeLandmarks(landmarks);
    const prediction = classifyKNN(vector, weights);
    if (faceIndex === 0) updatePredictionDisplay(prediction.name, prediction.distance.toFixed(3));
    drawPredictionLabel(canvasCtx, landmarks, prediction, faceIndex);
}

function drawPredictionLabel(canvasCtx, landmarks, prediction, faceIndex) {
    const topPoint = landmarks[10];
    const x = topPoint.x * canvasCtx.canvas.width;
    const y = topPoint.y * canvasCtx.canvas.height - 10;
    canvasCtx.font = 'bold 18px Arial';
    canvasCtx.textAlign = 'center';
    const text = prediction.name; // Solo el nombre, sin distancia
    const metrics = canvasCtx.measureText(text);
    const padding = 10;
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    canvasCtx.fillRect(x - metrics.width / 2 - padding, y - 22, metrics.width + padding * 2, 32);
    canvasCtx.fillStyle = prediction.name.includes('Desconocido') ? '#ff4444' : '#4ade80';
    canvasCtx.fillText(text, x, y);
}

function classifyKNN(vector, weights) {
    const k = parseInt(document.getElementById('k-value').value);
    const threshold = parseFloat(document.getElementById('threshold').value);
    const metric = document.getElementById('distance-metric').value;
    const distances = [];
    for (const [name, protoData] of Object.entries(state.prototypes)) {
        const prototype = protoData.vector;
        const dist = metric === 'euclidean' 
            ? weightedEuclideanDistance(vector, prototype, weights)
            : weightedCosineDistance(vector, prototype, weights);
        distances.push({ name, distance: dist });
    }
    distances.sort((a, b) => a.distance - b.distance);
    const neighbors = distances.slice(0, k);
    if (neighbors[0].distance > threshold) {
        return { name: '❓ Desconocido', distance: neighbors[0].distance };
    }
    const votes = {};
    for (const neighbor of neighbors) votes[neighbor.name] = (votes[neighbor.name] || 0) + 1;
    let maxVotes = 0, winner = neighbors[0].name;
    for (const [name, count] of Object.entries(votes)) {
        if (count > maxVotes) {
            maxVotes = count;
            winner = name;
        }
    }
    return { name: winner, distance: neighbors[0].distance };
}

function weightedEuclideanDistance(v1, v2, weights) {
    if (v1.length !== v2.length) {
        console.error('Vectores de diferentes dimensiones');
        return Infinity;
    }
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const diff = v1[i] - v2[i];
        const weight = weights && weights[i] ? weights[i] : 1.0;
        sum += weight * diff * diff;
    }
    return Math.sqrt(sum);
}

function weightedCosineDistance(v1, v2, weights) {
    if (v1.length !== v2.length) {
        console.error('Vectores de diferentes dimensiones');
        return Infinity;
    }
    let dotProduct = 0, norm1 = 0, norm2 = 0;
    for (let i = 0; i < v1.length; i++) {
        const weight = weights && weights[i] ? weights[i] : 1.0;
        dotProduct += weight * v1[i] * v2[i];
        norm1 += weight * v1[i] * v1[i];
        norm2 += weight * v2[i] * v2[i];
    }
    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);
    if (norm1 === 0 || norm2 === 0) return Infinity;
    const cosineSimilarity = dotProduct / (norm1 * norm2);
    return 1 - cosineSimilarity;
}

function updatePredictionDisplay(name, distance) {
    document.querySelector('.prediction-name').textContent = name;
    document.querySelector('.prediction-distance').textContent = `Distancia: ${distance}`;
}

function updateStudentList() {
    const listElement = document.getElementById('student-list');
    const countElement = document.getElementById('student-count');
    const students = Object.keys(state.prototypes);
    countElement.textContent = students.length;
    if (students.length === 0) {
        listElement.innerHTML = '<li style="text-align: center; color: #6c757d;">Ningún alumno registrado</li>';
        return;
    }
    listElement.innerHTML = students.map(name => {
        const sampleCount = state.prototypes[name].sampleCount || 0;
        return `<li><span>👤 ${name} <small>(${sampleCount} muestras)</small></span><button onclick="deleteStudent('${name}')">Eliminar</button></li>`;
    }).join('');
}

function deleteStudent(name) {
    if (confirm(`¿Eliminar a ${name}?`)) {
        delete state.prototypes[name];
        updateStudentList();
        logEvent(`🗑️ Eliminado: ${name}`, 'warning');
    }
}

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

function importData(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const imported = JSON.parse(e.target.result);
            if (typeof imported !== 'object') throw new Error('Formato JSON inválido');
            const overwritten = [];
            for (const [name, data] of Object.entries(imported)) {
                if (state.prototypes[name]) overwritten.push(name);
                state.prototypes[name] = data;
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
    event.target.value = '';
}

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
    }
}

function updateZoneState(zone, isActive) {
    state.activeZones[zone] = isActive;
    logEvent(`Zona ${zone}: ${isActive ? 'activada' : 'desactivada'}`, 'info');
}

function updateZoneWeight(zone, weight) {
    state.zoneWeights[zone] = weight;
    document.getElementById(`weight-${zone}`).textContent = weight.toFixed(1);
}

function resetZones() {
    const defaults = {
        eyes: { active: true, weight: 2.0 },
        nose: { active: true, weight: 1.5 },
        mouth: { active: true, weight: 1.0 },
        eyebrows: { active: true, weight: 1.2 },
        contour: { active: true, weight: 0.8 }
    };
    
    for (const [zone, config] of Object.entries(defaults)) {
        state.activeZones[zone] = config.active;
        state.zoneWeights[zone] = config.weight;
        
        // Actualizar diagrama SVG
        const svgZone = document.querySelector(`.face-zone[data-zone="${zone}"]`);
        if (svgZone) {
            if (config.active) {
                svgZone.classList.add('active');
            } else {
                svgZone.classList.remove('active');
            }
        }
        
        // Actualizar sliders
        document.querySelectorAll(`[data-zone="${zone}"]`).forEach(slider => {
            slider.value = config.weight;
        });
        
        // Actualizar display de peso
        document.getElementById(`weight-${zone}`).textContent = config.weight.toFixed(1);
        
        // Actualizar controles de peso
        const weightControl = document.querySelector(`.weight-control[data-zone="${zone}"]`);
        if (weightControl) {
            if (config.active) {
                weightControl.classList.remove('inactive');
            } else {
                weightControl.classList.add('inactive');
            }
        }
    }
    
    logEvent('🔄 Zonas restauradas a valores por defecto', 'info');
}

// ===== SELECTOR DE MODO =====
function initializeModeSelector() {
    const captureBtn = document.getElementById('mode-capture');
    const inferenceBtn = document.getElementById('mode-inference');
    
    // Iniciar en modo captura
    document.body.classList.add('mode-capture');
    state.currentMode = 'capture';
    
    captureBtn.addEventListener('click', () => {
        if (state.currentMode === 'capture') return;
        
        // Detener predicción si está activa
        if (state.isPredicting) {
            togglePrediction();
        }
        
        state.currentMode = 'capture';
        document.body.classList.remove('mode-inference');
        document.body.classList.add('mode-capture');
        captureBtn.classList.add('active');
        inferenceBtn.classList.remove('active');
        
        // Configurar para captura
        state.faceMesh.setOptions({ maxNumFaces: 1 });
        logEvent('📸 Modo Captura/Entrenamiento activado', 'info');
    });
    
    inferenceBtn.addEventListener('click', () => {
        if (state.currentMode === 'inference') return;
        
        // Verificar que hay alumnos registrados
        if (Object.keys(state.prototypes).length === 0) {
            alert('Registra al menos un alumno antes de usar el modo inferencia');
            return;
        }
        
        // Detener captura si está activa
        if (state.isCapturing) {
            alert('Detén la captura antes de cambiar de modo');
            return;
        }
        
        state.currentMode = 'inference';
        document.body.classList.remove('mode-capture');
        document.body.classList.add('mode-inference');
        inferenceBtn.classList.add('active');
        captureBtn.classList.remove('active');
        
        // Configurar para inferencia múltiple
        state.faceMesh.setOptions({ maxNumFaces: 5 });
        
        // Activar predicción automáticamente
        if (!state.isPredicting) {
            togglePrediction();
        }
        
        logEvent('🔍 Modo Inferencia activado (hasta 5 caras)', 'info');
    });
}

function initializeEventListeners() {
    document.getElementById('btn-capture').addEventListener('click', startCapture);
    document.getElementById('btn-predict').addEventListener('click', togglePrediction);
    document.getElementById('k-value').addEventListener('input', (e) => {
        document.getElementById('k-display').textContent = e.target.value;
    });
    document.getElementById('threshold').addEventListener('input', (e) => {
        document.getElementById('threshold-display').textContent = parseFloat(e.target.value).toFixed(1);
    });
    document.getElementById('btn-export').addEventListener('click', exportData);
    document.getElementById('file-import').addEventListener('change', importData);
    document.getElementById('btn-clear').addEventListener('click', clearAllData);
    
    // Inicializar diagrama facial interactivo
    initializeFaceDiagram();
    
    // Listeners de sliders de peso
    const zones = ['eyes', 'nose', 'mouth', 'eyebrows', 'contour'];
    for (const zone of zones) {
        document.querySelectorAll(`[data-zone="${zone}"]`).forEach(slider => {
            slider.addEventListener('input', (e) => {
                updateZoneWeight(zone, parseFloat(e.target.value));
            });
        });
    }
    
    document.getElementById('btn-reset-zones').addEventListener('click', resetZones);
    document.getElementById('student-name').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') startCapture();
    });
}

function initializeFaceDiagram() {
    const faceZones = document.querySelectorAll('.face-zone');
    
    faceZones.forEach(zone => {
        const zoneName = zone.getAttribute('data-zone');
        
        // Inicializar estado visual
        if (state.activeZones[zoneName]) {
            zone.classList.add('active');
        }
        
        // Click para activar/desactivar zona
        zone.addEventListener('click', () => {
            const isActive = !state.activeZones[zoneName];
            
            // Sincronizar solo las zonas del mismo grupo
            if (zoneName === 'eyes') {
                // Activar/desactivar solo los ojos
                state.activeZones.eyes = isActive;
                
                document.querySelectorAll('.zone-eyes').forEach(z => {
                    if (isActive) z.classList.add('active');
                    else z.classList.remove('active');
                });
                
                const weightControl = document.querySelector(`.weight-control[data-zone="eyes"]`);
                if (weightControl) {
                    if (isActive) {
                        weightControl.classList.remove('inactive');
                    } else {
                        weightControl.classList.add('inactive');
                    }
                }
                
                logEvent(`Ojos: ${isActive ? 'activados' : 'desactivados'}`, 'info');
                
            } else if (zoneName === 'eyebrows') {
                // Activar/desactivar solo las cejas
                state.activeZones.eyebrows = isActive;
                
                document.querySelectorAll('.zone-eyebrows').forEach(z => {
                    if (isActive) z.classList.add('active');
                    else z.classList.remove('active');
                });
                
                const weightControl = document.querySelector(`.weight-control[data-zone="eyebrows"]`);
                if (weightControl) {
                    if (isActive) {
                        weightControl.classList.remove('inactive');
                    } else {
                        weightControl.classList.add('inactive');
                    }
                }
                
                logEvent(`Cejas: ${isActive ? 'activadas' : 'desactivadas'}`, 'info');
                
            } else {
                // Otras zonas funcionan independientemente
                state.activeZones[zoneName] = isActive;
                
                // Actualizar visual del diagrama
                if (isActive) {
                    zone.classList.add('active');
                } else {
                    zone.classList.remove('active');
                }
                
                // Actualizar control de peso
                const weightControl = document.querySelector(`.weight-control[data-zone="${zoneName}"]`);
                if (weightControl) {
                    if (isActive) {
                        weightControl.classList.remove('inactive');
                    } else {
                        weightControl.classList.add('inactive');
                    }
                }
                
                logEvent(`Zona ${zoneName}: ${isActive ? 'activada' : 'desactivada'}`, 'info');
            }
        });
        
        // Efecto hover
        zone.addEventListener('mouseenter', () => {
            zone.style.opacity = '0.9';
        });
        
        zone.addEventListener('mouseleave', () => {
            zone.style.opacity = '1';
        });
    });
}

function togglePrediction() {
    const btn = document.getElementById('btn-predict');
    if (Object.keys(state.prototypes).length === 0) {
        alert('Registra al menos un alumno antes de predecir');
        return;
    }
    state.isPredicting = !state.isPredicting;
    if (state.isPredicting) {
        state.faceMesh.setOptions({ maxNumFaces: 5 });
        btn.textContent = '⏸️ Detener Predicción';
        btn.classList.add('active');
        logEvent('🔍 Predicción múltiple activada (hasta 5 caras)', 'info');
    } else {
        state.faceMesh.setOptions({ maxNumFaces: 1 });
        btn.textContent = '▶️ Iniciar Predicción';
        btn.classList.remove('active');
        updatePredictionDisplay('--', '--');
        logEvent('⏸️ Predicción detenida', 'info');
    }
}

// Función de log simplificada (sin UI)
function logEvent(message, type = 'info') {
    // Log solo en consola para debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
}

window.deleteStudent = deleteStudent;
