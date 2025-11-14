// main.js (VERSIÓN FINAL Y ESTABLE - Velocidad de Visualización Ajustada)

// Variables Globales
let simulationRunning = false;
let currentFrameHandle = null;
let L_BOX = 4.0; 
let energyChart, tempChart;
let chartData = {
    steps: [],
    energy: [],
    temperature: []
};
// Variables para control de frecuencia de gráficos y limite de puntos
let chartUpdateCounter = 0; 
const CHART_SAMPLING_RATE = 10; // Actualizar gráfico solo cada 10 pasos
const MAX_POINTS = 500; // Máximo de puntos visibles en el gráfico (Ventana Deslizante)
// *** AJUSTE: Nuevo retardo para suavizar la visualización ***
const VISUALIZATION_DELAY_MS = 30; // Antes 15ms. 30ms = ~33 FPS. Aumentar este valor ralentiza la animación.

// --- 1. FUNCIONES AUXILIARES DE LA INTERFAZ (UI) ---

function logMessage(message) {
    const logContainer = document.getElementById('logContainer');
    const p = document.createElement('p');
    p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.prepend(p);
}

// Lógica de habilitación/deshabilitación de inputs
function toggleInputs(disable) {
    document.getElementById('inputN').disabled = disable;
    document.getElementById('inputT0').disabled = disable;
    document.getElementById('inputDt').disabled = disable;
    document.getElementById('inputBoxSize').disabled = disable;
    document.getElementById('inputIgn').disabled = disable;
    document.getElementById('totalStepsDisplay').textContent = 5000;
}

function updateSimulationDisplay(step, T, E, P, Z, isEquilibrium) {
    document.getElementById('iterationCount').textContent = step;
    document.getElementById('displayT').textContent = T.toFixed(4);
    document.getElementById('displayE').textContent = E.toFixed(4);
    document.getElementById('displayP').textContent = P.toFixed(4);
    document.getElementById('displayZ').textContent = Z.toFixed(4);
    
    const statusEl = document.getElementById('statusDisplay');
    if (isEquilibrium) {
        statusEl.textContent = "Estado: Muestreo";
        statusEl.classList.remove('text-indigo-600');
        statusEl.classList.add('text-green-600');
    } else if (simulationRunning) {
        statusEl.textContent = "Estado: Equilibrio";
    } else {
        statusEl.textContent = "Estado: Detenido";
        statusEl.classList.remove('text-green-600');
        statusEl.classList.add('text-indigo-600');
    }
}

function drawParticles(r, L) {
    const canvas = document.getElementById('mdCanvas');
    const ctx = canvas.getContext('2d');
    const size = canvas.width; 
    const scale = size / L;
    const radius = 0.1 * scale; 

    ctx.fillStyle = '#1f2937'; 
    ctx.fillRect(0, 0, size, size);

    ctx.fillStyle = '#818cf8'; 

    ctx.strokeStyle = '#374151'; 
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, size, size);

    for (const particle of r) {
        let x = particle[0];
        let y = particle[1];

        const px = x * scale;
        const py = y * scale; 

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

function initCharts() {
    // FIX: Configuración del eje X para limitar el número de puntos visibles
    const commonOptions = {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: { 
            x: { 
                display: true,
                type: 'linear',
                // Usar min: 0 por defecto, se actualiza en updateCharts
                min: 0, 
                ticks: {
                    autoSkip: true,
                    maxTicksLimit: 10
                },
                title: {
                    display: true,
                    text: 'Paso de Tiempo'
                }
            } 
        }
    };
    
    const datasetOptions = {
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1
    };

    // 1. Gráfico de Energía
    energyChart = new Chart(document.getElementById('energyChart'), {
        type: 'line',
        data: {
            labels: chartData.steps,
            datasets: [{
                label: 'Energía Total (E)',
                data: chartData.energy,
                borderColor: '#4f46e5',
                ...datasetOptions
            }]
        },
        options: commonOptions
    });

    // 2. Gráfico de Temperatura
    tempChart = new Chart(document.getElementById('tempChart'), {
        type: 'line',
        data: {
            labels: chartData.steps,
            datasets: [{
                label: 'Temperatura (T)',
                data: chartData.temperature,
                borderColor: '#ef4444',
                ...datasetOptions
            }]
        },
        options: commonOptions
    });
}

function updateCharts(data) {
    chartData.steps.push(data.step);
    chartData.energy.push(data.E);
    chartData.temperature.push(data.T);
    
    // Limitar la cantidad de puntos a MAX_POINTS
    if (chartData.steps.length > MAX_POINTS) {
        chartData.steps.shift();
        chartData.energy.shift();
        chartData.temperature.shift();
    }
    
    // CRITICAL FIX: Actualizar dinámicamente el límite MÍNIMO y MÁXIMO del eje X
    if (chartData.steps.length > 0) {
        const minStep = chartData.steps[0];
        const maxStep = chartData.steps[chartData.steps.length - 1];
        
        // Forzar la vista a empezar en el paso mínimo actual
        energyChart.options.scales.x.min = minStep;
        tempChart.options.scales.x.min = minStep;
        
        // Forzar la vista a terminar en el paso máximo actual
        energyChart.options.scales.x.max = maxStep;
        tempChart.options.scales.x.max = maxStep;
    }


    // Forzar la actualización de los gráficos (usar 'none' para evitar problemas de animación)
    energyChart.update('none'); 
    tempChart.update('none');
}


// --- 2. CONEXIÓN CON PYTHON ---

// Función que inicializa el estado en el servidor Flask
function initializeSimulation() {
    logMessage("Recogiendo parámetros de la UI y enviando al servidor...");
    
    // Habilitar los inputs mientras el servidor se inicializa
    toggleInputs(false); 
    document.getElementById('startButton').disabled = true;

    // Limpiar el historial de datos de gráficos
    chartData = { steps: [], energy: [], temperature: [] };
    chartUpdateCounter = 0; 

    // Re-inicializar los gráficos para que el eje X se resetee
    if (energyChart) energyChart.destroy();
    if (tempChart) tempChart.destroy();
    initCharts();


    // Recoger parámetros de la UI
    const params = {
        N: parseInt(document.getElementById('inputN').value),
        NS: 5000, 
        dt: parseFloat(document.getElementById('inputDt').value),
        T_0: parseFloat(document.getElementById('inputT0').value),
        ign: parseInt(document.getElementById('inputIgn').value),
        bs: parseFloat(document.getElementById('inputBoxSize').value)
    };
    
    return fetch('/api/initialize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            logMessage("Error al inicializar: " + data.error);
            alert("Error al inicializar: " + data.error);
            toggleInputs(false); 
            return {success: false, data: data};
        }
        
        // Inicialización Exitosa
        L_BOX = data.L; 
        
        // Habilitar el botón de Inicio y mantener los inputs habilitados
        document.getElementById('startButton').disabled = false;
        document.getElementById('stopButton').disabled = true;
        
        // Dibujo y UI inicial
        drawParticles(data.r, L_BOX); 
        updateSimulationDisplay(0, data.T, data.E, data.P, data.Z, false);
        logMessage("Simulador Inicializado. Listo para empezar.");
        return {success: true, data: data};
    })
    .catch(error => {
        logMessage('Error de conexión con el servidor (Flask no corriendo?): ' + error.message);
        alert('Error de conexión. Asegúrate de que el servidor Flask esté corriendo. Error: ' + error.message);
        toggleInputs(false);
        return {success: false, error: error};
    });
}

function startSimulation() {
    if (simulationRunning) return;
    
    // 1. Re-inicializar el servidor con los parámetros actuales
    initializeSimulation().then(result => {
        if (result.success) {
            // 2. Aquí la lógica de inicio, solo si la inicialización fue exitosa
            logMessage("Simulación iniciada.");
            simulationRunning = true;
            
            // DESHABILITAR INPUTS al iniciar
            toggleInputs(true);
            document.getElementById('startButton').disabled = true;
            document.getElementById('stopButton').disabled = false;
            
            simulationLoop();
        } else {
             // Si initialize falló, ya se registró un error.
             logMessage("Inicio abortado debido a error de inicialización.");
        }
    });
}

function stopSimulation() {
    if (!simulationRunning) return;
    
    simulationRunning = false;
    if (currentFrameHandle) clearTimeout(currentFrameHandle); 
    
    // HABILITAR INPUTS al detener
    toggleInputs(false);

    // Al detener, obtener promedios
    fetch('/api/averages', { method: 'GET' })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logMessage("Error al obtener promedios: " + data.error);
            return;
        }
        logMessage(`Simulación Detenida. Promedios: T=${data.avg_T.toFixed(4)}, E=${data.avg_E.toFixed(4)}, P=${data.avg_P.toFixed(4)}`);
        document.getElementById('statusDisplay').textContent = "Estado: Detenido (Promedios calculados)";
    })
    .catch(error => {
        logMessage('Error de red al obtener promedios: ' + error);
    });

    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    document.getElementById('statusDisplay').textContent = "Estado: Detenido";
}

// Bucle principal de JavaScript 
function simulationLoop() {
    if (!simulationRunning) return;

    fetch('/api/step', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
             throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            logMessage("Error en el paso de simulación: " + data.error);
            stopSimulation();
            return;
        }
        
        // 1. Visualizar partículas
        drawParticles(data.r, L_BOX); 
        
        // 2. Actualizar displays de texto y estado
        updateSimulationDisplay(data.step, data.T, data.E, data.P, data.Z, data.is_equilibrium);

        // MUESTREO DE GRÁFICOS: Actualizar solo cada 10 pasos para reducir RAM/CPU
        chartUpdateCounter++;
        if (chartUpdateCounter % CHART_SAMPLING_RATE === 0) {
            updateCharts(data);
            chartUpdateCounter = 0;
        }
        
        // Si se han completado los pasos totales, detener (Asume 5000 pasos en NS)
        if (data.step >= 5000) { 
            stopSimulation();
            logMessage("Simulación completa.");
        }
        
        // CONTROL DE FRECUENCIA DE CPU (espera de 30ms para ralentizar visualmente)
        currentFrameHandle = setTimeout(simulationLoop, VISUALIZATION_DELAY_MS);
    })
    .catch(error => {
        logMessage('Error de red/API al obtener paso: ' + error.message);
        stopSimulation();
    });
}

// --- 3. EVENT LISTENERS Y SETUP ---

window.onload = () => {
    // Asegurar que el canvas tiene el mismo tamaño
    const canvas = document.getElementById('mdCanvas');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = canvas.parentElement.offsetHeight;

    initializeSimulation(); // Inicialización inicial para cargar el estado por defecto y gráficos
    
    document.getElementById('startButton').addEventListener('click', startSimulation);
    document.getElementById('stopButton').addEventListener('click', stopSimulation);
    
    // Event listener para redimensionar el canvas al cambiar el tamaño de la ventana
    window.addEventListener('resize', () => {
        canvas.width = canvas.parentElement.offsetWidth;
        canvas.height = canvas.parentElement.offsetHeight;
    });
};