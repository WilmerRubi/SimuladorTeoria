// main.js (VERSIÓN 3D FINAL Y ESTABLE)

// Variables Globales de la Simulación
let simulationRunning = false;
let currentFrameHandle = null;
let L_BOX = 4.0; 
let energyChart, tempChart;
let chartData = {
    steps: [],
    energy: [],
    temperature: []
};
let chartUpdateCounter = 0; 
const CHART_SAMPLING_RATE = 10; 
const MAX_POINTS = 500; 
const VISUALIZATION_DELAY_MS = 30; // ~33 FPS

// --- VARIABLES GLOBALES DE THREE.JS ---
let scene, camera, renderer, controls;
let particleMesh; 
const particleRadiusFactor = 0.05; // Radio visible
const dummy = new THREE.Object3D(); 

// --- 1. FUNCIONES AUXILIARES DE LA INTERFAZ (UI) ---

function logMessage(message) {
    const logContainer = document.getElementById('logContainer');
    const p = document.createElement('p');
    p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.prepend(p);
}

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
        statusEl.textContent = "Estado: Muestreo (Equilibrio)";
        statusEl.classList.remove('text-indigo-600');
        statusEl.classList.add('text-green-600');
    } else if (simulationRunning) {
        statusEl.textContent = "Estado: Calentamiento/Equilibrio";
        statusEl.classList.remove('text-green-600');
        statusEl.classList.add('text-indigo-600');
    } else {
        statusEl.textContent = "Estado: Detenido";
        statusEl.classList.remove('text-green-600');
        statusEl.classList.add('text-indigo-600');
    }
}

// --- 2. FUNCIONES DE VISUALIZACIÓN 3D (NUEVO) ---

function init3DScene() {
    const canvasContainer = document.querySelector('.canvas-container');
    const width = canvasContainer.offsetWidth;
    const height = canvasContainer.offsetHeight;

    // 1. Renderer 
    renderer = new THREE.WebGLRenderer({ antialias: true, canvas: document.getElementById('mdCanvas') });
    renderer.setSize(width, height);

    // 2. Escena
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1f2937); 

    // 3. Cámara
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = L_BOX * 2.5; 
    camera.position.y = L_BOX * 0.5;

    // 4. Controles (CRÍTICO: Inicialización de OrbitControls)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; 
    controls.dampingFactor = 0.05;
    controls.target.set(L_BOX / 2, L_BOX / 2, L_BOX / 2); // Apuntar al centro de la caja

    // 5. Luces
    scene.add(new THREE.AmbientLight(0x404040, 5)); 
    const directionalLight = new THREE.DirectionalLight(0xffffff, 3.0); 
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // 6. La Caja de Simulación (Marco de referencia)
    // Se crea mediante updateSimulationBox para permitir reconstruirla cuando cambie L_BOX
    updateSimulationBox(L_BOX);
    
    // Ejes de Coordenadas (DEBUG) - Si ves esto, Three.js funciona
    const axesHelper = new THREE.AxesHelper(L_BOX * 1.5); 
    scene.add(axesHelper);
    
    // Bucle de renderizado para Three.js
    const animate = () => {
        requestAnimationFrame(animate);
        controls.update(); 
        renderer.render(scene, camera);
    };
    animate();

    window.addEventListener('resize', onWindowResize, false);
}

// Reconstruye la caja de simulación y reajusta cámara/controles cuando cambia L
function updateSimulationBox(L) {
    if (!scene) return;

    // El objeto tendrá nombre 'simBox' para identificarlo fácilmente
    const existing = scene.getObjectByName('simBox');
    if (existing) {
        // eliminar y disponer recursos
        if (existing.geometry) existing.geometry.dispose();
        if (existing.material) existing.material.dispose();
        scene.remove(existing);
    }

    const boxGeometry = new THREE.BoxGeometry(L, L, L);
    const boxEdges = new THREE.EdgesGeometry(boxGeometry);
    const boxMaterial = new THREE.LineBasicMaterial({ color: 0x4b5563, linewidth: 2 });
    const boxLine = new THREE.LineSegments(boxEdges, boxMaterial);
    boxLine.name = 'simBox';
    boxLine.position.set(L / 2, L / 2, L / 2);
    scene.add(boxLine);

    // Ajustar cámara para que vea la caja completa
    if (camera) {
        // alejar la cámara proporcionalmente a L
        camera.position.set(L * 0.5, L * 0.5, L * 2.5);
        camera.lookAt(new THREE.Vector3(L / 2, L / 2, L / 2));
    }

    // Reajustar el target de los controles
    if (controls) {
        controls.target.set(L / 2, L / 2, L / 2);
        controls.update();
    }
}

function onWindowResize() {
    const canvasContainer = document.querySelector('.canvas-container');
    const width = canvasContainer.offsetWidth;
    const height = canvasContainer.offsetHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function updateParticles3D(r, L) {
    // r ahora tiene (N, 3) dimensiones, pero la visualización anterior usaba (N, 2)
    // El backend debe enviar SÓLO las 3 coordenadas (x, y, z)
    const N = r.length;
    const particleRadius = L * particleRadiusFactor; 
    // Si el tamaño de caja cambió, reconstruir la caja y ajustar cámara
    const prevL = L_BOX;
    L_BOX = L;
    if (Math.abs(prevL - L_BOX) > 1e-8) {
        updateSimulationBox(L_BOX);
    }

    // Inicializar o ajustar el InstancedMesh
    if (!particleMesh || particleMesh.count !== N) {
        if (particleMesh) {
            scene.remove(particleMesh);
            particleMesh.dispose();
        }

        const particleGeometry = new THREE.SphereGeometry(particleRadius, 16, 16); 
        const particleMaterial = new THREE.MeshPhongMaterial({ color: 0x818cf8 }); 

        particleMesh = new THREE.InstancedMesh(particleGeometry, particleMaterial, N);
        scene.add(particleMesh);
        
        // Actualizar la posición de la caja
        const box = scene.children.find(c => c.type === 'LineSegments');
        if (box) box.position.set(L_BOX / 2, L_BOX / 2, L_BOX / 2);
        
        // Reajustar el target de los controles
        controls.target.set(L_BOX / 2, L_BOX / 2, L_BOX / 2); 
    }
    
    // Actualizar la matriz de cada instancia
    for (let i = 0; i < N; i++) {
        const x = r[i][0];
        const y = r[i][1];
        const z = r[i][2]; // Ahora usamos Z

        dummy.position.set(x, y, z);
        
        dummy.updateMatrix(); 
        particleMesh.setMatrixAt(i, dummy.matrix);
    }

    particleMesh.instanceMatrix.needsUpdate = true;
}

// --- 3. FUNCIONES DE GRÁFICOS (Chart.js) ---

function initCharts() {
    // ... (Mantener la lógica de initCharts y updateCharts igual que tu versión actual) ...
    const commonOptions = {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: { 
            x: { 
                display: true,
                type: 'linear',
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

    if (energyChart) energyChart.destroy();
    if (tempChart) tempChart.destroy();

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
    
    if (chartData.steps.length > MAX_POINTS) {
        chartData.steps.shift();
        chartData.energy.shift();
        chartData.temperature.shift();
    }
    
    if (chartData.steps.length > 0) {
        const minStep = chartData.steps[0];
        const maxStep = chartData.steps[chartData.steps.length - 1];
        
        energyChart.options.scales.x.min = minStep;
        tempChart.options.scales.x.min = minStep;
        
        energyChart.options.scales.x.max = maxStep;
        tempChart.options.scales.x.max = maxStep;
    }

    energyChart.update('none'); 
    tempChart.update('none');
}

// --- 4. CONEXIÓN CON PYTHON ---

function initializeSimulation() {
    logMessage("Recogiendo parámetros de la UI y enviando al servidor...");
    
    toggleInputs(false); 
    document.getElementById('startButton').disabled = true;

    chartData = { steps: [], energy: [], temperature: [] };
    chartUpdateCounter = 0; 
    initCharts();

    const params = {
        N: parseInt(document.getElementById('inputN').value),
        NS: 5000, 
        dt: parseFloat(document.getElementById('inputDt').value),
        T_0: parseFloat(document.getElementById('inputT0').value),
        ign: parseInt(document.getElementById('inputIgn').value),
        bs: parseFloat(document.getElementById('inputBoxSize').value),
        sig: parseFloat(document.getElementById('inputSigma').value),
        eps: parseFloat(document.getElementById('inputEps').value),
        mass: parseFloat(document.getElementById('inputMass').value),
        wall_type: document.getElementById('selectWallType').value,
        restitution: parseFloat(document.getElementById('inputRestitution').value)
    };
    
    return fetch('/api/initialize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => Promise.reject(`HTTP error! status: ${response.status} - ${err.error}`));
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
        
        // L no se asigna aquí para que updateParticles3D detecte el cambio
        // y reconstruya la caja si es necesario.
        updateParticles3D(data.r, data.L);
        
        document.getElementById('startButton').disabled = false;
        document.getElementById('stopButton').disabled = true;
        
        updateSimulationDisplay(0, data.T, data.E, data.P, data.Z, false);
        logMessage(`Simulador Inicializado. N=${data.N}, L=${data.L.toFixed(2)}. Listo para empezar.`);
        return {success: true, data: data};
    })
    .catch(error => {
        const msg = (error && error.message) ? error.message : String(error);
        logMessage('Error de conexión con el servidor: ' + msg);
        // Mostrar mensaje claro en la UI
        const statusEl = document.getElementById('statusDisplay');
        if (statusEl) {
            statusEl.textContent = 'Error: servidor inalcanzable';
            statusEl.classList.remove('text-green-600');
            statusEl.classList.add('text-red-600');
        }
        alert('Error de conexión. Asegúrate de que el servidor Flask esté corriendo. Error: ' + msg);
        // Restaurar controles para permitir reintento
        toggleInputs(false);
        document.getElementById('startButton').disabled = false;
        document.getElementById('stopButton').disabled = true;
        return {success: false, error: error};
    });
}

// --- 4.b Gestión de Partículas (Frontend) ---
function fetchParticles() {
    return fetch('/api/particles', { method: 'GET' })
    .then(resp => resp.json())
    .then(data => {
        if (data.error) {
            logMessage('Error al obtener partículas: ' + data.error);
            return [];
        }
        return data.particles || [];
    })
    .catch(err => {
        logMessage('Error de red al obtener partículas: ' + err);
        return [];
    });
}

function renderParticlesTable() {
    fetchParticles().then(particles => {
        const tbody = document.querySelector('#particlesTable tbody');
        tbody.innerHTML = '';
        particles.forEach(p => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="px-2 py-1">${p.index}</td>
                <td class="px-2 py-1">${p.r.map(x=>x.toFixed(3)).join(', ')}</td>
                <td class="px-2 py-1">${p.v.map(x=>x.toFixed(3)).join(', ')}</td>
                <td class="px-2 py-1">${p.m.toFixed(3)}</td>
                <td class="px-2 py-1"><button data-idx="${p.index}" class="edit-part bg-gray-200 px-2 py-1 rounded mr-1">Editar</button><button data-idx="${p.index}" class="del-part bg-red-200 px-2 py-1 rounded">Borrar</button></td>
            `;
            tbody.appendChild(tr);
        });

        // attach handlers
        document.querySelectorAll('.del-part').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                deleteParticle(idx);
            });
        });
        document.querySelectorAll('.edit-part').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                // simple prompt-based edit for now
                const xr = prompt('Nueva posición x,y,z (coma separada)');
                const vv = prompt('Nueva velocidad vx,vy,vz (coma separada)');
                const mm = prompt('Nueva masa (m)');
                if (xr || vv || mm) {
                    const payload = {};
                    if (xr) payload.r = xr.split(',').map(s=>parseFloat(s.trim()));
                    if (vv) payload.v = vv.split(',').map(s=>parseFloat(s.trim()));
                    if (mm) payload.m = parseFloat(mm);
                    updateParticle(idx, payload).then(()=> renderParticlesTable());
                }
            });
        });
    });
}

function addParticle(p) {
    return fetch('/api/particles', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({particles: [p]}) })
    .then(r => r.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        logMessage('Partícula añadida. N=' + data.N);
        renderParticlesTable();
    })
    .catch(err => logMessage('Error al añadir partícula: ' + err));
}

function updateParticle(idx, payload) {
    return fetch(`/api/particles/${idx}`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) })
    .then(r => r.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        logMessage('Partícula actualizada.');
    })
    .catch(err => logMessage('Error al actualizar partícula: ' + err));
}

function deleteParticle(idx) {
    if (!confirm('Borrar partícula ' + idx + '?')) return;
    fetch(`/api/particles/${idx}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        logMessage('Partícula borrada. N=' + data.N);
        renderParticlesTable();
    })
    .catch(err => logMessage('Error al borrar partícula: ' + err));
}

function importCsvFile(file) {
    const form = new FormData();
    form.append('file', file);
    return fetch('/api/particles/import', { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        logMessage('CSV importado. N=' + data.N);
        // reload particles and visualization
        initializeSimulation();
        renderParticlesTable();
    })
    .catch(err => logMessage('Error al importar CSV: ' + err));
}

function exportParticlesCsv() {
    // Forcing browser to download
    window.location = '/api/particles/export';
}

function startSimulation() {
    if (simulationRunning) return;
    
    initializeSimulation().then(result => {
        if (result.success) {
            logMessage("Simulación iniciada.");
            simulationRunning = true;
            
            toggleInputs(true);
            document.getElementById('startButton').disabled = true;
            document.getElementById('stopButton').disabled = false;
            
            simulationLoop();
        } else {
             logMessage("Inicio abortado.");
        }
    });
}

function stopSimulation() {
    if (!simulationRunning) return;
    
    simulationRunning = false;
    if (currentFrameHandle) clearTimeout(currentFrameHandle); 
    
    toggleInputs(false);

    fetch('/api/averages', { method: 'GET' })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logMessage("Error al obtener promedios: " + data.error);
            return;
        }
        logMessage(`Simulación Detenida. Promedios: T=${data.avg_T.toFixed(4)}, E=${data.avg_E.toFixed(4)}, P=${data.avg_P.toFixed(4)}`);
        // Actualizar el estado con los promedios
        const statusEl = document.getElementById('statusDisplay');
        statusEl.textContent = `Estado: Detenido (T_avg: ${data.avg_T.toFixed(4)})`;
        statusEl.classList.remove('text-green-600', 'text-indigo-600');
        statusEl.classList.add('text-gray-500');

    })
    .catch(error => {
        logMessage('Error de red al obtener promedios: ' + error);
    });

    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    document.getElementById('statusDisplay').textContent = "Estado: Detenido";
    document.getElementById('statusDisplay').classList.add('text-indigo-600');
}

function simulationLoop() {
    if (!simulationRunning) return;

    fetch('/api/step', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
             return response.json().then(err => Promise.reject(`HTTP error! status: ${response.status} - ${err.error}`));
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            logMessage("Error en el paso de simulación: " + data.error);
            stopSimulation();
            return;
        }
        
        // 1. Visualizar partículas (AHORA EN 3D)
        updateParticles3D(data.r, L_BOX); 
        
        // 2. Actualizar displays de texto y estado
        updateSimulationDisplay(data.step, data.T, data.E, data.P, data.Z, data.is_equilibrium);

        // MUESTREO DE GRÁFICOS
        chartUpdateCounter++;
        if (chartUpdateCounter % CHART_SAMPLING_RATE === 0) {
            updateCharts(data);
            chartUpdateCounter = 0;
        }
        
        if (data.step >= 5000) { 
            stopSimulation();
            logMessage("Simulación completa.");
            return;
        }
        
        currentFrameHandle = setTimeout(simulationLoop, VISUALIZATION_DELAY_MS);
    })
    .catch(error => {
        const msg = (error && error.message) ? error.message : String(error);
        logMessage('Error de red/API al obtener paso: ' + msg);
        const statusEl = document.getElementById('statusDisplay');
        if (statusEl) {
            statusEl.textContent = 'Error durante la simulación: ' + msg;
            statusEl.classList.remove('text-indigo-600');
            statusEl.classList.add('text-red-600');
        }
        stopSimulation();
    });
}

// --- 5. EVENT LISTENERS Y SETUP ---

window.onload = () => {
    // Inicializar la escena 3D y los gráficos ANTES de llamar a initializeSimulation
    init3DScene(); 
    initCharts(); 

    // Ajustar el tamaño del canvas 3D (aunque init3DScene ya lo hace)
    const canvas = document.getElementById('mdCanvas');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = canvas.parentElement.offsetHeight;

    initializeSimulation().then(()=> renderParticlesTable()); 
    
    document.getElementById('startButton').addEventListener('click', startSimulation);
    document.getElementById('stopButton').addEventListener('click', stopSimulation);
    document.getElementById('addParticleBtn').addEventListener('click', function(){
        const x = parseFloat(document.getElementById('new_x').value || 0);
        const y = parseFloat(document.getElementById('new_y').value || 0);
        const z = parseFloat(document.getElementById('new_z').value || 0);
        const vx = parseFloat(document.getElementById('new_vx').value || 0);
        const vy = parseFloat(document.getElementById('new_vy').value || 0);
        const vz = parseFloat(document.getElementById('new_vz').value || 0);
        const m = parseFloat(document.getElementById('new_m').value || 1.0);
        addParticle({r: [x,y,z], v: [vx,vy,vz], m: m});
    });

    document.getElementById('importCsvBtn').addEventListener('click', function(){
        const fileInput = document.getElementById('particlesFile');
        if (fileInput.files.length === 0) { alert('Selecciona un archivo CSV primero'); return; }
        importCsvFile(fileInput.files[0]);
    });

    document.getElementById('exportCsvBtn').addEventListener('click', function(){ exportParticlesCsv(); });
    
    window.addEventListener('resize', onWindowResize, false);
};