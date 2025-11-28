// ------------------------------
// Estado global
// ------------------------------

let simulationRunning = false;
let L_BOX = 8.0;
let NS_MAX = 2000;

let plot3dInitialized = false;
let plot3dLayout = null;
let plot3dConfig = null;

// Charts
let tempChart = null;
let energyChart = null;

const chartData = {
    steps: [],
    temperature: [],
    energy: []
};

// ------------------------------
// Helpers DOM
// ------------------------------

function $(id) {
    return document.getElementById(id);
}

function formatNumber(x, decimals = 3) {
    if (x === null || x === undefined || isNaN(x)) return "–";
    return Number(x).toFixed(decimals);
}

// ------------------------------
// Visualización 3D con Plotly
// ------------------------------

function buildBoxWireframe(L) {
    const corners = [
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L]
    ];

    const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ];

    const x = [];
    const y = [];
    const z = [];

    edges.forEach(([i, j]) => {
        x.push(corners[i][0], corners[j][0], null);
        y.push(corners[i][1], corners[j][1], null);
        z.push(corners[i][2], corners[j][2], null);
    });

    return { x, y, z };
}

function drawParticles(r, L) {
    // r = [[x,y,z], [x,y,z], ...]
    if (!r || r.length === 0) return;

    const x = r.map(p => p[0]);
    const y = r.map(p => p[1]);
    const z = r.map(p => p[2]);

    const box = buildBoxWireframe(L);

    const traceBox = {
        x: box.x,
        y: box.y,
        z: box.z,
        mode: "lines",
        type: "scatter3d",
        line: {
            width: 1,
            color: "rgba(148, 163, 184, 0.6)"
        },
        hoverinfo: "skip",
        showlegend: false
    };

    const traceParticles = {
        x,
        y,
        z,
        mode: "markers",
        type: "scatter3d",
        marker: {
            size: 4,
            opacity: 0.9,
            color: "rgba(129, 140, 248, 0.95)"
        },
        name: "Partículas"
    };

    if (!plot3dInitialized) {
        plot3dLayout = {
            title: {
                text: "Caja de partículas (3D)",
                font: { size: 16, color: "#e5e7eb" }
            },
            margin: { l: 0, r: 0, t: 40, b: 0 },
            paper_bgcolor: "#020617",
            plot_bgcolor: "#020617",
            scene: {
                bgcolor: "#020617",
                xaxis: {
                    range: [0, L],
                    title: "X",
                    gridcolor: "#374151",
                    zerolinecolor: "#4b5563",
                    color: "#9ca3af"
                },
                yaxis: {
                    range: [0, L],
                    title: "Y",
                    gridcolor: "#374151",
                    zerolinecolor: "#4b5563",
                    color: "#9ca3af"
                },
                zaxis: {
                    range: [0, L],
                    title: "Z",
                    gridcolor: "#374151",
                    zerolinecolor: "#4b5563",
                    color: "#9ca3af"
                },
                aspectmode: "cube"
            }
        };

        plot3dConfig = {
            responsive: true,
            displaylogo: false
        };

        Plotly.newPlot("plot3d", [traceBox, traceParticles], plot3dLayout, plot3dConfig);
        plot3dInitialized = true;
    } else {
        plot3dLayout.scene.xaxis.range = [0, L];
        plot3dLayout.scene.yaxis.range = [0, L];
        plot3dLayout.scene.zaxis.range = [0, L];

        Plotly.react("plot3d", [traceBox, traceParticles], plot3dLayout, plot3dConfig);
    }
}

// ------------------------------
// Gráficas (Chart.js)
// ------------------------------

function initCharts() {
    const tempCtx = $("tempChart").getContext("2d");
    const energyCtx = $("energyChart").getContext("2d");

    tempChart = new Chart(tempCtx, {
        type: "line",
        data: {
            labels: chartData.steps,
            datasets: [{
                label: "Temperatura",
                data: chartData.temperature,
                borderWidth: 1.5,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: "#e5e7eb" } }
            },
            scales: {
                x: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "#111827" }
                },
                y: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "#111827" }
                }
            }
        }
    });

    energyChart = new Chart(energyCtx, {
        type: "line",
        data: {
            labels: chartData.steps,
            datasets: [{
                label: "Energía total",
                data: chartData.energy,
                borderWidth: 1.5,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: "#e5e7eb" } }
            },
            scales: {
                x: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "#111827" }
                },
                y: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "#111827" }
                }
            }
        }
    });
}

function updateCharts(step, T, E) {
    chartData.steps.push(step);
    chartData.temperature.push(T);
    chartData.energy.push(E);

    if (chartData.steps.length > 300) {
        chartData.steps.shift();
        chartData.temperature.shift();
        chartData.energy.shift();
    }

    if (tempChart && energyChart) {
        tempChart.update("none");
        energyChart.update("none");
    }
}

// ------------------------------
// UI: Estado instantáneo y promedios
// ------------------------------

function updateInstantDisplay(step, T, E, P, Z) {
    $("stepDisplay").textContent = step;
    $("tempDisplay").textContent = formatNumber(T);
    $("energyDisplay").textContent = formatNumber(E);
    $("pressureDisplay").textContent = formatNumber(P);
    $("zDisplay").textContent = formatNumber(Z);
}

function updateAvgDisplay(avg) {
    $("avgTempDisplay").textContent = formatNumber(avg.avg_T);
    $("avgEnergyDisplay").textContent = formatNumber(avg.avg_E);
    $("avgPressureDisplay").textContent = formatNumber(avg.avg_P);
}

// ------------------------------
// Inicializar simulación
// ------------------------------

async function initializeSimulation() {
    const N = parseInt($("particlesInput").value) || 60;
    const NS = parseInt($("stepsInput").value) || 2000;
    const dt = parseFloat($("dtInput").value) || 0.002;
    const T0 = parseFloat($("tempInput").value) || 1.0;
    const ign = parseInt($("ignInput").value) || 200;
    const boxSize = parseFloat($("boxSizeInput").value) || 8.0;

    const payload = {
        N,
        NS,
        dt,
        T_0: T0,
        ign,
        bs: boxSize
    };

    try {
        const res = await fetch("/api/initialize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
            alert(data.error || "Error al inicializar la simulación");
            return;
        }

        // Reset estado
        simulationRunning = true;
        L_BOX = data.L;
        NS_MAX = NS;

        chartData.steps.length = 0;
        chartData.temperature.length = 0;
        chartData.energy.length = 0;

        if (!tempChart || !energyChart) {
            initCharts();
        } else {
            tempChart.data.labels = chartData.steps;
            tempChart.data.datasets[0].data = chartData.temperature;

            energyChart.data.labels = chartData.steps;
            energyChart.data.datasets[0].data = chartData.energy;

            tempChart.update();
            energyChart.update();
        }

        drawParticles(data.r, data.L);
        updateInstantDisplay(0, data.T, data.E, data.P, data.Z);

        simulationLoop();
    } catch (err) {
        console.error(err);
        alert("Error de red al inicializar la simulación");
    }
}

// ------------------------------
// Bucle de simulación
// ------------------------------

async function simulationLoop() {
    let continueLoop = true;

    while (simulationRunning && continueLoop) {
        try {
            const res = await fetch("/api/step", { method: "POST" });
            const data = await res.json();

            if (!res.ok) {
                console.warn("Simulación detenida:", data.error);
                simulationRunning = false;
                break;
            }

            drawParticles(data.r, L_BOX);
            updateInstantDisplay(data.step, data.T, data.E, data.P, data.Z);
            updateCharts(data.step, data.T, data.E);

            if (data.step >= NS_MAX) {
                simulationRunning = false;
                continueLoop = false;
                break;
            }

            // Pequeña pausa para no saturar
            await new Promise(r => setTimeout(r, 10));
        } catch (err) {
            console.error(err);
            simulationRunning = false;
            break;
        }
    }
}

// ------------------------------
// Promedios
// ------------------------------

async function fetchAverages() {
    try {
        const res = await fetch("/api/averages");
        const data = await res.json();
        updateAvgDisplay(data);
    } catch (err) {
        console.error(err);
        alert("Error al obtener promedios");
    }
}

// ------------------------------
// Eventos
// ------------------------------

window.addEventListener("DOMContentLoaded", () => {
    $("startBtn").addEventListener("click", (e) => {
        e.preventDefault();
        if (!simulationRunning) {
            initializeSimulation();
        }
    });

    $("stopBtn").addEventListener("click", (e) => {
        e.preventDefault();
        simulationRunning = false;
    });

    $("avgBtn").addEventListener("click", (e) => {
        e.preventDefault();
        fetchAverages();
    });
});
