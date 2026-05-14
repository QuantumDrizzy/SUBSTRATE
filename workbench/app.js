/* app.js */
const container = document.getElementById('canvas-3d-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const geometry = new THREE.PlaneBufferGeometry(100, 100, 50, 50);
const material = new THREE.MeshPhongMaterial({ color: 0x8a8a8a, wireframe: true, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
const mesh = new THREE.Mesh(geometry, material);
mesh.rotation.x = -Math.PI / 2.5;
scene.add(mesh);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 50, 50);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
camera.position.set(0, 40, 80);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 20;
controls.maxDistance = 200;
controls.maxPolarAngle = Math.PI / 2;

let time = 0;
let activeViz = mesh;

function animate() {
    time += 0.01;
    controls.update(); // Actualizar interacción de usuario
    if (activeViz && activeViz.geometry && activeViz.geometry.attributes.position) {
        const positions = activeViz.geometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i]; const y = positions[i + 1];
            let z = (activeViz.type === 'Points') ? Math.sin(x/5 + time)*2 : Math.sin(x/10+time)*Math.cos(y/10+time)*5 + Math.sin(x/5-time*0.5)*2;
            positions[i+2] = z;
        }
        activeViz.geometry.attributes.position.needsUpdate = true;
    }
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
animate();

// --- INTERACTIVIDAD DE LABORATORIO ---

// Listener para cambios de Tolerancia
document.querySelector('input[value="1.0e-8"]').addEventListener('change', function(e) {
    console.log("Nueva tolerancia:", e.target.value);
    if (window.eel) eel.update_solver_param('tolerance', e.target.value);
});

// Listener para cambio de Algoritmo
document.querySelector('select').addEventListener('change', function(e) {
    console.log("Nuevo algoritmo:", e.target.value);
    if (window.eel) eel.update_solver_param('algorithm', e.target.value);
});

// Botones de Play/Pause
document.getElementById('playBtn').addEventListener('click', () => {
    document.getElementById('playBtn').style.color = 'var(--accent-green)';
    if (window.eel) eel.set_engine_state('RUNNING');
});

document.getElementById('stopBtn').addEventListener('click', () => {
    document.getElementById('playBtn').style.color = '';
    if (window.eel) eel.set_engine_state('STOPPED');
});

let charts = [];

function clearCharts() {
    charts.forEach(c => c.destroy());
    charts = [];
}

function initCharts(config) {
    clearCharts();
    const ctx1 = document.getElementById('chart1').getContext('2d');
    const ctx2 = document.getElementById('chart2').getContext('2d');

    charts.push(new Chart(ctx1, {
        type: config.chart1Type || 'line',
        data: { labels: Array.from({length: 30}, (_, i) => i), datasets: [{ label: config.chart1Label, borderColor: config.colorHex, data: Array.from({length: 30}, () => Math.random()), tension: 0.4, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } }, plugins: { legend: { display: false } } }
    }));

    charts.push(new Chart(ctx2, {
        type: config.chart2Type || 'line',
        data: { labels: Array.from({length: 10}, (_, i) => i), datasets: [{ label: config.chart2Label, backgroundColor: config.colorHex + '44', borderColor: config.colorHex, data: Array.from({length: 10}, () => Math.random()) }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    }));
}

// Configuración de Instrumentos
const instruments = {
    "Quantum Layer (GPU)": { mode: "MESH", color: 0x0071e3, colorHex: "#0071e3", chart1Label: "Q-Strain", chart2Label: "Phase Shift" },
    "Geomagnetic Layer": { mode: "VECTORS", color: 0x27ae60, colorHex: "#27ae60", chart1Label: "Mag Vector", chart2Label: "Radial Var", chart2Type: "polarArea" },
    "Magnon Layer (Bio)": { mode: "SPIKES", color: 0x8e44ad, colorHex: "#8e44ad", chart1Label: "Bio-Potential", chart2Label: "Signal Frequency", chart2Type: "bar" },
    "Quantum Lab (TN)": { mode: "LATTICE", color: 0xd35400, colorHex: "#d35400", chart1Label: "Fidelity", chart2Label: "Bond Entropy", chart2Type: "bar" },
    "Solar Layer (Cycle)": { mode: "SPHERE", color: 0xf1c40f, colorHex: "#f1c40f", chart1Label: "Solar Flux", chart2Label: "Thermal Var" },
    "Cosmological Layer": { mode: "POINTS", color: 0x34495e, colorHex: "#34495e", chart1Label: "Expansion", chart2Label: "Redshift Dist", chart1Type: "scatter" }
};

document.querySelectorAll('.tree-list li').forEach(item => {
    item.addEventListener('click', () => {
        const name = item.textContent.trim();
        const cfg = instruments[name] || instruments["Quantum Layer (GPU)"];
        
        document.querySelectorAll('.tree-list li').forEach(i => i.classList.remove('active'));
        item.classList.add('active');

        // 1. Reconfigurar 3D
        scene.remove(activeViz);
        if (cfg.mode === "SPHERE") {
            const geom = new THREE.SphereBufferGeometry(20, 32, 32);
            const mat = new THREE.MeshPhongMaterial({ color: cfg.color, wireframe: true });
            activeViz = new THREE.Mesh(geom, mat);
        } else if (cfg.mode === "VECTORS") {
            activeViz = new THREE.Group();
            for(let i=0; i<12; i++) {
                const arrow = new THREE.ArrowHelper(new THREE.Vector3(Math.random()-0.5, 1, Math.random()-0.5).normalize(), new THREE.Vector3(Math.random()*40-20, 0, Math.random()*40-20), 10, cfg.color);
                activeViz.add(arrow);
            }
        } else if (cfg.mode === "LATTICE") {
            const pointsGeom = new THREE.BufferGeometry();
            const pos = []; for (let i=-40; i<=40; i+=10) for (let j=-40; j<=40; j+=10) pos.push(i, Math.sin(i/10)*5, j);
            pointsGeom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
            activeViz = new THREE.Points(pointsGeom, new THREE.PointsMaterial({ color: cfg.color, size: 3 }));
        } else {
            material.color.setHex(cfg.color);
            activeViz = mesh;
        }
        scene.add(activeViz);

        // 2. Reconfigurar Gráficas
        document.getElementById('chart-panel-header').textContent = `DATA STREAM: ${name.toUpperCase()}`;
        initCharts(cfg);
    });
});

// Inicializar por defecto
initCharts(instruments["Quantum Layer (GPU)"]);

setInterval(async () => {
    if (window.eel) {
        const data = await eel.get_latest_data()();
        if (data && !data.error) {
            // 1. Actualizar el inspector (Campos filtrados)
            const rows = document.querySelectorAll('.stat-row');
            rows.forEach(row => {
                const label = row.firstElementChild.textContent.trim();
                if (data[label]) {
                    row.lastElementChild.textContent = data[label];
                    if (label === 'q_snr') row.lastElementChild.className = 'val green';
                }
            });

            // 2. Actualizar Coherencia y Diagnóstico
            const coherenceEl = document.querySelector('.stat-value');
            if (coherenceEl && data.coherence) {
                coherenceEl.textContent = `${data.coherence} Coherence`;
            }

            const latencyEl = document.getElementById('engine-latency');
            if (latencyEl && data.compute_ms) {
                latencyEl.textContent = data.compute_ms;
                const ms = parseFloat(data.compute_ms);
                latencyEl.style.color = ms > 50 ? '#ff3b30' : 'var(--accent-green)';
            }

            // 3. ACTUALIZACIÓN CIENTÍFICA: Gráficos de Datos Reales
            if (charts.length >= 2) {
                // Chart 1: Time Series (Strain o Signal)
                const val1 = parseFloat(data.q_strain_h1) || Math.random();
                charts[0].data.datasets[0].data.shift();
                charts[0].data.datasets[0].data.push(val1);
                charts[0].update('none');

                // Chart 2: PSD / Spectrum (Capa B: DSP)
                if (data.spectrum) {
                    charts[1].data.datasets[0].data = data.spectrum.slice(0, 30); // Primeros 30 bins
                    charts[1].update('none');
                }
            }
        }
    }
}, 100); 

// Event Listener para el Toggle de GPU
document.getElementById('gpu-toggle').addEventListener('click', function() {
    this.classList.toggle('active');
    const isActive = this.classList.contains('active');
    if (window.eel) {
        eel.toggle_gpu(isActive);
    }
});

// Event Listener para el botón de Record
let isRecording = false;
document.querySelectorAll('.inspector-panel button').forEach(btn => {
    if (btn.textContent.includes('Record')) {
        btn.id = 'record-btn';
        btn.addEventListener('click', () => {
            isRecording = !isRecording;
            btn.textContent = isRecording ? 'Recording...' : 'Record Session';
            btn.style.color = isRecording ? 'var(--accent-orange)' : '';
            if (window.eel) eel.toggle_recording(isRecording);
        });
    }
});

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
