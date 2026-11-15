if (typeof AOS !== "undefined") {
    AOS.init({ once: true, offset: 50 });
}

function initHeroGlobeParallax() {
    const globeImg = document.getElementById("hero-globe-img");
    if (!globeImg) return;

    let targetRotX = 0, targetRotY = 0, targetShiftX = 0, targetShiftY = 0;
    let currentRotX = 0, currentRotY = 0, currentShiftX = 0, currentShiftY = 0;

    window.addEventListener("mousemove", (e) => {
        const nx = e.clientX / window.innerWidth - 0.5;
        const ny = e.clientY / window.innerHeight - 0.5;
        targetRotY = nx * 32;
        targetRotX = -ny * 28;
        targetShiftX = nx * -30;
        targetShiftY = ny * -22;
    });

    function animate() {
        currentRotX += (targetRotX - currentRotX) * 0.08;
        currentRotY += (targetRotY - currentRotY) * 0.08;
        currentShiftX += (targetShiftX - currentShiftX) * 0.08;
        currentShiftY += (targetShiftY - currentShiftY) * 0.08;

        globeImg.style.transform =
            `perspective(900px) ` +
            `translate3d(${currentShiftX.toFixed(2)}px, ${currentShiftY.toFixed(2)}px, 0) ` +
            `rotateX(${currentRotX.toFixed(2)}deg) rotateY(${currentRotY.toFixed(2)}deg) scale(1.04)`;

        requestAnimationFrame(animate);
    }
    animate();
}

class TerrainInspector {
    constructor() {
        this.wrapper = document.getElementById("slide-3");
        this.navLink = document.getElementById("nav-lab-link");
        this.currentData = null;
        this.activeLayer = "relief";
        this.autoSpin = true;
        this.earthSatImg = null;
        this.earthSatTex = null;
        this._isSyncingCamera = false;

        this.offscreen = {
            earth: this._makeOffscreenCanvas(),
            target: this._makeOffscreenCanvas(),
        };
        this.hoverCell = { row: 32, col: 32, active: false };

        this.profileChart = null;
        this.vectorChart = null;

        this.viewports = {
            earth: this._createViewport("canvas-earth", true),
            target: this._createViewport("canvas-target", false),
        };

        this._bindCameraSync();
        this._bindEvents();
    }

    _makeOffscreenCanvas() {
        const c = document.createElement("canvas");
        c.width = 64;
        c.height = 64;
        return c;
    }

    _createViewport(containerId, isPrimary) {
        const container = document.getElementById(containerId);
        const scene = new THREE.Scene();

        const width = container.clientWidth || 420;
        const height = container.clientHeight || 315;
        const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
        camera.position.set(0, 56, 68);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.innerHTML = "";
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.07;
        controls.autoRotate = isPrimary;
        controls.autoRotateSpeed = 1.4;
        controls.maxPolarAngle = Math.PI * 0.48;
        controls.minDistance = 35;
        controls.maxDistance = 145;

        controls.addEventListener("start", () => {
            if (this.autoSpin) {
                this.autoSpin = false;
                this._syncSpinState();
            }
        });

        scene.add(new THREE.AmbientLight(0xffffff, 0.68));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.92);
        dirLight.position.set(45, 85, 50);
        scene.add(dirLight);

        const gridHelper = new THREE.GridHelper(64, 16, 0x4c1d95, 0x1e1b4b);
        gridHelper.position.y = -6;
        scene.add(gridHelper);

        const state = { container, scene, camera, renderer, controls, mesh: null };

        const renderLoop = () => {
            requestAnimationFrame(renderLoop);
            if (!this.wrapper.classList.contains("open")) return;
            controls.update();
            renderer.render(scene, camera);
        };
        renderLoop();

        const ro = new ResizeObserver(() => {
            if (!container.clientWidth || !container.clientHeight) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
        ro.observe(container);

        return state;
    }

    _bindCameraSync() {
        const copyCameraState = (src, dst) => {
            if (this._isSyncingCamera) return;
            this._isSyncingCamera = true;
            dst.camera.position.copy(src.camera.position);
            dst.camera.quaternion.copy(src.camera.quaternion);
            dst.controls.target.copy(src.controls.target);
            dst.controls.update();
            this._isSyncingCamera = false;
        };

        this.viewports.earth.controls.addEventListener("change", () => {
            copyCameraState(this.viewports.earth, this.viewports.target);
        });

        this.viewports.target.controls.addEventListener("change", () => {
            copyCameraState(this.viewports.target, this.viewports.earth);
        });
    }

    _resetCameraAngles() {
        this._isSyncingCamera = true;
        ["earth", "target"].forEach((key) => {
            const vp = this.viewports[key];
            vp.camera.position.set(0, 56, 68);
            vp.controls.target.set(0, 0, 0);
            vp.controls.update();
        });
        this._isSyncingCamera = false;
    }

    _syncSpinState() {
        this.viewports.earth.controls.autoRotate = this.autoSpin;
        this.viewports.target.controls.autoRotate = false;
        const btn = document.getElementById("btn-toggle-spin");
        btn.textContent = `3D Spin: ${this.autoSpin ? "ON" : "OFF"}`;
        btn.classList.toggle("active", this.autoSpin);
    }

    _bindEvents() {
        document.getElementById("close-inspector-btn").addEventListener("click", () => this.close());

        document.getElementById("btn-toggle-spin").addEventListener("click", () => {
            this.autoSpin = !this.autoSpin;
            this._syncSpinState();
        });

        document.querySelectorAll(".planet-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                if (!this.currentData) return;
                this.open({
                    earthId: this.currentData.earth.earth_id,
                    planet: btn.dataset.targetPlanet,
                });
            });
        });

        document.querySelectorAll(".layer-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".layer-btn").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                this.activeLayer = btn.dataset.layer;
                if (this.currentData) this._refreshAllLayers();
            });
        });

        ["grid2d-earth", "grid2d-target"].forEach((id) => {
            const canvas = document.getElementById(id);
            canvas.addEventListener("mousemove", (e) => {
                if (!this.currentData) return;
                const rect = canvas.getBoundingClientRect();
                const nx = Math.max(0, Math.min(0.999, (e.clientX - rect.left) / rect.width));
                const ny = Math.max(0, Math.min(0.999, (e.clientY - rect.top) / rect.height));
                const col = Math.floor(nx * 64);
                const row = Math.floor(ny * 64);

                if (this.hoverCell.row !== row || this.hoverCell.col !== col || !this.hoverCell.active) {
                    this.hoverCell = { row, col, active: true };
                    this._paint2DWithCrosshair();
                    this._updatePixelReadouts(row, col);
                    this._renderProfileChart(row);
                }
            });

            canvas.addEventListener("mouseleave", () => {
                if (!this.currentData) return;
                this.hoverCell = { row: 32, col: 32, active: false };
                this._paint2DWithCrosshair();
                this._updatePixelReadouts(32, 32);
                this._renderProfileChart(32);
            });
        });
    }

    async open({ earthId = null, lat = null, lon = null, continent = "ALL", planet = null }) {
        const params = new URLSearchParams();
        if (earthId !== null) {
            params.set("earth_id", earthId);
        } else {
            params.set("lat", lat);
            params.set("lon", lon);
            params.set("continent", continent);
        }
        if (planet && planet !== "ALL") {
            params.set("planet", planet);
        }

        this.wrapper.classList.add("open");
        if (this.navLink) this.navLink.style.display = "inline-block";
        document.getElementById("insp-title").textContent = "Synthesizing 3D & 2D Terrain Telemetry...";

        setTimeout(() => {
            this.wrapper.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 120);

        try {
            const resp = await fetch(`/api/compare?${params.toString()}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            this.currentData = await resp.json();

            this._resetCameraAngles();
            this.autoSpin = true;
            this._syncSpinState();

            this._preloadEarthSatellite();
            this._updateUI();
        } catch (err) {
            console.error("Compare API failed:", err);
            document.getElementById("insp-title").textContent = "Failed to load terrain data";
        }
    }

    close() {
        this.wrapper.classList.remove("open");
        if (this.navLink) this.navLink.style.display = "none";
        document.getElementById("slide-2").scrollIntoView({ behavior: "smooth" });
    }

    _preloadEarthSatellite() {
        const e = this.currentData.earth;
        this.earthSatImg = null;
        this.earthSatTex = null;

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this.earthSatImg = img;
            const tex = new THREE.Texture(img);
            tex.needsUpdate = true;
            this.earthSatTex = tex;
            if (this.activeLayer === "satellite") {
                this._refreshAllLayers();
            }
        };
        img.src = `/api/earth_sat?lat=${e.lat}&lon=${e.lon}`;
    }

    _updateUI() {
        const d = this.currentData;
        const e = d.earth;
        const t = d.target;
        const planetName = t.planet === "moon" ? "Moon" : "Mars";

        document.getElementById("insp-title").textContent =
            `${e.continent} (${e.lat.toFixed(2)}°, ${e.lon.toFixed(2)}°)  ↔  ${planetName} Site #${t.target_id}`;
        document.getElementById("matcher-subtitle").textContent =
            `Earth-to-${planetName} Terrain Matcher`;
        document.getElementById("insp-score").textContent = `${d.score}%`;

        document.getElementById("btn-moon-score").textContent = `${d.sim_moon}%`;
        document.getElementById("btn-mars-score").textContent = `${d.sim_mars}%`;
        document.querySelectorAll(".planet-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.targetPlanet === t.planet);
        });

        document.getElementById("earth-vp-coords").textContent = `${e.lat.toFixed(3)}°, ${e.lon.toFixed(3)}°`;
        document.getElementById("earth-vp-source").textContent = e.terrain.source;

        const targetTag = document.getElementById("target-vp-tag");
        targetTag.textContent = planetName.toUpperCase();
        targetTag.className = `vp-tag target ${t.planet}`;
        document.getElementById("target-vp-coords").textContent = `${t.lat.toFixed(3)}°, ${t.lon.toFixed(3)}°`;
        document.getElementById("target-vp-source").textContent = t.terrain.source;

        document.getElementById("grid2d-earth-title").textContent = `Earth (${e.continent})`;
        document.getElementById("grid2d-target-title").textContent = `${planetName} (#${t.target_id})`;
        document.getElementById("leg-target-name").textContent = planetName;

        this._refreshAllLayers();
        this._renderVectorChart();
    }

    _refreshAllLayers() {
        this._rebuildMeshes();
        this._build2DOffscreenBuffers();
        this._paint2DWithCrosshair();
        this._updatePixelReadouts(this.hoverCell.row, this.hoverCell.col);
        this._renderProfileChart(this.hoverCell.row);
        this._updateNeuralInsight();
    }

    _colorFromValue(norm, mode, planet) {
        const c = new THREE.Color();
        const t = Math.max(0, Math.min(1, norm));

        if (mode === "relief") {
            c.setHSL(0.64 - t * 0.48, 0.88, 0.36 + t * 0.24);
        } else if (mode === "slope") {
            c.setHSL((1.0 - t) * 0.36, 0.86, 0.46);
        } else if (mode === "roughness") {
            c.setHSL(0.76 - t * 0.64, 0.85, 0.42 + t * 0.2);
        } else if (mode === "thermal") {
            c.setHSL(0.66 - t * 0.58, 0.85, 0.48);
        } else {
            if (planet === "moon") {
                const g = 0.24 + t * 0.58;
                c.setRGB(g, g, g * 1.03);
            } else if (planet === "mars") {
                c.setRGB(0.42 + t * 0.46, 0.18 + t * 0.24, 0.1 + t * 0.12);
            } else {
                c.setRGB(0.16 + t * 0.42, 0.3 + t * 0.36, 0.22 + t * 0.28);
            }
        }
        return c;
    }

    _getLayerMatrixAndBounds(siteData) {
        const g = siteData.terrain;
        if (this.activeLayer === "slope") {
            return { mat: g.slope_deg, min: 0, max: 30, unit: "°", label: "Terrain Slope" };
        }
        if (this.activeLayer === "roughness") {
            return { mat: g.roughness, min: 0, max: 22, unit: " m", label: "Micro-Roughness" };
        }
        if (this.activeLayer === "thermal") {
            return { mat: g.thermal, min: 0, max: 1, unit: "", label: "Thermal Swing Index" };
        }
        if (this.activeLayer === "satellite") {
            return { mat: g.soil, min: 0, max: 1, unit: "", label: "Surface Albedo / Aridity" };
        }

        const zMat = g.z_relief;
        let minZ = Infinity, maxZ = -Infinity;
        for (let r = 0; r < 64; r++) {
            for (let c = 0; c < 64; c++) {
                const v = zMat[r][c];
                if (v < minZ) minZ = v;
                if (v > maxZ) maxZ = v;
            }
        }
        return { mat: zMat, min: minZ, max: Math.max(maxZ, minZ + 1), unit: " m", label: "Relative Relief" };
    }

    _buildTerrainMesh(vpState, siteData, planetType) {
        if (vpState.mesh) {
            vpState.scene.remove(vpState.mesh);
            vpState.mesh.geometry.dispose();
            vpState.mesh.material.dispose();
        }

        const zMat = siteData.terrain.z_relief;
        const size = 64;
        const geom = new THREE.PlaneGeometry(52, 52, size - 1, size - 1);
        geom.rotateX(-Math.PI / 2);

        const pos = geom.attributes.position;
        let minZ = Infinity, maxZ = -Infinity;
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const v = zMat[r][c];
                if (v < minZ) minZ = v;
                if (v > maxZ) maxZ = v;
            }
        }
        const spanZ = Math.max(maxZ - minZ, 1.0);
        const verticalScale = Math.min(16.0, Math.max(5.0, spanZ * 0.045));

        const { mat: layerMat, min: lMin, max: lMax } = this._getLayerMatrixAndBounds(siteData);
        const lSpan = Math.max(lMax - lMin, 1e-4);
        const colors = new Float32Array(pos.count * 3);

        for (let i = 0; i < pos.count; i++) {
            const r = Math.floor(i / size);
            const c = i % size;
            const normZ = (zMat[r][c] - minZ) / spanZ;
            pos.setY(i, (normZ - 0.5) * verticalScale);

            const normVal = (layerMat[r][c] - lMin) / lSpan;
            const col = this._colorFromValue(normVal, this.activeLayer, planetType);
            colors[i * 3] = col.r;
            colors[i * 3 + 1] = col.g;
            colors[i * 3 + 2] = col.b;
        }

        geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geom.computeVertexNormals();

        const useSat = planetType === "earth" && this.activeLayer === "satellite" && this.earthSatTex;
        const mat = new THREE.MeshStandardMaterial({
            vertexColors: !useSat,
            map: useSat ? this.earthSatTex : null,
            roughness: 0.82,
            metalness: 0.06,
        });

        const mesh = new THREE.Mesh(geom, mat);
        vpState.scene.add(mesh);
        vpState.mesh = mesh;
    }

    _rebuildMeshes() {
        this._buildTerrainMesh(this.viewports.earth, this.currentData.earth, "earth");
        this._buildTerrainMesh(this.viewports.target, this.currentData.target, this.currentData.target.planet);
    }

    _build2DOffscreenBuffers() {
        ["earth", "target"].forEach((key) => {
            const siteData = this.currentData[key];
            const planetType = key === "earth" ? "earth" : this.currentData.target.planet;
            const off = this.offscreen[key];
            const ctx = off.getContext("2d");

            if (key === "earth" && this.activeLayer === "satellite" && this.earthSatImg) {
                ctx.drawImage(this.earthSatImg, 0, 0, 64, 64);
                return;
            }

            const { mat, min, max } = this._getLayerMatrixAndBounds(siteData);
            const span = Math.max(max - min, 1e-4);
            const imgData = ctx.createImageData(64, 64);

            for (let r = 0; r < 64; r++) {
                for (let c = 0; c < 64; c++) {
                    const norm = (mat[r][c] - min) / span;
                    const col = this._colorFromValue(norm, this.activeLayer, planetType);
                    const idx = (r * 64 + c) * 4;
                    imgData.data[idx] = Math.round(col.r * 255);
                    imgData.data[idx + 1] = Math.round(col.g * 255);
                    imgData.data[idx + 2] = Math.round(col.b * 255);
                    imgData.data[idx + 3] = 255;
                }
            }
            ctx.putImageData(imgData, 0, 0);
        });
    }

    _paint2DWithCrosshair() {
        const { row, col } = this.hoverCell;
        const px = ((col + 0.5) / 64) * 256;
        const py = ((row + 0.5) / 64) * 256;

        ["earth", "target"].forEach((key) => {
            const canvas = document.getElementById(`grid2d-${key}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, 256, 256);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(this.offscreen[key], 0, 0, 256, 256);

            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, 256);
            ctx.moveTo(0, py);
            ctx.lineTo(256, py);
            ctx.stroke();

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(px - 3, py - 3, 6, 6);
        });
    }

    _updatePixelReadouts(row, col) {
        const xKm = ((col / 63) * 5).toFixed(2);
        const yKm = ((row / 63) * 5).toFixed(2);

        ["earth", "target"].forEach((key) => {
            const siteData = this.currentData[key];
            const { mat, unit, label } = this._getLayerMatrixAndBounds(siteData);
            const rawVal = mat[row][col];
            const formatted =
                this.activeLayer === "relief"
                    ? `${rawVal >= 0 ? "+" : ""}${rawVal.toFixed(1)}${unit}`
                    : `${rawVal.toFixed(2)}${unit}`;

            document.getElementById(`grid2d-${key}-metric`).textContent = `${label} (${xKm}, ${yKm} km)`;
            document.getElementById(`grid2d-${key}-val`).textContent = formatted;
        });
    }

    _updateNeuralInsight() {
        const e = this.currentData.earth;
        const t = this.currentData.target;
        const pName = t.planet === "moon" ? "Moon" : "Mars";
        const el = document.getElementById("neural-insight-text");

        if (this.activeLayer === "slope") {
            el.textContent = `Slope comparison: Earth mean ${e.mean_slope}° (max ${e.max_slope}°) vs ${pName} mean ${t.mean_slope}° (max ${t.max_slope}°).`;
        } else if (this.activeLayer === "roughness") {
            el.textContent = `Surface roughness: Earth ${e.mean_roughness} m vs ${pName} ${t.mean_roughness} m — extracted from 3×3 local elevation variance.`;
        } else if (this.activeLayer === "thermal") {
          const hasFallback = e.thermal_swing_idx === 0.5 || t.thermal_swing_idx === 0.5;
          const note = hasFallback
              ? " (0.50 fallback applied due to missing orbital thermal data)"
              : " (if orbital thermal data was unavailable, a neutral 0.50 baseline was used)";
          el.textContent = `Thermophysical regime: Earth ${e.thermal_swing_idx} vs ${pName} ${t.thermal_swing_idx}${note}.`;
        } else if (this.activeLayer === "satellite") {
            el.textContent = `Surface aridity / albedo index: Earth ${e.surface_soil_idx} vs ${pName} ${t.surface_soil_idx} (high values = dry regolith/sand).`;
        } else {
            el.textContent = `Relief geometry: Earth std dev ${e.relief_std} m vs ${pName} ${t.relief_std} m — normalized independent of planetary sea level.`;
        }
    }

    _renderProfileChart(rowIdx = 32) {
        const ctx = document.getElementById("profile-chart").getContext("2d");
        const eInfo = this._getLayerMatrixAndBounds(this.currentData.earth);
        const tInfo = this._getLayerMatrixAndBounds(this.currentData.target);

        const earthRow = eInfo.mat[rowIdx];
        const targetRow = tInfo.mat[rowIdx];
        const labels = earthRow.map((_, idx) => `${Math.round((idx / 63) * 5000)}m`);

        document.getElementById("profile-chart-title").textContent =
            `Row ${rowIdx} Cross-Section (${eInfo.label})`;

        if (this.profileChart) this.profileChart.destroy();

        this.profileChart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: this.currentData.target.planet === "moon" ? "Moon" : "Mars",
                        data: targetRow,
                        borderColor: "#60a5fa",
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.25,
                    },
                    {
                        label: "Earth",
                        data: earthRow,
                        borderColor: "#34d399",
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.25,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        ticks: { color: "#64748b", font: { size: 10 }, maxTicksLimit: 4 },
                        grid: { color: "rgba(255,255,255,0.06)" },
                    },
                },
            },
        });
    }

    _renderVectorChart() {
        const ctx = document.getElementById("vector-chart").getContext("2d");
        const eVec = this.currentData.earth.vec_64.map((v) => Math.abs(v));
        const tVec = this.currentData.target.vec_64.map((v) => Math.abs(v));
        const labels = eVec.map((_, i) => i + 1);

        if (this.vectorChart) this.vectorChart.destroy();

        this.vectorChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "Target",
                        data: tVec,
                        backgroundColor: "rgba(96, 165, 250, 0.75)",
                        barPercentage: 0.85,
                    },
                    {
                        label: "Earth",
                        data: eVec,
                        backgroundColor: "rgba(52, 211, 153, 0.75)",
                        barPercentage: 0.85,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false },
                },
            },
        });
    }
}

function initEarthExplorer(inspector) {
    const mapEl = document.getElementById("earth-map");
    if (!mapEl || typeof L === "undefined") return;

    const continentSel = document.getElementById("filter-continent");
    const planetSel = document.getElementById("filter-planet");
    const matchSel = document.getElementById("filter-match");
    const listContainer = document.getElementById("locations-list");
    const listStatus = document.getElementById("list-status");
    const matchedCountEl = document.getElementById("matched-count");

    const canvasRenderer = L.canvas({ padding: 0.5 });
    const map = L.map("earth-map", {
        center: [18, 12],
        zoom: 2,
        minZoom: 2,
        maxZoom: 12,
        worldCopyJump: true,
        preferCanvas: true,
    });

    map.createPane("landPane");
    map.getPane("landPane").style.zIndex = 200;

    fetch("https://cdn.jsdelivr.net/gh/johan/world.geo.json@master/countries.geo.json")
        .then((res) => res.json())
        .then((geoData) => {
            L.geoJSON(geoData, {
                pane: "landPane",
                interactive: false,
                style: {
                    color: "#4c2889",
                    weight: 0.9,
                    opacity: 0.85,
                    fillColor: "#131030",
                    fillOpacity: 1,
                },
            }).addTo(map);
        })
        .catch(() => {});

    const pointsLayer = L.layerGroup().addTo(map);
    let selectedRing = null;
    let activeEarthId = null;

    function highlightSelection(lat, lon, earthId = null) {
        activeEarthId = earthId;
        if (selectedRing) map.removeLayer(selectedRing);

        selectedRing = L.circleMarker([lat, lon], {
            renderer: canvasRenderer,
            radius: 11,
            color: "#a855f7",
            weight: 2.5,
            fillColor: "#a855f7",
            fillOpacity: 0.28,
        }).addTo(map);

        document.querySelectorAll(".location-item").forEach((card) => {
            const isMatch = earthId !== null && Number(card.dataset.earthId) === earthId;
            card.classList.toggle("active", isMatch);
        });
    }

    function openLocation({ earthId = null, lat = null, lon = null }) {
        if (lat !== null && lon !== null) {
            highlightSelection(lat, lon, earthId);
        }
        inspector.open({
            earthId,
            lat,
            lon,
            continent: continentSel.value,
            planet: planetSel.value,
        });
    }

    map.on("click", (e) => {
        const lat = Number(e.latlng.lat.toFixed(4));
        const lon = Number((((e.latlng.lng + 180) % 360 + 360) % 360 - 180).toFixed(4));
        openLocation({ lat, lon });
    });

    async function loadLocations() {
        listStatus.textContent = "Updating...";
        const params = new URLSearchParams({
            continent: continentSel.value,
            planet: planetSel.value,
            match_type: matchSel.value,
        });

        try {
            const resp = await fetch(`/api/earth_locations?${params.toString()}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            matchedCountEl.textContent = data.total_matched.toLocaleString();
            listStatus.textContent = `Top ${data.list_items.length}`;

            pointsLayer.clearLayers();
            for (const pt of data.map_points) {
                const [eid, lat, lon, score, planetCode] = pt;
                const isMoon = planetCode === 1;
                const color = isMoon ? "#38bdf8" : "#f97316";

                const marker = L.circleMarker([lat, lon], {
                    renderer: canvasRenderer,
                    radius: score >= 95 ? 4.2 : 3.0,
                    color: color,
                    weight: 0,
                    fillColor: color,
                    fillOpacity: score >= 95 ? 0.9 : 0.68,
                });

                marker.bindTooltip(
                    `<strong>${isMoon ? "Moon" : "Mars"} Analog: ${score}%</strong><br/>` +
                    `Earth: ${lat.toFixed(2)}°, ${lon.toFixed(2)}°`,
                    { direction: "top", className: "astra-tooltip" }
                );

                marker.on("click", (ev) => {
                    L.DomEvent.stopPropagation(ev);
                    openLocation({ earthId: eid, lat, lon });
                });

                marker.addTo(pointsLayer);
            }

            listContainer.innerHTML = "";
            if (data.list_items.length === 0) {
                listContainer.innerHTML = `<div class="list-status" style="padding: 12px;">No analogs found for this filter.</div>`;
                return;
            }

            for (const item of data.list_items) {
                const card = document.createElement("div");
                card.className = "location-item";
                card.dataset.earthId = item.earth_id;
                if (activeEarthId === item.earth_id) card.classList.add("active");

                card.innerHTML = `
                    <div class="loc-top">
                        <span class="planet-badge ${item.planet}">${item.planet}</span>
                        <span class="loc-score mono">${item.score}%</span>
                    </div>
                    <div class="loc-coords mono">${item.continent} &bull; ${item.lat.toFixed(2)}°, ${item.lon.toFixed(2)}°</div>
                    <div class="loc-meta mono">
                        <span>Slope: ${item.mean_slope}°</span>
                        <span>Target: ${item.target_lat}°, ${item.target_lon}°</span>
                    </div>
                `;

                card.addEventListener("click", () => {
                    map.flyTo([item.lat, item.lon], 5, { duration: 0.9 });
                    openLocation({
                        earthId: item.earth_id,
                        lat: item.lat,
                        lon: item.lon,
                    });
                });

                listContainer.appendChild(card);
            }
        } catch (err) {
            console.error("Failed to load locations:", err);
            listStatus.textContent = "API Error";
        }
    }

    continentSel.addEventListener("change", loadLocations);
    planetSel.addEventListener("change", loadLocations);
    matchSel.addEventListener("change", loadLocations);

    loadLocations();
}

window.addEventListener("DOMContentLoaded", () => {
    initHeroGlobeParallax();
    const inspector = new TerrainInspector();
    initEarthExplorer(inspector);
});
