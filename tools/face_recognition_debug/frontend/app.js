/**
 * Face Recognition Debug Tool - Frontend Application
 */

class FaceRecognitionApp {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.enrolling = false;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fps = 0;

        this.initElements();
        this.bindEvents();
        this.refreshPorts();
        this.loadDatabase();

        // Start FPS counter
        setInterval(() => this.updateFPS(), 1000);
    }

    initElements() {
        // Connection
        this.portSelect = document.getElementById('portSelect');
        this.refreshPortsBtn = document.getElementById('refreshPorts');
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.connectionStatus = document.getElementById('connectionStatus');

        // Video
        this.videoCanvas = document.getElementById('videoCanvas');
        this.ctx = this.videoCanvas.getContext('2d');
        this.overlay = document.getElementById('overlay');
        this.videoInfo = document.getElementById('videoInfo');

        // Enrollment
        this.enrollName = document.getElementById('enrollName');
        this.enrollBtn = document.getElementById('enrollBtn');
        this.cancelEnrollBtn = document.getElementById('cancelEnrollBtn');
        this.enrollmentProgress = document.getElementById('enrollmentProgress');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.enrollModal = document.getElementById('enrollModal');
        this.countdown = document.getElementById('countdown');
        this.modalCancelBtn = document.getElementById('modalCancelBtn');

        // Database
        this.databaseList = document.getElementById('databaseList');

        // Recognition
        this.recognitionResult = document.getElementById('recognitionResult');
    }

    bindEvents() {
        this.refreshPortsBtn.addEventListener('click', () => this.refreshPorts());
        this.connectBtn.addEventListener('click', () => this.connect());
        this.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.enrollBtn.addEventListener('click', () => this.startEnrollment());
        this.cancelEnrollBtn.addEventListener('click', () => this.cancelEnrollment());
        this.modalCancelBtn.addEventListener('click', () => this.cancelEnrollment());

        this.enrollName.addEventListener('input', () => {
            this.enrollBtn.disabled = !this.enrollName.value.trim() || !this.connected;
        });
    }

    async refreshPorts() {
        try {
            const response = await fetch('/api/ports');
            const data = await response.json();

            this.portSelect.innerHTML = '<option value="">Select port...</option>';
            data.ports.forEach(port => {
                const option = document.createElement('option');
                option.value = port.device;
                option.textContent = `${port.device} - ${port.description}`;
                this.portSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to refresh ports:', error);
        }
    }

    async connect() {
        const port = this.portSelect.value;
        if (!port) {
            alert('Please select a serial port');
            return;
        }

        try {
            const response = await fetch('/api/connect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ port, baudrate: 921600 })
            });

            if (response.ok) {
                this.connectWebSocket();
            } else {
                const error = await response.json();
                alert(`Connection failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Connect error:', error);
            alert('Failed to connect');
        }
    }

    async disconnect() {
        try {
            await fetch('/api/disconnect', { method: 'POST' });
            this.disconnectWebSocket();
        } catch (error) {
            console.error('Disconnect error:', error);
        }
    }

    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.setConnected(true);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.setConnected(false);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('Message parse error:', error);
            }
        };

        // Ping to keep connection alive
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    disconnectWebSocket() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.setConnected(false);
    }

    setConnected(connected) {
        this.connected = connected;

        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('.status-text');

        if (connected) {
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = false;
            this.enrollBtn.disabled = !this.enrollName.value.trim();
        } else {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
            this.connectBtn.disabled = false;
            this.disconnectBtn.disabled = true;
            this.enrollBtn.disabled = true;
        }
    }

    handleMessage(message) {
        switch (message.type) {
            case 'face_result':
                this.handleFaceResult(message);
                break;
            case 'enrollment_complete':
                this.handleEnrollmentComplete(message);
                break;
            case 'database_update':
                this.loadDatabase();
                break;
            case 'pong':
                // Connection alive
                break;
        }
    }

    handleFaceResult(data) {
        // Update frame counter
        this.frameCount++;

        // === DEBUG: Print detailed face data to console ===
        if (data.faces && data.faces.length > 0 && this.frameCount % 30 === 1) {
            console.group(`[Frame ${this.frameCount}] Face Detection Debug`);
            console.log('Resolution (image space):', data.resolution);
            data.faces.forEach((face, i) => {
                console.log(`Face[${i}]:`);
                console.log('  bbox (orig):', face.bbox, `[x=${face.bbox[0]}, y=${face.bbox[1]}, w=${face.bbox[2]}, h=${face.bbox[3]}]`);
                console.log('  confidence:', face.confidence);
                if (face.landmarks) {
                    console.log('  landmarks (orig):');
                    const lmNames = ['L_EYE', 'R_EYE', 'NOSE', 'L_MOUTH', 'R_MOUTH'];
                    face.landmarks.forEach((lm, j) => {
                        console.log(`    ${j}:${lmNames[j]}: (${lm[0].toFixed(1)}, ${lm[1].toFixed(1)})`);
                    });
                }
                // Check bbox validity
                const [x, y, w, h] = face.bbox;
                const [imgW, imgH] = data.resolution || [320, 240];
                if (x < 0 || y < 0 || x + w > imgW || y + h > imgH) {
                    console.warn('  ⚠️ bbox OUT OF BOUNDS!');
                }
                if (y === 0) {
                    console.warn('  ⚠️ y=0 detected - possible coordinate bug!');
                }
                // Check if bbox center matches face position expectation
                const cx = x + w / 2;
                const cy = y + h / 2;
                console.log(`  bbox_center: (${cx.toFixed(1)}, ${cy.toFixed(1)}) in ${imgW}x${imgH}`);
                console.log(`  bbox_center%: (${(cx / imgW * 100).toFixed(1)}%, ${(cy / imgH * 100).toFixed(1)}%)`);
            });
            console.groupEnd();
        }
        // === END DEBUG ===

        // Store current data for debug drawing
        this.currentData = data;

        // Draw image first, then draw debug info in callback
        if (data.image) {
            this.drawImageWithDebug(data.image, data);
        }

        // Update resolution info
        if (data.resolution) {
            this.videoInfo.querySelector('span:first-child').textContent =
                `Resolution: ${data.resolution[0]}x${data.resolution[1]}`;
        }

        // Clear previous overlays (we now draw directly on canvas)
        this.overlay.innerHTML = '';

        // Update recognition result panel
        if (data.faces && data.faces.length > 0) {
            this.updateRecognitionResult(data.faces[0]);
        } else {
            this.updateRecognitionResult(null);
        }

        // Update enrollment progress
        if (data.enrollment && data.enrollment.active) {
            this.updateEnrollmentProgress(data.enrollment);
        }
    }

    drawImage(base64) {
        const img = new Image();
        img.onload = () => {
            // Clear canvas
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);

            // Calculate scale to fit canvas while maintaining aspect ratio
            const canvasRatio = this.videoCanvas.width / this.videoCanvas.height;
            const imgRatio = img.width / img.height;

            let drawWidth, drawHeight, offsetX, offsetY;

            if (imgRatio > canvasRatio) {
                drawWidth = this.videoCanvas.width;
                drawHeight = drawWidth / imgRatio;
                offsetX = 0;
                offsetY = (this.videoCanvas.height - drawHeight) / 2;
            } else {
                drawHeight = this.videoCanvas.height;
                drawWidth = drawHeight * imgRatio;
                offsetX = (this.videoCanvas.width - drawWidth) / 2;
                offsetY = 0;
            }

            this.ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

            // Store transform for face box positioning
            this.imageTransform = { offsetX, offsetY, scale: drawWidth / img.width };
        };
        img.src = 'data:image/jpeg;base64,' + base64;
    }

    drawImageWithDebug(base64, data) {
        const img = new Image();
        img.onload = () => {
            // DEBUG MODE: Draw image at original size (scale = 1)
            // This ensures bbox coordinates align perfectly with the image
            const drawWidth = img.width;
            const drawHeight = img.height;
            const offsetX = 0;
            const offsetY = 0;
            const scale = 1.0;

            // Resize canvas to match image size
            this.videoCanvas.width = drawWidth;
            this.videoCanvas.height = drawHeight;

            // Clear canvas
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);

            // Draw Image at original size (no scaling)
            this.ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

            // Store transform
            this.imageTransform = { offsetX, offsetY, scale };

            // Draw debug info
            this.drawDebugInfo(data, img.width, img.height, offsetX, offsetY, scale, drawWidth, drawHeight);
        };
        img.src = 'data:image/jpeg;base64,' + base64;
    }

    drawDebugInfo(data, imgW, imgH, offsetX, offsetY, scale, drawW, drawH) {
        const ctx = this.ctx;

        // Draw image boundary (green dashed line)
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(offsetX, offsetY, drawW, drawH);
        ctx.setLineDash([]);

        // Draw debug text background (smaller)
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(3, 3, 140, 48);

        // Draw debug text (smaller font)
        ctx.fillStyle = '#00ff00';
        ctx.font = '8px monospace';
        ctx.fillText(`JPEG: ${imgW}x${imgH}`, 6, 12);
        ctx.fillText(`Canvas: ${this.videoCanvas.width}x${this.videoCanvas.height}`, 6, 21);
        ctx.fillText(`Draw: ${drawW.toFixed(0)}x${drawH.toFixed(0)} @(${offsetX.toFixed(0)},${offsetY.toFixed(0)})`, 6, 30);
        ctx.fillText(`Scale: ${scale.toFixed(3)}`, 6, 39);

        if (data.resolution) {
            ctx.fillText(`Orig: ${data.resolution[0]}x${data.resolution[1]}`, 6, 48);
        }

        // Draw face boxes directly on canvas
        if (data.faces && data.faces.length > 0) {
            data.faces.forEach((face, idx) => {
                this.drawFaceBoxOnCanvas(face, idx, offsetX, offsetY, scale);
            });
        }
    }

    drawFaceBoxOnCanvas(face, idx, offsetX, offsetY, scale) {
        if (!face.bbox) return;

        const [x, y, w, h] = face.bbox;
        const ctx = this.ctx;

        // Convert to canvas coordinates
        const canvasX = offsetX + x * scale;
        const canvasY = offsetY + y * scale;
        const canvasW = w * scale;
        const canvasH = h * scale;

        // Draw bbox rectangle (thinner line)
        ctx.strokeStyle = face.recognized_name ? '#00ff00' : '#ff6600';
        ctx.lineWidth = 1;
        ctx.strokeRect(canvasX, canvasY, canvasW, canvasH);

        // Draw corner markers for precision check (smaller)
        const cornerSize = 4;
        ctx.fillStyle = '#ff0000';
        // Top-left corner
        ctx.fillRect(canvasX - 1, canvasY - 1, cornerSize, 2);
        ctx.fillRect(canvasX - 1, canvasY - 1, 2, cornerSize);
        // Top-right corner
        ctx.fillRect(canvasX + canvasW - cornerSize + 1, canvasY - 1, cornerSize, 2);
        ctx.fillRect(canvasX + canvasW - 1, canvasY - 1, 2, cornerSize);
        // Bottom-left corner
        ctx.fillRect(canvasX - 1, canvasY + canvasH - 1, cornerSize, 2);
        ctx.fillRect(canvasX - 1, canvasY + canvasH - cornerSize + 1, 2, cornerSize);
        // Bottom-right corner
        ctx.fillRect(canvasX + canvasW - cornerSize + 1, canvasY + canvasH - 1, cornerSize, 2);
        ctx.fillRect(canvasX + canvasW - 1, canvasY + canvasH - cornerSize + 1, 2, cornerSize);

        // Draw bbox info label (smaller background)
        const labelY = canvasY > 30 ? canvasY - 26 : canvasY + canvasH + 2;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(canvasX, labelY, 90, 24);

        // Draw bbox info text (smaller font)
        ctx.fillStyle = '#ffff00';
        ctx.font = '7px monospace';
        ctx.fillText(`bbox[${idx}]: ${x},${y}`, canvasX + 2, labelY + 8);
        // Clamp confidence to [0, 1] for display (model outputs logits that can be >1)
        const confDisplay = Math.min(Math.max(face.confidence, 0), 1.0);
        ctx.fillText(`${w}x${h} ${(confDisplay * 100).toFixed(0)}%`, canvasX + 2, labelY + 17);

        // Draw landmarks if available (smaller)
        if (face.landmarks && face.landmarks.length > 0) {
            ctx.fillStyle = '#00ffff';
            face.landmarks.forEach((lm, i) => {
                const lmX = offsetX + lm[0] * scale;
                const lmY = offsetY + lm[1] * scale;
                ctx.beginPath();
                ctx.arc(lmX, lmY, 2, 0, 2 * Math.PI);
                ctx.fill();
            });

            // Draw landmark info panel (smaller, bottom-left)
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(3, 55, 95, 50);
            ctx.fillStyle = '#00ffff';
            ctx.font = '7px monospace';
            ctx.fillText('Landmarks:', 6, 63);
            const lmNames = ['LE', 'RE', 'NS', 'LM', 'RM'];
            face.landmarks.forEach((lm, i) => {
                ctx.fillText(`${lmNames[i]}:(${Math.round(lm[0])},${Math.round(lm[1])})`, 6, 71 + i * 8);
            });
        }

        // Draw cross at bbox center (smaller)
        const centerX = canvasX + canvasW / 2;
        const centerY = canvasY + canvasH / 2;
        ctx.strokeStyle = '#ff00ff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(centerX - 5, centerY);
        ctx.lineTo(centerX + 5, centerY);
        ctx.moveTo(centerX, centerY - 5);
        ctx.lineTo(centerX, centerY + 5);
        ctx.stroke();
    }

    drawFaceBox(face, resolution) {
        if (!face.bbox || !this.imageTransform) return;

        const [x, y, w, h] = face.bbox;
        const { offsetX, offsetY, scale } = this.imageTransform;

        // Convert to canvas coordinates
        const canvasX = offsetX + x * scale;
        const canvasY = offsetY + y * scale;
        const canvasW = w * scale;
        const canvasH = h * scale;

        // Convert to percentage for overlay
        const percentX = (canvasX / this.videoCanvas.width) * 100;
        const percentY = (canvasY / this.videoCanvas.height) * 100;
        const percentW = (canvasW / this.videoCanvas.width) * 100;
        const percentH = (canvasH / this.videoCanvas.height) * 100;

        // Create face box element
        const box = document.createElement('div');
        box.className = 'face-box' + (face.recognized_name ? '' : ' unknown');
        box.style.left = percentX + '%';
        box.style.top = percentY + '%';
        box.style.width = percentW + '%';
        box.style.height = percentH + '%';

        // Add label
        const label = document.createElement('div');
        label.className = 'face-label';
        if (face.recognized_name) {
            label.textContent = `${face.recognized_name} (${(face.similarity * 100).toFixed(1)}%)`;
        } else {
            const conf = Math.min(Math.max(face.confidence, 0), 1.0);
            label.textContent = `Unknown (${(conf * 100).toFixed(1)}%)`;
        }
        box.appendChild(label);

        this.overlay.appendChild(box);
    }

    updateRecognitionResult(face) {
        const nameEl = this.recognitionResult.querySelector('.result-name');
        const confEl = this.recognitionResult.querySelector('.result-confidence');

        if (!face) {
            nameEl.textContent = '--';
            nameEl.className = 'result-name';
            confEl.textContent = 'Confidence: --';
            return;
        }

        if (face.recognized_name) {
            nameEl.textContent = face.recognized_name;
            nameEl.className = 'result-name recognized';
            confEl.textContent = `Similarity: ${(face.similarity * 100).toFixed(1)}%`;
        } else {
            nameEl.textContent = 'Unknown';
            nameEl.className = 'result-name unknown';
            const conf = Math.min(Math.max(face.confidence, 0), 1.0);
            confEl.textContent = `Detection: ${(conf * 100).toFixed(1)}%`;
        }
    }

    async startEnrollment() {
        const name = this.enrollName.value.trim();
        if (!name) {
            alert('Please enter a name');
            return;
        }

        try {
            const response = await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, duration: 5.0 })
            });

            if (response.ok) {
                this.enrolling = true;
                this.enrollBtn.disabled = true;
                this.cancelEnrollBtn.disabled = false;
                this.enrollmentProgress.style.display = 'block';
                this.enrollModal.style.display = 'flex';

                // Start countdown
                this.startCountdown(5);
            } else {
                const error = await response.json();
                alert(`Enrollment failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Enrollment error:', error);
            alert('Failed to start enrollment');
        }
    }

    startCountdown(seconds) {
        let remaining = seconds;
        this.countdown.textContent = remaining;

        this.countdownInterval = setInterval(() => {
            remaining--;
            this.countdown.textContent = remaining;

            if (remaining <= 0) {
                clearInterval(this.countdownInterval);
            }
        }, 1000);
    }

    async cancelEnrollment() {
        try {
            await fetch('/api/enroll/cancel', { method: 'POST' });
        } catch (error) {
            console.error('Cancel error:', error);
        }

        this.finishEnrollment();
    }

    handleEnrollmentComplete(data) {
        this.finishEnrollment();

        if (data.success) {
            alert(`Successfully enrolled "${data.name}" with ${data.samples} samples`);
            this.enrollName.value = '';
        } else {
            alert(`Enrollment failed for "${data.name}": ${data.error || 'Unknown error'}`);
        }
    }

    finishEnrollment() {
        this.enrolling = false;
        this.enrollBtn.disabled = !this.enrollName.value.trim() || !this.connected;
        this.cancelEnrollBtn.disabled = true;
        this.enrollmentProgress.style.display = 'none';
        this.enrollModal.style.display = 'none';

        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
        }
    }

    updateEnrollmentProgress(enrollment) {
        const progress = ((enrollment.elapsed / enrollment.duration) * 100).toFixed(0);
        this.progressFill.style.width = progress + '%';
        this.progressText.textContent =
            `Collecting samples: ${enrollment.samples}/${enrollment.min_samples} (${enrollment.remaining.toFixed(1)}s remaining)`;
    }

    async loadDatabase() {
        try {
            const response = await fetch('/api/database');
            const data = await response.json();

            if (data.faces.length === 0) {
                this.databaseList.innerHTML = '<p class="empty-message">No faces enrolled</p>';
                return;
            }

            this.databaseList.innerHTML = '';
            data.faces.forEach(face => {
                const item = document.createElement('div');
                item.className = 'database-item';
                item.innerHTML = `
                    <div>
                        <div class="database-item-name">${face.name}</div>
                        <div class="database-item-info">${face.sample_count} samples</div>
                    </div>
                    <button class="btn-danger" onclick="app.deleteFace('${face.name}')">Delete</button>
                `;
                this.databaseList.appendChild(item);
            });
        } catch (error) {
            console.error('Load database error:', error);
        }
    }

    async deleteFace(name) {
        if (!confirm(`Delete face "${name}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/database/${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadDatabase();
            } else {
                const error = await response.json();
                alert(`Delete failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Delete error:', error);
            alert('Failed to delete face');
        }
    }

    updateFPS() {
        this.fps = this.frameCount;
        this.frameCount = 0;
        this.videoInfo.querySelector('span:last-child').textContent = `FPS: ${this.fps}`;
    }
}

// Initialize app
const app = new FaceRecognitionApp();
