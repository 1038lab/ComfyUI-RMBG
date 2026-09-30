/**
 * COLOR Widget for ComfyUI
 *
 * This integration script is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * If you incorporate or modify this code, please credit AILab as the original source:
 * https://github.com/1038lab
 */

import { app } from "/scripts/app.js";

const getContrastTextColor = (hexColor) => {
    if (typeof hexColor !== 'string' || !/^#?[0-9a-fA-F]{6}$/.test(hexColor)) {
        return '#cccccc';
    }

    const hex = hexColor.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    return luminance > 0.5 ? '#333333' : '#cccccc';
};

// ── Color Conversion Helpers ──────────────────────────────────────────

function hexToRGB(hex) {
    hex = hex.replace('#', '');
    return {
        r: parseInt(hex.substr(0, 2), 16),
        g: parseInt(hex.substr(2, 2), 16),
        b: parseInt(hex.substr(4, 2), 16),
    };
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}

function rgbToHSV(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0, s = max === 0 ? 0 : d / max, v = max;
    if (d !== 0) {
        switch (max) {
            case r: h = ((g - b) / d + 6) % 6; break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h *= 60;
    }
    return { h, s, v };
}

function hsvToRGB(h, s, v) {
    h = ((h % 360) + 360) % 360;
    const c = v * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = v - c;
    let r, g, b;
    if (h < 60)       { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else              { r = c; g = 0; b = x; }
    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255),
    };
}

function hexToHSV(hex) {
    const { r, g, b } = hexToRGB(hex);
    return rgbToHSV(r, g, b);
}

function hsvToHex(h, s, v) {
    const { r, g, b } = hsvToRGB(h, s, v);
    return rgbToHex(r, g, b);
}

// ── Custom Color Picker ───────────────────────────────────────────────

function openColorPicker(initialColor, clientX, clientY, onApply) {
    // Close any existing picker
    const existing = document.querySelector('.ailab-cp-backdrop');
    if (existing) existing.remove();
    const existingPanel = document.querySelector('.ailab-cp-panel');
    if (existingPanel) existingPanel.remove();

    const SV_SIZE = 180;
    const HUE_H = 14;
    const PAD = 12;
    const GAP = 8;

    let hsv = hexToHSV(initialColor);

    // ── Create DOM elements ──

    const backdrop = document.createElement('div');
    backdrop.className = 'ailab-cp-backdrop';
    Object.assign(backdrop.style, {
        position: 'fixed', top: '0', left: '0',
        width: '100%', height: '100%', zIndex: '99998',
    });

    const panel = document.createElement('div');
    panel.className = 'ailab-cp-panel';
    Object.assign(panel.style, {
        position: 'fixed', zIndex: '99999',
        background: '#1e1e1e', border: '1px solid #444',
        borderRadius: '8px', padding: `${PAD}px`,
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        fontFamily: 'sans-serif', fontSize: '12px', color: '#ccc',
        userSelect: 'none',
    });

    // SV (saturation-value) canvas
    const svCanvas = document.createElement('canvas');
    svCanvas.width = SV_SIZE;
    svCanvas.height = SV_SIZE;
    Object.assign(svCanvas.style, {
        display: 'block', borderRadius: '4px', cursor: 'crosshair',
        marginBottom: `${GAP}px`,
    });

    // Hue bar canvas
    const hueCanvas = document.createElement('canvas');
    hueCanvas.width = SV_SIZE;
    hueCanvas.height = HUE_H;
    Object.assign(hueCanvas.style, {
        display: 'block', borderRadius: '8px', cursor: 'pointer',
        marginBottom: `${GAP}px`,
    });

    // Bottom row: preview swatch + hex input + OK button
    const row = document.createElement('div');
    Object.assign(row.style, {
        display: 'flex', alignItems: 'center', gap: `${GAP}px`,
    });

    const preview = document.createElement('div');
    Object.assign(preview.style, {
        width: '28px', height: '28px', borderRadius: '4px',
        border: '1px solid #555', flexShrink: '0',
    });

    const hexInput = document.createElement('input');
    hexInput.type = 'text';
    Object.assign(hexInput.style, {
        flex: '1', background: '#333', border: '1px solid #555',
        borderRadius: '4px', color: '#eee', padding: '4px 8px',
        fontFamily: 'monospace', fontSize: '13px', outline: 'none',
        width: '80px', minWidth: '0',
    });

    const okBtn = document.createElement('button');
    okBtn.textContent = 'OK';
    Object.assign(okBtn.style, {
        background: '#4a9eff', color: '#fff', border: 'none',
        borderRadius: '4px', padding: '5px 14px', cursor: 'pointer',
        fontSize: '12px', fontWeight: 'bold',
    });

    row.append(preview, hexInput, okBtn);
    panel.append(svCanvas, hueCanvas, row);

    // ── Position panel below click point, clamped to viewport ──

    const panelW = SV_SIZE + PAD * 2;
    const panelH = SV_SIZE + HUE_H + 28 + GAP * 2 + PAD * 2 + 4;

    let px = clientX - panelW / 2;
    let py = clientY + 10;

    if (px + panelW > window.innerWidth - 10) px = window.innerWidth - panelW - 10;
    if (px < 10) px = 10;
    if (py + panelH > window.innerHeight - 10) py = clientY - panelH - 10;
    if (py < 10) py = 10;

    panel.style.left = `${px}px`;
    panel.style.top = `${py}px`;

    // ── Drawing functions ──

    function drawSV() {
        const ctx = svCanvas.getContext('2d');
        const { r, g, b } = hsvToRGB(hsv.h, 1, 1);

        // Base hue fill
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, 0, SV_SIZE, SV_SIZE);

        // White gradient (left → right = low → high saturation)
        const whiteGrad = ctx.createLinearGradient(0, 0, SV_SIZE, 0);
        whiteGrad.addColorStop(0, '#ffffff');
        whiteGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = whiteGrad;
        ctx.fillRect(0, 0, SV_SIZE, SV_SIZE);

        // Black gradient (top → bottom = high → low value)
        const blackGrad = ctx.createLinearGradient(0, 0, 0, SV_SIZE);
        blackGrad.addColorStop(0, 'rgba(0,0,0,0)');
        blackGrad.addColorStop(1, '#000000');
        ctx.fillStyle = blackGrad;
        ctx.fillRect(0, 0, SV_SIZE, SV_SIZE);

        // Crosshair cursor
        const cx = hsv.s * SV_SIZE;
        const cy = (1 - hsv.v) * SV_SIZE;
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    function drawHue() {
        const ctx = hueCanvas.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, SV_SIZE, 0);
        [0, 60, 120, 180, 240, 300, 360].forEach(deg => {
            const { r, g, b } = hsvToRGB(deg, 1, 1);
            grad.addColorStop(deg / 360, `rgb(${r},${g},${b})`);
        });
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, SV_SIZE, HUE_H);

        // Indicator thumb
        const ix = (hsv.h / 360) * SV_SIZE;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(ix - 2, 0, 4, HUE_H);
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 1;
        ctx.strokeRect(ix - 2, 0, 4, HUE_H);
    }

    function updateAll() {
        const hex = hsvToHex(hsv.h, hsv.s, hsv.v);
        preview.style.backgroundColor = hex;
        hexInput.value = hex;
        drawSV();
        drawHue();
    }

    // ── Interaction handlers ──

    let svDrag = false, hueDrag = false;

    function onSV(e) {
        const rect = svCanvas.getBoundingClientRect();
        hsv.s = Math.max(0, Math.min(1, (e.clientX - rect.left) / SV_SIZE));
        hsv.v = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / SV_SIZE));
        updateAll();
    }

    function onHue(e) {
        const rect = hueCanvas.getBoundingClientRect();
        hsv.h = Math.max(0, Math.min(360, ((e.clientX - rect.left) / SV_SIZE) * 360));
        updateAll();
    }

    function onPointerMove(e) {
        if (svDrag) onSV(e);
        if (hueDrag) onHue(e);
    }

    function onPointerUp() {
        svDrag = false;
        hueDrag = false;
    }

    svCanvas.addEventListener('pointerdown', (e) => {
        svDrag = true; onSV(e);
        e.preventDefault(); e.stopPropagation();
    });
    hueCanvas.addEventListener('pointerdown', (e) => {
        hueDrag = true; onHue(e);
        e.preventDefault(); e.stopPropagation();
    });

    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);

    // Hex input: live update on valid input
    hexInput.addEventListener('input', () => {
        const val = hexInput.value.trim();
        if (/^#?[0-9a-fA-F]{6}$/.test(val)) {
            const hex = val.startsWith('#') ? val : '#' + val;
            hsv = hexToHSV(hex);
            preview.style.backgroundColor = hex;
            drawSV();
            drawHue();
        }
    });

    // Prevent keystrokes from reaching ComfyUI canvas
    hexInput.addEventListener('keydown', (e) => e.stopPropagation());

    // Prevent panel clicks from reaching backdrop
    panel.addEventListener('pointerdown', (e) => e.stopPropagation());

    // ── Close / Apply ──

    function close() {
        document.removeEventListener('pointermove', onPointerMove);
        document.removeEventListener('pointerup', onPointerUp);
        backdrop.remove();
        panel.remove();
    }

    backdrop.addEventListener('pointerdown', (e) => {
        e.preventDefault(); e.stopPropagation();
        onApply(hsvToHex(hsv.h, hsv.s, hsv.v));
        close();
    });

    okBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        onApply(hsvToHex(hsv.h, hsv.s, hsv.v));
        close();
    });

    // ── Mount & render ──

    document.body.append(backdrop, panel);
    updateAll();
}

// ── ComfyUI Widget Definition ─────────────────────────────────────────

const AILabColorWidget = {
    COLORCODE: (key, val) => {
        const widget = {};
        widget.y = 0;
        widget.name = key;
        widget.type = 'COLORCODE';

        const defaultColor = '#222222';
        widget.options = { default: defaultColor };

        let initialValue = defaultColor;
        if (Array.isArray(val) && val.length > 1 && val[1] && val[1].default) {
            initialValue = val[1].default;
        }

        if (typeof initialValue === 'string' && /^#?[0-9a-fA-F]{6}$/.test(initialValue)) {
            widget.value = initialValue;
        } else {
            widget.value = defaultColor;
        }


        widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
            const hide = this.type !== 'COLORCODE' && app.canvas.ds.scale > 0.5;
            if (hide) {
                return;
            }

            const actualWidth = node.size[0];
            const drawHeight = 22;
            const margin = 15;
            const radius = 10;

            ctx.fillStyle = this.value;
            ctx.beginPath();
            const x = margin;
            const y = widgetY + (height - drawHeight) / 2;
            const w = actualWidth - margin * 2;
            const h = drawHeight;
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + w - radius, y);
            ctx.arcTo(x + w, y, x + w, y + radius, radius);
            ctx.lineTo(x + w, y + h - radius);
            ctx.arcTo(x + w, y + h, x + w - radius, y + h, radius);
            ctx.lineTo(x + radius, y + h);
            ctx.arcTo(x, y + h, x, y + h - radius, radius);
            ctx.lineTo(x, y + radius);
            ctx.arcTo(x, y, x + radius, y, radius);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = getContrastTextColor(this.value);
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';

            const text = `${this.name} (${this.value})`;
            ctx.fillText(text, actualWidth * 0.5, y + drawHeight * 0.65);
        };

        widget.mouse = function (e, pos, node) {
            if (e.type === 'pointerdown') {
                const margin = 15;

                if (pos[0] >= margin && pos[0] <= node.size[0] - margin) {
                    const widgetRef = this;
                    openColorPicker(this.value, e.clientX, e.clientY, (newColor) => {
                        widgetRef.value = newColor;
                        widgetRef.callback?.(newColor);
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                    });
                    return true;
                }
            }
            return false;
        };

        widget.computeSize = function () {
            return [50, 22];
        };

        return widget;
    }
};

app.registerExtension({
    name: "AILab.colorWidget",

    getCustomWidgets() {
        return {
            COLORCODE: (node, inputName, inputData) => {
                return {
                    widget: node.addCustomWidget(
                        AILabColorWidget.COLORCODE(inputName, inputData)
                    ),
                    minWidth: 150,
                    minHeight: 22,
                };
            }
        };
    }
});