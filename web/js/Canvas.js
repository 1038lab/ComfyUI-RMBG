/**
 * Canvas (RMBG) — Frontend for ComfyUI
 *
 * This integration script is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * If you incorporate or modify this code, please credit AILab as the original source:
 * https://github.com/1038lab/ComfyUI-RMBG
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const STYLE_ID = "ailab-canvas-styles";
const MIN_DIM = 320;
const MAX_DIM = 4096;
const SIDE_SPACING = 10;
const BOTTOM_SPACING = 10;
const DOM_CHROME_HEIGHT = 146; // 90 (Deck) + 30 (Toolbar) + 12 (Gaps) + 12 (Paddings) + 2 (Borders)

function snap8(v) {
    return Math.round(v / 8) * 8;
}

function clampDimensions(targetW, ratio) {
    let w = snap8(targetW);
    let h = snap8(w / ratio);

    // Lower bound: neither width nor height may drop below 320
    if (w < MIN_DIM || h < MIN_DIM) {
        if (ratio >= 1.0) {
            h = MIN_DIM;
            w = snap8(MIN_DIM * ratio);
        } else {
            w = MIN_DIM;
            h = snap8(MIN_DIM / ratio);
        }
    }

    // Upper bound: neither width nor height may exceed 4096
    if (w > MAX_DIM || h > MAX_DIM) {
        if (ratio >= 1.0) {
            w = MAX_DIM;
            h = snap8(MAX_DIM / ratio);
        } else {
            h = MAX_DIM;
            w = snap8(MAX_DIM * ratio);
        }
    }

    return { w, h };
}

function clampFromHeight(targetH, ratio) {
    const h = snap8(targetH);
    const w = snap8(h * ratio);
    return clampDimensions(w, ratio);
}

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .alc-root {
            display: flex;
            flex-direction: column;
            width: 100%;
            box-sizing: border-box;
            background: #182230;
            border: 1px solid #2d4157;
            border-radius: 8px;
            padding: 6px;
            color: #d7e3ef;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            user-select: none;
            gap: 6px;
            overflow: hidden;
        }
        .alc-layer-deck {
            flex: 0 0 90px;
            height: 90px;
            min-height: 90px;
            max-height: 90px;
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
            align-items: center;
            gap: 6px;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 4px 6px;
            border-radius: 6px;
            background: rgba(13, 21, 31, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            scrollbar-width: thin;
            scrollbar-color: #2a3d52 transparent;
            box-sizing: border-box;
        }
        .alc-layer-deck::-webkit-scrollbar {
            height: 4px;
        }
        .alc-layer-deck::-webkit-scrollbar-thumb {
            background: #2a3d52;
            border-radius: 2px;
        }
        .alc-card {
            flex: 0 0 80px;
            width: 80px;
            height: 80px;
            min-width: 80px;
            max-width: 80px;
            min-height: 80px;
            max-height: 80px;
            border: 1.5px solid #2a3f54;
            border-radius: 6px;
            background: repeating-conic-gradient(#141d29 0% 25%, #1d2938 0% 50%) 50% / 8px 8px;
            position: relative;
            cursor: pointer;
            overflow: hidden;
            transition: border-color 0.15s, box-shadow 0.15s;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }
        .alc-card:hover {
            border-color: #08b4ed;
        }
        .alc-card.alc-active {
            border: 2px solid #08b4ed;
            box-shadow: 0 0 8px rgba(8, 180, 237, 0.5);
        }
        .alc-card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
        }
        .alc-card-badge {
            position: absolute;
            left: 2px;
            top: 2px;
            background: rgba(0, 0, 0, 0.75);
            color: #ffffff;
            font-size: 8px;
            font-weight: 600;
            padding: 1px 3px;
            border-radius: 3px;
            pointer-events: none;
            z-index: 5;
        }
        .alc-card-empty {
            font-size: 8.5px;
            color: #6a8299;
            text-align: center;
            padding: 2px;
            pointer-events: none;
        }
        .alc-card-remove {
            position: absolute;
            right: 2px;
            top: 2px;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: rgba(15, 23, 33, 0.85);
            color: #ff6b6b;
            border: 1px solid rgba(255, 107, 107, 0.6);
            font-size: 10px;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.15s, background 0.15s, transform 0.1s;
            z-index: 10;
            padding: 0;
        }
        .alc-card:hover .alc-card-remove {
            opacity: 1;
        }
        .alc-card-remove:hover {
            background: #e63946;
            color: #ffffff;
            transform: scale(1.15);
        }
        .alc-toolbar {
            flex: 0 0 30px;
            height: 30px;
            display: flex;
            align-items: center;
            gap: 6px;
            background: #111923;
            padding: 2px 6px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 11px;
            box-sizing: border-box;
        }
        .alc-dim-driver {
            display: flex;
            align-items: center;
            gap: 3px;
            font-size: 11px;
            color: #7b96b2;
            font-weight: 500;
        }
        .alc-dim-input {
            width: 44px;
            height: 20px;
            background: #182230;
            border: 1px solid #2d4157;
            color: #d7e3ef;
            font-size: 11px;
            text-align: center;
            border-radius: 3px;
            padding: 0 2px;
            outline: none;
            box-sizing: border-box;
        }
        .alc-dim-input:focus {
            border-color: #08b4ed;
        }
        .alc-btn {
            background: #223244;
            border: 1px solid #354e66;
            color: #e2edf8;
            border-radius: 4px;
            width: 24px;
            height: 24px;
            min-width: 24px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.15s, border-color 0.15s;
            padding: 0;
            box-sizing: border-box;
        }
        .alc-btn:hover {
            background: #2c425a;
            border-color: #08b4ed;
            color: #ffffff;
        }
        .alc-btn.alc-active {
            background: #08b4ed;
            border-color: #38c8f5;
            color: #ffffff;
        }
        .alc-slider-wrap {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-left: auto;
            font-size: 11px;
            color: #9cb5cb;
            white-space: nowrap;
        }
        .alc-slider-wrap input[type=range] {
            width: 60px;
            accent-color: #08b4ed;
            cursor: pointer;
        }
        .alc-stage-wrap {
            position: relative;
            width: 100%;
            background: repeating-conic-gradient(#141c26 0% 25%, #192432 0% 50%) 50% / 16px 16px;
            border: 1px solid #2d4255;
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }
        .alc-canvas {
            display: block;
            width: 100%;
            height: 100%;
            cursor: default;
            background: transparent;
        }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "AILab.Canvas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AILab_Canvas") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            this.hideOutputImages = true;
            this.imgs = null;

            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            this.hideOutputImages = true;
            this.imgs = null;

            injectStyles();

            const node = this;

            node.canvasLayers = [];
            node.activeLayerId = null;
            node.bgImage = null;
            node.isLocked = false;
            node.currentW = 640;
            node.currentH = 360;

            const hideWidget = (name) => {
                const w = node.widgets?.find(x => x.name === name);
                if (w) {
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                    w.draw = () => { };
                }
                return w;
            };

            const wCanvasWidth = hideWidget("canvas_width");
            const wCanvasHeight = hideWidget("canvas_height");
            let wCanvasState = hideWidget("canvas_state");
            if (!wCanvasState) {
                wCanvasState = node.widgets?.find(x => x.name === "canvas_state");
            }
            if (!wCanvasState && node.addWidget) {
                wCanvasState = node.addWidget("text", "canvas_state", "{}", () => { }, { multiline: true });
                wCanvasState.type = "hidden";
                wCanvasState.computeSize = () => [0, -4];
                wCanvasState.draw = () => { };
            }
            const wBg = node.widgets?.find(x => x.name === "background");
            const wBgColor = node.widgets?.find(x => x.name === "background_color");
            const wFitBg = hideWidget("fit_background_size");

            if (wBg) {
                const oldBgCb = wBg.callback;
                wBg.callback = function () {
                    if (oldBgCb) oldBgCb.apply(this, arguments);
                    node.renderStage();
                };
            }
            if (wBgColor) {
                const oldColorCb = wBgColor.callback;
                wBgColor.callback = function () {
                    if (oldColorCb) oldColorCb.apply(this, arguments);
                    node.renderStage();
                };
            }
            if (wFitBg) {
                const oldFitCb = wFitBg.callback;
                wFitBg.callback = function () {
                    if (oldFitCb) oldFitCb.apply(this, arguments);
                    if (wFitBg.value && node.bgImage) {
                        node.fitToBackground();
                    }
                };
            }

            const root = document.createElement("div");
            root.className = "alc-root";

            // Tier 1: Layer Preview Deck (Top of DOM Panel)
            const previewDeck = document.createElement("div");
            previewDeck.className = "alc-layer-deck";

            // Tier 2: Control Toolbar (Middle of DOM Panel)
            const toolbar = document.createElement("div");
            toolbar.className = "alc-toolbar";

            const dimDriver = document.createElement("div");
            dimDriver.className = "alc-dim-driver";

            const wLabel = document.createElement("span");
            wLabel.textContent = "W:";
            const wInput = document.createElement("input");
            wInput.type = "number";
            wInput.step = "8";
            wInput.className = "alc-dim-input alc-w-input";
            wInput.value = 640;

            const xLabel = document.createElement("span");
            xLabel.textContent = "×";

            const hLabel = document.createElement("span");
            hLabel.textContent = "H:";
            const hInput = document.createElement("input");
            hInput.type = "number";
            hInput.step = "8";
            hInput.className = "alc-dim-input alc-h-input";
            hInput.value = 360;

            dimDriver.appendChild(wLabel);
            dimDriver.appendChild(wInput);
            dimDriver.appendChild(xLabel);
            dimDriver.appendChild(hLabel);
            dimDriver.appendChild(hInput);

            const btnSep = document.createElement("div");
            btnSep.style.cssText = "width: 1px; height: 16px; background: rgba(255, 255, 255, 0.1); margin: 0 2px;";

            const createBtn = (icon, title, onClick) => {
                const b = document.createElement("button");
                b.className = "alc-btn";
                b.textContent = icon;
                b.title = title;
                b.addEventListener("click", (e) => {
                    e.stopPropagation();
                    onClick(b);
                });
                return b;
            };

            const btnFlipH = createBtn("⇄", "Flip Horizontal", () => node.flipActiveLayer(true, false));
            const btnFlipV = createBtn("⇅", "Flip Vertical", () => node.flipActiveLayer(false, true));
            const btnReset = createBtn("↺", "Reset Transform", () => node.resetActiveLayer());
            const btnFit = createBtn("⤢", "Fit Background Size", () => {
                if (wFitBg) wFitBg.value = true;
                node.fitToBackground();
            });
            const btnLock = createBtn("🔓", "Lock Layout", (btn) => {
                node.isLocked = !node.isLocked;
                btn.textContent = node.isLocked ? "🔒" : "🔓";
                btn.classList.toggle("alc-active", node.isLocked);
                node.saveCanvasState();
                node.renderStage();
            });

            const sliderWrap = document.createElement("div");
            sliderWrap.className = "alc-slider-wrap";
            const opLabel = document.createElement("span");
            opLabel.textContent = "Opacity:";
            const opSlider = document.createElement("input");
            opSlider.type = "range";
            opSlider.min = "0";
            opSlider.max = "100";
            opSlider.value = "100";
            const opVal = document.createElement("span");
            opVal.style.cssText = "min-width: 32px; text-align: right; color: #9cb5cb;";
            opVal.textContent = "100%";

            opSlider.addEventListener("input", (e) => {
                if (node.isLocked) return;
                const v = parseInt(e.target.value, 10);
                opVal.textContent = v + "%";
                node.setActiveLayerOpacity(v / 100.0);
            });

            sliderWrap.appendChild(opLabel);
            sliderWrap.appendChild(opSlider);
            sliderWrap.appendChild(opVal);

            toolbar.appendChild(dimDriver);
            toolbar.appendChild(btnSep);
            toolbar.appendChild(btnFlipH);
            toolbar.appendChild(btnFlipV);
            toolbar.appendChild(btnReset);
            toolbar.appendChild(btnFit);
            toolbar.appendChild(btnLock);
            toolbar.appendChild(sliderWrap);

            // Tier 3: Stage Area (Bottom of DOM Panel)
            const stageWrap = document.createElement("div");
            stageWrap.className = "alc-stage-wrap";
            const stageCanvas = document.createElement("canvas");
            stageCanvas.className = "alc-canvas";
            stageCanvas.width = 640;
            stageCanvas.height = 360;
            stageWrap.appendChild(stageCanvas);

            // Strict Vertical Stacking: Deck (Top) -> Toolbar (Middle) -> Stage (Bottom)
            root.appendChild(previewDeck);
            root.appendChild(toolbar);
            root.appendChild(stageWrap);

            const domWidget = node.addDOMWidget("AILab_Canvas_UI", "custom_ui", root, {
                serialize: false,
                hideOnZoom: false,
            });

            const origWidgetDraw = domWidget.draw;
            domWidget.draw = function (ctx, n, widget_width, y, widget_height) {
                if (typeof y === "number" && y > 30) {
                    domWidget.last_y = y;
                    const stageH = stageCanvas.height || 360;
                    const domH = DOM_CHROME_HEIGHT + stageH;
                    const expectedH = Math.ceil(y + domH + BOTTOM_SPACING);
                    if (node.size && Math.abs(node.size[1] - expectedH) > 2) {
                        node.size[1] = expectedH;
                    }
                }
                return origWidgetDraw ? origWidgetDraw.apply(this, arguments) : undefined;
            };

            node.getNativeHeaderHeight = function () {
                if (domWidget && typeof domWidget.last_y === "number" && domWidget.last_y > 30) {
                    return domWidget.last_y;
                }
                const numInputs = node.inputs ? node.inputs.length : 1;
                const numOutputs = node.outputs ? node.outputs.length : 1;
                const slotsH = Math.max(numInputs, numOutputs) * 20;
                const startY = Math.max(30, slotsH + 10);
                const regularWidgets = (node.widgets || []).filter(w => w !== domWidget && w.type !== "hidden").length;
                return startY + regularWidgets * 24 + 4;
            };

            node.getCurrentAspectRatio = function () {
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    return node.bgImage.naturalWidth / node.bgImage.naturalHeight;
                }
                const w = node.currentW || 640;
                const h = node.currentH || 360;
                return Math.max(0.01, w / Math.max(1, h));
            };

            // Single Source of Truth for all node dimensions & aspect ratio scaling
            node.setCanvasDimensions = function (targetW, targetH) {
                const w = Math.max(MIN_DIM, Math.min(MAX_DIM, snap8(targetW)));
                const h = Math.max(MIN_DIM, Math.min(MAX_DIM, snap8(targetH)));

                const oldW = node.currentW;
                const oldH = node.currentH;

                node.currentW = w;
                node.currentH = h;
                wInput.value = w;
                hInput.value = h;
                const cw = wCanvasWidth || node.widgets?.find(x => x.name === "canvas_width");
                if (cw) cw.value = w;
                const ch = wCanvasHeight || node.widgets?.find(x => x.name === "canvas_height");
                if (ch) ch.value = h;

                // Reposition existing layers relative to canvas, but NEVER distort layer aspect ratio
                if (oldW && oldH && (oldW !== w || oldH !== h)) {
                    const factorX = w / oldW;
                    const factorY = h / oldH;
                    const isUniformCanvasScale = Math.abs(factorX - factorY) < 0.05;
                    const uniformScaleFactor = isUniformCanvasScale ? factorX : 1.0;
                    for (const l of node.canvasLayers) {
                        if (typeof l.x === "number") l.x = Math.round(l.x * factorX);
                        if (typeof l.y === "number") l.y = Math.round(l.y * factorY);
                        // Layer aspect ratio is locked: scaleX and scaleY are strictly equal
                        const curScale = (typeof l.scaleX === "number") ? l.scaleX : (l.scaleY ?? 1.0);
                        const newScale = Number((curScale * uniformScaleFactor).toFixed(4));
                        l.scaleX = newScale;
                        l.scaleY = newScale;
                    }
                }

                stageCanvas.width = w;
                stageCanvas.height = h;
                stageWrap.style.height = h + "px";

                const nodeW = w + SIDE_SPACING * 2;
                const nodeH = node.getNativeHeaderHeight() + DOM_CHROME_HEIGHT + h + BOTTOM_SPACING;
                node.size = [nodeW, nodeH];

                node.saveCanvasState();
                node.renderStage();
                node.setDirtyCanvas?.(true, true);
            };

            node.syncGeometry = function (targetCanvasWidth) {
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    const ratio = node.bgImage.naturalWidth / node.bgImage.naturalHeight;
                    const { w, h } = clampDimensions(targetCanvasWidth, ratio);
                    node.setCanvasDimensions(w, h);
                } else {
                    const w = Math.max(MIN_DIM, Math.min(MAX_DIM, snap8(targetCanvasWidth)));
                    const h = node.currentH || 360;
                    node.setCanvasDimensions(w, h);
                }
            };

            node.syncGeometryFromHeight = function (targetCanvasHeight) {
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    const ratio = node.bgImage.naturalWidth / node.bgImage.naturalHeight;
                    const { w, h } = clampFromHeight(targetCanvasHeight, ratio);
                    node.setCanvasDimensions(w, h);
                } else {
                    const w = node.currentW || 640;
                    const h = Math.max(MIN_DIM, Math.min(MAX_DIM, snap8(targetCanvasHeight)));
                    node.setCanvasDimensions(w, h);
                }
            };

            // LiteGraph min size definition
            node.computeSize = function (out) {
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    const ratio = node.bgImage.naturalWidth / node.bgImage.naturalHeight;
                    const minDims = clampDimensions(0, ratio);
                    const minNodeW = minDims.w + SIDE_SPACING * 2;
                    const minNodeH = node.getNativeHeaderHeight() + DOM_CHROME_HEIGHT + minDims.h + BOTTOM_SPACING;
                    if (out) {
                        out[0] = minNodeW;
                        out[1] = minNodeH;
                        return out;
                    }
                    return [minNodeW, minNodeH];
                }
                const minNodeW = MIN_DIM + SIDE_SPACING * 2;
                const minNodeH = node.getNativeHeaderHeight() + DOM_CHROME_HEIGHT + MIN_DIM + BOTTOM_SPACING;
                if (out) {
                    out[0] = minNodeW;
                    out[1] = minNodeH;
                    return out;
                }
                return [minNodeW, minNodeH];
            };

            domWidget.computeSize = function (width) {
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    const ratio = node.bgImage.naturalWidth / node.bgImage.naturalHeight;
                    const targetW = width ? Math.max(MIN_DIM, width - SIDE_SPACING * 2) : (node.currentW || 640);
                    const { w, h } = clampDimensions(targetW, ratio);
                    return [w + SIDE_SPACING * 2, DOM_CHROME_HEIGHT + h];
                }
                const w = width ? Math.max(MIN_DIM, width - SIDE_SPACING * 2) : (node.currentW || 640);
                const h = node.currentH || 360;
                return [w + SIDE_SPACING * 2, DOM_CHROME_HEIGHT + h];
            };

            node.onResize = function (newSize) {
                if (wFitBg) wFitBg.value = false;
                if (node.bgImage && node.bgImage.naturalWidth && node.bgImage.naturalHeight) {
                    const targetW = newSize[0] - SIDE_SPACING * 2;
                    node.syncGeometry(targetW);
                } else {
                    const targetW = newSize[0] - SIDE_SPACING * 2;
                    const headerH = node.getNativeHeaderHeight();
                    const targetH = newSize[1] - headerH - DOM_CHROME_HEIGHT - BOTTOM_SPACING;
                    node.setCanvasDimensions(targetW, targetH);
                }
                newSize[0] = node.size[0];
                newSize[1] = node.size[1];
            };

            wInput.addEventListener("change", () => {
                if (wFitBg) wFitBg.value = false;
                const val = parseInt(wInput.value, 10) || 640;
                node.syncGeometry(val);
            });

            hInput.addEventListener("change", () => {
                if (wFitBg) wFitBg.value = false;
                const val = parseInt(hInput.value, 10) || 360;
                node.syncGeometryFromHeight(val);
            });

            // Enforce that any newly loaded layer does not exceed 85% of stage width/height
            node.applyInitialLayerConstraints = function (layer, img) {
                const stageW = node.currentW || 640;
                const stageH = node.currentH || 360;
                const maxW = stageW * 0.85;
                const maxH = stageH * 0.85;
                const imgW = img.naturalWidth || 1;
                const imgH = img.naturalHeight || 1;

                if (imgW > maxW || imgH > maxH) {
                    const fitScale = Math.min(maxW / imgW, maxH / imgH);
                    layer.scaleX = Number(fitScale.toFixed(4));
                    layer.scaleY = Number(fitScale.toFixed(4));
                } else {
                    layer.scaleX = 1.0;
                    layer.scaleY = 1.0;
                }
                layer.x = Math.round(stageW / 2);
                layer.y = Math.round(stageH / 2);
                layer.angle = 0;
                layer.flipX = false;
                layer.flipY = false;
                layer.userModified = false;
            };

            node.renderPreviewDeck = function () {
                previewDeck.innerHTML = "";
                const totalCards = [
                    { id: "Background", label: "BG", isBg: true, image: node.bgImage }
                ];

                if (node.inputs) {
                    for (const inp of node.inputs) {
                        if (inp.name.startsWith("layer_")) {
                            const lNum = inp.name.replace("layer_", "");
                            const l = node.canvasLayers.find(x => x.id === inp.name);
                            const hasLink = inp.link != null && (!app.graph || !app.graph.links || app.graph.links[inp.link] != null);
                            const hasImage = Boolean(l && l.image && l.image.src);

                            // Layer 1 is base layer: show if connected, has image, or empty initial state.
                            // Dynamic layers (layer_2+) MUST ONLY be shown if actively connected or having an image.
                            const shouldShow = (inp.name === "layer_1")
                                ? (hasLink || hasImage || (!node.canvasLayers.length && !node.inputs.some(i => i.name.startsWith("layer_") && i.name !== "layer_1" && (i.link != null && (!app.graph?.links || app.graph.links[i.link])))))
                                : (hasLink || hasImage);

                            if (shouldShow) {
                                totalCards.push({
                                    id: inp.name,
                                    label: `L${lNum}`,
                                    isBg: false,
                                    layer: l,
                                    image: (l && l.image && l.image.src) ? l.image : null
                                });
                            }
                        }
                    }
                }

                totalCards.forEach((c) => {
                    const card = document.createElement("div");
                    card.className = "alc-card";
                    if (node.activeLayerId === c.id) {
                        card.classList.add("alc-active");
                    }

                    const badge = document.createElement("div");
                    badge.className = "alc-card-badge";
                    badge.textContent = c.label;
                    card.appendChild(badge);

                    if (c.image && c.image.src) {
                        const img = document.createElement("img");
                        img.src = c.image.src;
                        card.appendChild(img);
                    } else {
                        const empty = document.createElement("div");
                        empty.className = "alc-card-empty";
                        empty.textContent = c.isBg ? "No BG" : "No Img";
                        card.appendChild(empty);
                    }

                    if (c.isBg && node.bgImage) {
                        const closeBtn = document.createElement("div");
                        closeBtn.className = "alc-card-remove";
                        closeBtn.textContent = "✕";
                        closeBtn.title = "Delete Background Image";
                        closeBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
                        closeBtn.addEventListener("mousedown", (e) => e.stopPropagation());
                        closeBtn.addEventListener("click", (e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            node.disconnectBgInput();
                        });
                        card.appendChild(closeBtn);
                    } else if (!c.isBg && (c.image || c.layer)) {
                        const closeBtn = document.createElement("div");
                        closeBtn.className = "alc-card-remove";
                        closeBtn.textContent = "✕";
                        closeBtn.title = `Delete ${c.label}`;
                        closeBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
                        closeBtn.addEventListener("mousedown", (e) => e.stopPropagation());
                        closeBtn.addEventListener("click", (e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            node.disconnectLayerInput(c.id);
                        });
                        card.appendChild(closeBtn);
                    }

                    card.addEventListener("click", () => {
                        node.activeLayerId = c.id;
                        if (c.layer) {
                            const pct = Math.round((c.layer.opacity ?? 1.0) * 100);
                            opSlider.value = pct;
                            opVal.textContent = pct + "%";
                        }
                        node.renderPreviewDeck();
                        node.renderStage();
                    });

                    previewDeck.appendChild(card);
                });
            };

            node.disconnectBgInput = function () {
                const slotIndex = node.findInputSlot("Background");
                if (slotIndex >= 0 && node.inputs && node.inputs[slotIndex] && node.inputs[slotIndex].link != null) {
                    node.disconnectInput(slotIndex);
                }
                node.bgImage = null;
                node.setCanvasDimensions(node.currentW || 640, node.currentH || 360);
                node.saveCanvasState();
                node.renderPreviewDeck();
                node.renderStage();
            };

            node.disconnectLayerInput = function (layerId) {
                const slotIndex = node.findInputSlot(layerId);
                if (slotIndex >= 0 && node.inputs && node.inputs[slotIndex] && node.inputs[slotIndex].link != null) {
                    node.disconnectInput(slotIndex);
                }
                node.canvasLayers = (node.canvasLayers || []).filter(l => l.id !== layerId);
                if (node.activeLayerId === layerId) {
                    node.activeLayerId = null;
                }
                node.manageDynamicSlots();
                node.saveCanvasState();
                node.renderPreviewDeck();
                node.renderStage();
            };

            node.renderStage = function () {
                const ctx = stageCanvas.getContext("2d");
                const W = stageCanvas.width;
                const H = stageCanvas.height;
                ctx.clearRect(0, 0, W, H);

                const bgType = wBg ? wBg.value : "Alpha";
                const bgColor = wBgColor ? wBgColor.value : "#222222";

                if (node.bgImage && node.bgImage.complete && node.bgImage.naturalWidth > 0) {
                    ctx.drawImage(node.bgImage, 0, 0, W, H);
                } else if (bgType === "Color") {
                    ctx.fillStyle = bgColor;
                    ctx.fillRect(0, 0, W, H);
                } else {
                    ctx.save();
                    const patSize = 16;
                    for (let y = 0; y < H; y += patSize) {
                        for (let x = 0; x < W; x += patSize) {
                            ctx.fillStyle = ((x / patSize + y / patSize) % 2 === 0) ? "#141c26" : "#192432";
                            ctx.fillRect(x, y, patSize, patSize);
                        }
                    }
                    ctx.restore();
                }

                const sortedLayers = [...node.canvasLayers].sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));
                const scaleRatioX = W / (node.currentW || W);
                const scaleRatioY = H / (node.currentH || H);

                sortedLayers.forEach(l => {
                    if (!l.image || !l.image.complete) return;
                    ctx.save();
                    const lx = (l.x ?? (node.currentW / 2)) * scaleRatioX;
                    const ly = (l.y ?? (node.currentH / 2)) * scaleRatioY;
                    ctx.translate(lx, ly);
                    ctx.rotate(((l.angle || 0) * Math.PI) / 180);
                    ctx.scale(l.flipX ? -1 : 1, l.flipY ? -1 : 1);
                    ctx.globalAlpha = Math.max(0, Math.min(1, l.opacity ?? 1.0));

                    // Strictly lock layer scale to natural aspect ratio
                    const scale = (typeof l.scaleX === "number") ? l.scaleX : (l.scaleY ?? 1.0);
                    l.scaleX = scale;
                    l.scaleY = scale;
                    const lw = (l.image.naturalWidth * scale) * scaleRatioX;
                    const lh = (l.image.naturalHeight * scale) * scaleRatioY;
                    ctx.drawImage(l.image, -lw / 2, -lh / 2, lw, lh);
                    ctx.restore();

                    if (!node.isLocked && node.activeLayerId === l.id) {
                        node.drawGizmo(ctx, lx, ly, lw, lh, l.angle || 0);
                    }
                });
            };

            node.drawGizmo = function (ctx, cx, cy, w, h, angle) {
                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate((angle * Math.PI) / 180);

                ctx.strokeStyle = "#08b4ed";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(-w / 2, -h / 2, w, h);
                ctx.setLineDash([]);

                const handles = [
                    [-w / 2, -h / 2], [w / 2, -h / 2],
                    [-w / 2, h / 2], [w / 2, h / 2],
                ];

                ctx.fillStyle = "#ffffff";
                ctx.strokeStyle = "#08b4ed";
                ctx.lineWidth = 1.5;
                const hs = 7;
                handles.forEach(([hx, hy]) => {
                    ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
                    ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
                });

                ctx.beginPath();
                ctx.moveTo(0, -h / 2);
                ctx.lineTo(0, -h / 2 - 18);
                ctx.strokeStyle = "#08b4ed";
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(0, -h / 2 - 18, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = "#08b4ed";
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.2;
                ctx.stroke();

                ctx.restore();
            };

            node.flipActiveLayer = function (h, v) {
                if (node.isLocked || !node.activeLayerId) return;
                const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                if (!l) return;
                if (h) l.flipX = !l.flipX;
                if (v) l.flipY = !l.flipY;
                l.userModified = true;
                node.saveCanvasState();
                node.renderStage();
            };

            node.resetActiveLayer = function () {
                if (node.isLocked || !node.activeLayerId) return;
                const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                if (!l) return;
                if (l.image && l.image.complete) {
                    node.applyInitialLayerConstraints(l, l.image);
                } else {
                    l.x = (node.currentW || 640) / 2;
                    l.y = (node.currentH || 360) / 2;
                    l.scaleX = 1.0;
                    l.scaleY = 1.0;
                    l.angle = 0;
                    l.flipX = false;
                    l.flipY = false;
                }
                l.opacity = 1.0;
                opSlider.value = 100;
                opVal.textContent = "100%";
                node.saveCanvasState();
                node.renderStage();
            };

            node.fitToBackground = function () {
                if (!node.bgImage || !node.bgImage.naturalWidth || !node.bgImage.naturalHeight) return;
                if (wFitBg) wFitBg.value = true;
                node.setCanvasDimensions(node.bgImage.naturalWidth, node.bgImage.naturalHeight);
            };

            node.onBackgroundLoaded = function (img) {
                node.bgImage = img;
                const bgW = img.naturalWidth;
                const bgH = img.naturalHeight;
                if (!bgW || !bgH) return;

                const ratio = bgW / bgH;
                const curW = node.currentW || 640;
                const curH = node.currentH || 360;
                const curRatio = curW / Math.max(1, curH);
                const isRatioMatched = Math.abs(curRatio - ratio) / ratio < 0.02;

                if (wFitBg && wFitBg.value) {
                    node.setCanvasDimensions(bgW, bgH);
                } else if (!isRatioMatched) {
                    const { w, h } = clampDimensions(curW, ratio);
                    node.setCanvasDimensions(w, h);
                } else {
                    node.setCanvasDimensions(curW, curH);
                }

                for (const l of node.canvasLayers) {
                    if (l.image && l.userModified !== true) {
                        node.applyInitialLayerConstraints(l, l.image);
                    }
                }

                node.saveCanvasState();
                node.renderPreviewDeck();
                node.renderStage();
            };

            node.onLayerImageLoaded = function (slotName, img) {
                let layer = node.canvasLayers.find(l => l.id === slotName);
                const isNew = !layer;
                if (isNew) {
                    layer = {
                        id: slotName,
                        opacity: 1.0,
                        zIndex: parseInt(slotName.replace("layer_", ""), 10) || 1,
                    };
                    node.canvasLayers.push(layer);
                }
                layer.image = img;
                if (isNew || layer.userModified !== true) {
                    node.applyInitialLayerConstraints(layer, img);
                }
                node.saveCanvasState();
                node.renderPreviewDeck();
                node.renderStage();
            };

            node.setActiveLayerOpacity = function (opacity) {
                if (node.isLocked || !node.activeLayerId) return;
                const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                if (!l) return;
                l.opacity = opacity;
                node.saveCanvasState();
                node.renderStage();
            };

            node.saveCanvasState = function () {
                const state = {
                    canvas_width: node.currentW || 640,
                    canvas_height: node.currentH || 360,
                    locked: !!node.isLocked,
                    layers: node.canvasLayers.map(l => ({
                        id: l.id,
                        x: l.x ?? ((node.currentW || 640) / 2),
                        y: l.y ?? ((node.currentH || 360) / 2),
                        scaleX: l.scaleX ?? 1.0,
                        scaleY: l.scaleY ?? 1.0,
                        angle: l.angle ?? 0.0,
                        flipX: !!l.flipX,
                        flipY: !!l.flipY,
                        opacity: l.opacity ?? 1.0,
                        zIndex: l.zIndex ?? 1
                    }))
                };
                let w = wCanvasState || node.widgets?.find(x => x.name === "canvas_state");
                if (!w && node.addWidget) {
                    w = node.addWidget("text", "canvas_state", "{}", () => { }, { multiline: true });
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                    w.draw = () => { };
                }
                if (w) {
                    w.value = JSON.stringify(state);
                }
            };

            function getStageCoords(e) {
                const rect = stageCanvas.getBoundingClientRect();
                const sx = stageCanvas.width / Math.max(1, rect.width);
                const sy = stageCanvas.height / Math.max(1, rect.height);
                return {
                    x: (e.clientX - rect.left) * sx,
                    y: (e.clientY - rect.top) * sy
                };
            }

            let dragMode = null;
            let dragStartPos = null;
            let activeLayerInitial = null;

            stageCanvas.addEventListener("pointerdown", (e) => {
                if (node.isLocked) return;
                const pos = getStageCoords(e);
                const W = stageCanvas.width;
                const H = stageCanvas.height;
                const scaleRatioX = W / (node.currentW || W);
                const scaleRatioY = H / (node.currentH || H);

                let hitLayer = null;
                dragMode = null;

                // Test handles on currently active layer first
                if (node.activeLayerId) {
                    const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                    if (l && l.image && l.image.complete) {
                        const lx = (l.x ?? (node.currentW / 2)) * scaleRatioX;
                        const ly = (l.y ?? (node.currentH / 2)) * scaleRatioY;
                        const lw = (l.image.naturalWidth * (l.scaleX || 1.0)) * scaleRatioX;
                        const lh = (l.image.naturalHeight * (l.scaleY || 1.0)) * scaleRatioY;

                        const rad = -((l.angle || 0) * Math.PI) / 180;
                        const dx = pos.x - lx;
                        const dy = pos.y - ly;
                        const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
                        const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

                        // 1. Rotation handle
                        const rotDist = Math.hypot(rx, ry - (-lh / 2 - 18));
                        if (rotDist <= 12) {
                            hitLayer = l;
                            dragMode = "rotate";
                        } else {
                            // 2. Four corner scale handles
                            const scaleHandles = [
                                { id: "scale_tl", x: -lw / 2, y: -lh / 2 },
                                { id: "scale_tr", x: lw / 2, y: -lh / 2 },
                                { id: "scale_bl", x: -lw / 2, y: lh / 2 },
                                { id: "scale_br", x: lw / 2, y: lh / 2 },
                            ];
                            const tol = 9;
                            for (const h of scaleHandles) {
                                if (Math.abs(rx - h.x) <= tol && Math.abs(ry - h.y) <= tol) {
                                    hitLayer = l;
                                    dragMode = h.id;
                                    break;
                                }
                            }
                        }
                    }
                }

                // If no handle was clicked, test body click across all layers (topmost first)
                if (!hitLayer) {
                    const sorted = [...node.canvasLayers].sort((a, b) => (b.zIndex || 0) - (a.zIndex || 0));
                    for (const l of sorted) {
                        if (!l.image || !l.image.complete) continue;
                        const lx = (l.x ?? (node.currentW / 2)) * scaleRatioX;
                        const ly = (l.y ?? (node.currentH / 2)) * scaleRatioY;
                        const lw = (l.image.naturalWidth * (l.scaleX || 1.0)) * scaleRatioX;
                        const lh = (l.image.naturalHeight * (l.scaleY || 1.0)) * scaleRatioY;

                        const rad = -((l.angle || 0) * Math.PI) / 180;
                        const dx = pos.x - lx;
                        const dy = pos.y - ly;
                        const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
                        const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

                        if (Math.abs(rx) <= lw / 2 && Math.abs(ry) <= lh / 2) {
                            hitLayer = l;
                            dragMode = "move";
                            break;
                        }
                    }
                }

                if (hitLayer) {
                    node.activeLayerId = hitLayer.id;
                    const pct = Math.round((hitLayer.opacity ?? 1.0) * 100);
                    opSlider.value = pct;
                    opVal.textContent = pct + "%";
                    dragStartPos = pos;
                    activeLayerInitial = {
                        x: hitLayer.x ?? (node.currentW / 2),
                        y: hitLayer.y ?? (node.currentH / 2),
                        scaleX: hitLayer.scaleX ?? 1.0,
                        scaleY: hitLayer.scaleY ?? 1.0,
                        angle: hitLayer.angle ?? 0.0,
                        natW: hitLayer.image.naturalWidth,
                        natH: hitLayer.image.naturalHeight,
                    };
                    stageCanvas.setPointerCapture(e.pointerId);
                    node.renderPreviewDeck();
                    node.renderStage();
                } else {
                    node.activeLayerId = null;
                    dragMode = null;
                    node.renderPreviewDeck();
                    node.renderStage();
                }
            });

            stageCanvas.addEventListener("pointermove", (e) => {
                const pos = getStageCoords(e);
                const W = stageCanvas.width;
                const H = stageCanvas.height;
                const scaleRatioX = W / (node.currentW || W);
                const scaleRatioY = H / (node.currentH || H);

                // Hover cursor feedback when not dragging
                if (!dragMode) {
                    if (node.isLocked || !node.activeLayerId) {
                        stageCanvas.style.cursor = "default";
                        return;
                    }
                    const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                    if (!l || !l.image || !l.image.complete) {
                        stageCanvas.style.cursor = "default";
                        return;
                    }
                    const lx = (l.x ?? (node.currentW / 2)) * scaleRatioX;
                    const ly = (l.y ?? (node.currentH / 2)) * scaleRatioY;
                    const lw = (l.image.naturalWidth * (l.scaleX || 1.0)) * scaleRatioX;
                    const lh = (l.image.naturalHeight * (l.scaleY || 1.0)) * scaleRatioY;

                    const rad = -((l.angle || 0) * Math.PI) / 180;
                    const dx = pos.x - lx;
                    const dy = pos.y - ly;
                    const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
                    const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

                    const rotDist = Math.hypot(rx, ry - (-lh / 2 - 18));
                    if (rotDist <= 12) {
                        stageCanvas.style.cursor = "crosshair";
                        return;
                    }

                    const handles = [
                        { id: "scale_tl", x: -lw / 2, y: -lh / 2, cursor: "nwse-resize" },
                        { id: "scale_tr", x: lw / 2, y: -lh / 2, cursor: "nesw-resize" },
                        { id: "scale_bl", x: -lw / 2, y: lh / 2, cursor: "nesw-resize" },
                        { id: "scale_br", x: lw / 2, y: lh / 2, cursor: "nwse-resize" },
                    ];
                    const tol = 9;
                    const matched = handles.find(h => Math.abs(rx - h.x) <= tol && Math.abs(ry - h.y) <= tol);
                    if (matched) {
                        stageCanvas.style.cursor = matched.cursor;
                        return;
                    }
                    if (Math.abs(rx) <= lw / 2 && Math.abs(ry) <= lh / 2) {
                        stageCanvas.style.cursor = "move";
                        return;
                    }
                    stageCanvas.style.cursor = "default";
                    return;
                }

                if (node.isLocked || !node.activeLayerId) return;
                const l = node.canvasLayers.find(x => x.id === node.activeLayerId);
                if (!l || !activeLayerInitial) return;

                l.userModified = true;

                if (dragMode === "move") {
                    stageCanvas.style.cursor = "move";
                    const dx = (pos.x - dragStartPos.x) / scaleRatioX;
                    const dy = (pos.y - dragStartPos.y) / scaleRatioY;
                    l.x = activeLayerInitial.x + dx;
                    l.y = activeLayerInitial.y + dy;
                    node.renderStage();
                } else if (dragMode === "rotate") {
                    stageCanvas.style.cursor = "crosshair";
                    const lx = (l.x ?? (node.currentW / 2)) * scaleRatioX;
                    const ly = (l.y ?? (node.currentH / 2)) * scaleRatioY;
                    let angle = Math.atan2(pos.y - ly, pos.x - lx) * (180 / Math.PI) + 90;
                    if (e.shiftKey) {
                        angle = Math.round(angle / 15) * 15;
                    }
                    l.angle = angle;
                    node.renderStage();
                } else if (dragMode.startsWith("scale_")) {
                    const initCx = activeLayerInitial.x * scaleRatioX;
                    const initCy = activeLayerInitial.y * scaleRatioY;
                    const rad = ((activeLayerInitial.angle || 0) * Math.PI) / 180;
                    const cos = Math.cos(rad);
                    const sin = Math.sin(rad);
                    const invCos = Math.cos(-rad);
                    const invSin = Math.sin(-rad);

                    const dx = pos.x - initCx;
                    const dy = pos.y - initCy;
                    const localX = dx * invCos - dy * invSin;
                    const localY = dx * invSin + dy * invCos;

                    const W0 = activeLayerInitial.natW * activeLayerInitial.scaleX * scaleRatioX;
                    const H0 = activeLayerInitial.natH * activeLayerInitial.scaleY * scaleRatioY;
                    const hw = W0 / 2;
                    const hh = H0 / 2;

                    const anchors = {
                        "scale_br": { ax: -hw, ay: -hh, dirX: 1, dirY: 1, cursor: "nwse-resize" },
                        "scale_tl": { ax: hw, ay: hh, dirX: -1, dirY: -1, cursor: "nwse-resize" },
                        "scale_tr": { ax: -hw, ay: hh, dirX: 1, dirY: -1, cursor: "nesw-resize" },
                        "scale_bl": { ax: hw, ay: -hh, dirX: -1, dirY: 1, cursor: "nesw-resize" },
                    };

                    const h = anchors[dragMode];
                    if (h) {
                        stageCanvas.style.cursor = h.cursor;
                        const vx = (localX - h.ax) * h.dirX;
                        const vy = (localY - h.ay) * h.dirY;

                        // Proportional scaling maintaining aspect ratio
                        const diag = Math.hypot(W0, H0);
                        const proj = (vx * W0 + vy * H0) / Math.max(1, diag);
                        const scaleFactor = Math.max(0.05, proj / Math.max(1, diag));
                        const newW = Math.max(16, W0 * scaleFactor);
                        const newH = Math.max(16, H0 * scaleFactor);

                        const newLocalCx = h.ax + (newW / 2) * h.dirX;
                        const newLocalCy = h.ay + (newH / 2) * h.dirY;

                        const newStageCx = initCx + newLocalCx * cos - newLocalCy * sin;
                        const newStageCy = initCy + newLocalCx * sin + newLocalCy * cos;

                        l.x = newStageCx / scaleRatioX;
                        l.y = newStageCy / scaleRatioY;
                        const s = Number((newW / Math.max(1, activeLayerInitial.natW * scaleRatioX)).toFixed(4));
                        l.scaleX = s;
                        l.scaleY = s;
                        node.renderStage();
                    }
                }
            });

            stageCanvas.addEventListener("pointerup", (e) => {
                if (dragMode) {
                    dragMode = null;
                    stageCanvas.style.cursor = "default";
                    stageCanvas.releasePointerCapture(e.pointerId);
                    node.saveCanvasState();
                }
            });

            // Initial geometry setup at standard 640px canvas
            node.syncGeometry(640);
            node.renderPreviewDeck();
        };

        // Workflow reload / browser refresh handler: restores layout without void gap
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = origOnConfigure?.apply(this, arguments);
            const wCanvasWidth = this.widgets?.find(x => x.name === "canvas_width");
            const savedW = (wCanvasWidth && wCanvasWidth.value > 0)
                ? wCanvasWidth.value
                : ((this.size && this.size[0] > 0) ? this.size[0] - SIDE_SPACING * 2 : 640);
            this.syncGeometry(savedW);

            const wCanvasState = this.widgets?.find(x => x.name === "canvas_state");
            if (wCanvasState && wCanvasState.value) {
                try {
                    const parsed = JSON.parse(wCanvasState.value);
                    if (parsed && Array.isArray(parsed.layers)) {
                        for (const sl of parsed.layers) {
                            let layer = this.canvasLayers.find(l => l.id === sl.id);
                            if (!layer) {
                                layer = { id: sl.id };
                                this.canvasLayers.push(layer);
                            }
                            layer.x = sl.x;
                            layer.y = sl.y;
                            // Enforce uniform scale to un-distort any previously deformed layer states
                            const unifScale = (typeof sl.scaleX === "number") ? sl.scaleX : (sl.scaleY ?? 1.0);
                            layer.scaleX = unifScale;
                            layer.scaleY = unifScale;
                            layer.angle = sl.angle;
                            layer.flipX = sl.flipX;
                            layer.flipY = sl.flipY;
                            layer.opacity = sl.opacity;
                            layer.zIndex = sl.zIndex;
                            layer.userModified = true;
                        }
                    }
                } catch (e) { }
            }
            return res;
        };

        // Serialization hook: ensure latest layout is saved to canvas_state right before prompt execution
        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            if (this.saveCanvasState) {
                this.saveCanvasState();
            }
            return origOnSerialize ? origOnSerialize.apply(this, arguments) : undefined;
        };

        // Suppress native ComfyUI output images preview (which draws duplicate preview images below node)
        nodeType.prototype.onDrawBackground = function () {
            if (this.flags?.collapsed) return;
            if (this.imgs) this.imgs = null;
        };

        // Execution Handler - do NOT call origOnExecuted as it triggers native preview images at bottom
        nodeType.prototype.onExecuted = function (message) {
            this.hideOutputImages = true;
            this.imgs = null;
            if (!message || !message.images || !Array.isArray(message.images)) return;

            const node = this;
            message.images.forEach(imgMeta => {
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(imgMeta.filename)}` +
                    `&type=${encodeURIComponent(imgMeta.type || "temp")}` +
                    `&subfolder=${encodeURIComponent(imgMeta.subfolder || "")}` +
                    `&t=${Date.now()}`
                );
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => {
                    const slot = imgMeta.slot || "";
                    if (slot === "Background" || slot.toLowerCase() === "background") {
                        node.onBackgroundLoaded(img);
                    } else if (slot.startsWith("layer_")) {
                        node.onLayerImageLoaded(slot, img);
                    }
                };
                img.src = url;
            });
        };

        nodeType.prototype.tryLoadUpstreamImage = function (slotName) {
            const node = this;
            if (!node.inputs || !app.graph) return;
            const inp = node.inputs.find(i => i.name === slotName);
            if (!inp || inp.link == null) return;
            const link = app.graph.links[inp.link];
            if (!link) return;
            const originNode = app.graph.getNodeById(link.origin_id);
            if (!originNode) return;

            let imgSrc = null;
            if (originNode.imgs && originNode.imgs.length > 0 && originNode.imgs[0]?.src) {
                imgSrc = originNode.imgs[0].src;
            } else if (originNode.widgets) {
                const imgWidget = originNode.widgets.find(w => w.name === "image" || w.name === "image_path");
                if (imgWidget && imgWidget.value) {
                    imgSrc = api.apiURL(
                        `/view?filename=${encodeURIComponent(imgWidget.value)}` +
                        `&type=input` +
                        `&t=${Date.now()}`
                    );
                }
            }

            if (!imgSrc) return;

            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                if (slotName === "Background") {
                    node.onBackgroundLoaded(img);
                } else if (slotName.startsWith("layer_")) {
                    node.onLayerImageLoaded(slotName, img);
                }
            };
            img.src = imgSrc;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, isConnected, linkInfo) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }
            if (type === 1) { // Input connection
                if (!isConnected) {
                    const inp = this.inputs && this.inputs[index];
                    const slotName = inp ? inp.name : null;
                    if (slotName === "Background" || index === this.findInputSlot("Background")) {
                        this.bgImage = null;
                        this.setCanvasDimensions(this.currentW || 640, this.currentH || 360);
                    } else if (slotName && slotName.startsWith("layer_")) {
                        this.canvasLayers = (this.canvasLayers || []).filter(l => l.id !== slotName);
                        if (this.activeLayerId === slotName) {
                            this.activeLayerId = null;
                        }
                    }
                }
                this.manageDynamicSlots();
                if (isConnected && this.inputs && this.inputs[index]) {
                    this.tryLoadUpstreamImage(this.inputs[index].name);
                }
                this.saveCanvasState();
                this.renderPreviewDeck();
                this.renderStage();
            }
        };

        nodeType.prototype.manageDynamicSlots = function () {
            let maxConnectedLayer = 0;
            if (this.inputs) {
                for (let i = 0; i < this.inputs.length; i++) {
                    const input = this.inputs[i];
                    const hasLink = input.link != null && (!app.graph || !app.graph.links || app.graph.links[input.link] != null);
                    if (input.name.startsWith("layer_") && hasLink) {
                        const num = parseInt(input.name.split("_")[1], 10);
                        if (num > maxConnectedLayer) maxConnectedLayer = num;
                    }
                }
            }

            // Always provide at least layer_1, plus next available slot up to layer_4
            const desiredSlots = Math.min(5, Math.max(1, maxConnectedLayer + 1));
            for (let i = 2; i <= desiredSlots; i++) {
                const name = `layer_${i}`;
                if (!this.inputs.find(inp => inp.name === name)) {
                    this.addInput(name, "IMAGE");
                }
            }

            // Clean up trailing unconnected slots beyond desiredSlots
            if (this.inputs) {
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const input = this.inputs[i];
                    if (input.name.startsWith("layer_")) {
                        const num = parseInt(input.name.split("_")[1], 10);
                        const hasLink = input.link != null && (!app.graph || !app.graph.links || app.graph.links[input.link] != null);
                        if (num > desiredSlots && !hasLink) {
                            this.removeInput(i);
                        }
                    }
                }
            }
        };
    }
});

api.addEventListener("executed", (e) => {
    const detail = e?.detail;
    if (!detail || !app.graph) return;
    const canvasNodes = app.graph.findNodesByType("AILab_Canvas");
    if (!canvasNodes) return;
    for (const cNode of canvasNodes) {
        if (!cNode.inputs) continue;
        for (const inp of cNode.inputs) {
            if (inp.link != null) {
                const link = app.graph.links[inp.link];
                if (link && link.origin_id === detail.node) {
                    cNode.tryLoadUpstreamImage(inp.name);
                }
            }
        }
    }
});
