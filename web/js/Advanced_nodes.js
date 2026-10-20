/**
 * AILab - Collapsible Advanced Options Extension
 * Fully compatible with both WebUI 1.0 (LiteGraph) and WebUI 2.0 (Vue Canvas).
 * 
 * This integration script is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * If you incorporate or modify this code, please credit AILab as the original source:
 * https://github.com/1038lab
 */

import { app } from "/scripts/app.js";

const EXTENSION_NAME = "AILab.RMBG.AdvancedToggle";

const NODE_CONFIG = {
    "AILab_Yolo": [
        "model",
        "mask_count",
        "select_mask_index",
        "unload_model"
    ],
    "AILab_LoadImageAdvanced": [
        "image_path_or_URL",
        "image",
        "upload"
    ]
};
const TARGET_NODES = Object.keys(NODE_CONFIG);

function getNodeClass(node) {
    return node?.comfyClass || node?.type || "";
}

function isPreviewWidget(w) {
    return !!w && (w.name === "$$canvas-image-preview" || w.type === "IMAGE_PREVIEW" || (typeof w.name === "string" && w.name.toLowerCase().includes("preview")));
}

function hideWidget(w) {
    if (!w) return;
    if (!w._origType) {
        w._origType = w.type;
        w._origComputeSize = w.computeSize;
    }
    w.type = "hidden";
    w.hidden = true;
    w.options = w.options || {};
    w.options.hidden = true;
    w.computeSize = () => [0, -4];
    if (w.element) w.element.style.display = "none";
    if (w.inputEl) w.inputEl.style.display = "none";
}

function showWidget(w) {
    if (!w) return;
    w.type = w._origType || "text";
    w.hidden = false;
    if (w.options) w.options.hidden = false;
    if (w._origComputeSize) {
        w.computeSize = w._origComputeSize;
    } else {
        try { delete w.computeSize; } catch (_) { }
    }
    if (w.element) w.element.style.display = "";
    if (w.inputEl) w.inputEl.style.display = "";
}

function getControlsHeight(node) {
    const prev = node.widgets?.find(w => isPreviewWidget(w));
    let h = 0;
    if (prev) {
        const origLayout = prev.computeLayoutSize;
        const origComp = prev.computeSize;
        prev.computeLayoutSize = () => ({ minHeight: 0, minWidth: 1 });
        prev.computeSize = () => [0, -4];
        h = node.computeSize()[1];
        if (origLayout) prev.computeLayoutSize = origLayout;
        else delete prev.computeLayoutSize;
        if (origComp) prev.computeSize = origComp;
        else delete prev.computeSize;
    } else {
        h = node.computeSize()[1];
    }
    return h;
}

function applyVisibility(node) {
    if (!node || !node.widgets) return;

    const cls = getNodeClass(node);
    const primaryNames = NODE_CONFIG[cls] || [];
    const show = !!node._showAdvanced;

    const btn = node.widgets.find(w => w._isAdvBtn);
    if (btn) {
        const btnIdx = node.widgets.indexOf(btn);
        if (btnIdx !== -1) {
            node.widgets.splice(btnIdx, 1);
            const previewIdx = node.widgets.findIndex(w => isPreviewWidget(w));
            if (previewIdx !== -1) {
                node.widgets.splice(previewIdx, 0, btn);
            } else {
                node.widgets.push(btn);
            }
        }
    }

    for (const w of node.widgets) {
        if (w._isAdvBtn || isPreviewWidget(w)) continue;
        if (primaryNames.includes(w.name) || show) {
            showWidget(w);
        } else {
            hideWidget(w);
        }
    }

    const prevWidget = node.widgets.find(w => isPreviewWidget(w));
    if (prevWidget) delete prevWidget.computeSize;

    const minWidth = node.computeSize()[0] || 280;
    const currentWidth = (node.size && node.size[0] > 0) ? node.size[0] : 0;
    const targetWidth = Math.max(currentWidth, minWidth);
    const controlsHeight = getControlsHeight(node);

    let previewHeight = 0;
    if (node.imgs && node.imgs.length) {
        const img = node.imgs[node.imageIndex || 0] || node.imgs[0];
        if (img) {
            if (img.naturalWidth && img.naturalHeight) {
                const scale = Math.min(targetWidth / img.naturalWidth, 1);
                previewHeight = Math.round(img.naturalHeight * scale) + 15;
            } else {
                previewHeight = 220;
                if (!img._advHooked) {
                    img._advHooked = true;
                    const origOnload = img.onload;
                    img.onload = function () {
                        if (origOnload) origOnload.apply(this, arguments);
                        applyVisibility(node);
                    };
                }
            }
        }
    }

    if (prevWidget && previewHeight > 0) {
        prevWidget.computeLayoutSize = () => ({
            minHeight: previewHeight,
            prefHeight: previewHeight,
            minWidth: 1
        });
    }

    node.setSize([targetWidth, controlsHeight + previewHeight]);
    node.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas(true, true);
}

function setupNode(node) {
    if (node._advToggleInstalled) return;
    const cls = getNodeClass(node);
    if (!TARGET_NODES.includes(cls)) return;

    if (!node.widgets || node.widgets.length === 0) {
        setTimeout(() => setupNode(node), 30);
        return;
    }

    node._advToggleInstalled = true;
    node._showAdvanced = false;

    const imgWidget = node.widgets.find(w => w.name === "image");
    if (imgWidget && !imgWidget._advHooked) {
        imgWidget._advHooked = true;
        const origCallback = imgWidget.callback;
        imgWidget.callback = function () {
            const res = origCallback ? origCallback.apply(this, arguments) : undefined;
            setTimeout(() => applyVisibility(node), 60);
            return res;
        };
    }

    const btn = node.addWidget("button", "Advanced", null, () => {
        node._showAdvanced = !node._showAdvanced;
        applyVisibility(node);
    });
    btn._isAdvBtn = true;
    btn.serialize = false;
    btn.computeSize = () => [0, 26];

    Object.defineProperty(btn, "width", {
        get() { return undefined; },
        set() {}
    });

    btn.draw = function (ctx, n, widgetWidth, y, widgetHeight) {
        const margin = 15;
        const x = margin;
        const w = widgetWidth - margin * 2;
        const h = widgetHeight || 24;
        const radius = 8;

        const isClicked = !!this.clicked;
        this.clicked = false;

        ctx.save();
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, w, h, radius);
        } else {
            ctx.moveTo(x + radius, y);
            ctx.arcTo(x + w, y, x + w, y + h, radius);
            ctx.arcTo(x + w, y + h, x, y + h, radius);
            ctx.arcTo(x, y + h, x, y, radius);
            ctx.arcTo(x, y, x + w, y, radius);
        }
        ctx.closePath();

        ctx.fillStyle = isClicked ? "#282828" : "#181818";
        ctx.fill();

        ctx.strokeStyle = isClicked ? "rgba(255, 255, 255, 0.22)" : "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = isClicked ? "#ffffff" : "#8e9297";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Advanced", x + w / 2, y + h / 2);

        ctx.restore();

        if (isClicked) {
            n?.setDirtyCanvas?.(true);
        }
    };

    const origOnDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        const b = this.widgets?.find(w => w._isAdvBtn);
        if (b) {
            const bIdx = this.widgets.indexOf(b);
            if (this.widgets.slice(bIdx + 1).some(w => !isPreviewWidget(w))) {
                applyVisibility(this);
            }
        }
        return origOnDrawForeground ? origOnDrawForeground.apply(this, arguments) : undefined;
    };

    setTimeout(() => {
        applyVisibility(node);
    }, 30);
}

app.registerExtension({
    name: EXTENSION_NAME,
    async setup() {
        if (app.graph?._nodes) {
            for (const node of app.graph._nodes) {
                if (TARGET_NODES.includes(getNodeClass(node))) {
                    setupNode(node);
                }
            }
        }
    },
    async nodeCreated(node) {
        if (TARGET_NODES.includes(getNodeClass(node))) {
            setTimeout(() => setupNode(node), 20);
        }
    },
    async loadedGraphNode(node) {
        if (TARGET_NODES.includes(getNodeClass(node))) {
            setTimeout(() => setupNode(node), 20);
        }
    }
});
