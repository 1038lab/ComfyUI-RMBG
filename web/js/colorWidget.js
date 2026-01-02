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

            const drawHeight = 22;
            const margin = 15;
            const radius = 10;

            ctx.fillStyle = this.value;
            ctx.beginPath();
            const x = margin;
            const y = widgetY + (height - drawHeight) / 2;
            const w = widgetWidth - margin * 2;
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
            ctx.fillText(text, widgetWidth * 0.5, y + drawHeight * 0.65);
        };

        widget.mouse = function (e, pos, node) {
            if (e.type === 'pointerdown') {
                const margin = 15;

                if (pos[0] >= margin && pos[0] <= node.size[0] - margin) {
                    const picker = document.createElement('input');
                    picker.type = 'color';
                    picker.value = this.value;

                    picker.style.position = 'absolute';
                    picker.style.left = '-9999px';
                    picker.style.top = '-9999px';

                    document.body.appendChild(picker);

                    picker.addEventListener('change', () => {
                        this.value = picker.value;
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                        picker.remove();
                    });

                    picker.click();
                    return true;
                }
            }
            return false;
        };

        widget.computeSize = function (width) {
            return [width, 22];
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