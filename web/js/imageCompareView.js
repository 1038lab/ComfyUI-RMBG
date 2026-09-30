import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/* ═══════════════════════════════════════════════════════════════════
 *  Image Compare (RMBG) — Frontend
 *  ───────────────────────────────────────────────────────────────────
 *  ComfyUI v2 compatible — follows official DOM widget patterns:
 *    • V1 (LiteGraph):  onDrawForeground + LiteGraph mouse hooks
 *    • V2 (Vue Nodes):   addDOMWidget with <canvas> + ResizeObserver
 *
 *  Key v2 rules:
 *    1. Never call setDirtyCanvas in V2 mode
 *    2. Use requestAnimationFrame to throttle DOM renders
 *    3. Only render on actual state changes, not every pointermove
 *    4. Clean up event listeners on node removal
 * ═══════════════════════════════════════════════════════════════════ */

// ── Mode string → internal numeric ID ────────────────────────────────
const MODE_MAP = {
  left_right: 0,
  up_down: 1,
  overlay: 2,
  difference: 3,
  side_by_side: 4,
  highlight_diff: 5,
};

const STATE_KEY = "compareState";
const CANVAS_BACKING_CAP = 6000;
const MIN_W = 320;
const MIN_H = 280;
const INIT_W = 440;
const INIT_H = 360;

// ── V2 detection ─────────────────────────────────────────────────────
function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}

// ── API helpers ──────────────────────────────────────────────────────
function pixApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch (_) { }
  return route;
}

function buildCmpUrl(d) {
  return pixApiUrl(
    `/view?filename=${encodeURIComponent(d.filename)}` +
    `&type=${encodeURIComponent(d.type)}` +
    `&subfolder=${encodeURIComponent(d.subfolder || "")}` +
    `&t=${Date.now()}`
  );
}

function loadCmpImage(node, meta, idx) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    if (idx === 0) node._cmpImg1 = img;
    else node._cmpImg2 = img;
    node.imgs = null;
    // Invalidate diff cache when images change
    node._cmpDiffCacheKey = null;
    node._cmpDiffCanvas = null;
    repaint(node);
  };
  img.src = buildCmpUrl(meta);
}

// ── State persistence (debounced) ────────────────────────────────────
let _saveTimer = null;
function saveCompareState(node) {
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    node.properties = node.properties || {};
    const prev = node.properties[STATE_KEY] || {};
    node.properties[STATE_KEY] = {
      opacity: node._cmpOpacity ?? 0.5,
      splitX: node._cmpSplitX ?? 0.5,
      splitY: node._cmpSplitY ?? 0.5,
      sideRatio: node._cmpSideRatio ?? 0.5,
      diffThreshold: node._cmpDiffThreshold ?? 38,
      images: prev.images || [],
    };
    _saveTimer = null;
  }, 300);
}

function saveCompareImagesToProps(node, d1, d2) {
  node.properties = node.properties || {};
  const clean = (d) =>
    d ? { filename: d.filename, subfolder: d.subfolder || "", type: d.type || "temp" } : null;
  node.properties[STATE_KEY] = {
    ...node.properties[STATE_KEY],
    images: [clean(d1), clean(d2)],
  };
}

function restoreCompareFromProperties(node) {
  if (node._cmpImg1 || node._cmpImg2) return;
  const s = node.properties?.[STATE_KEY];
  if (!s) return;
  if (typeof s.opacity === "number") node._cmpOpacity = s.opacity;
  if (typeof s.splitX === "number") node._cmpSplitX = s.splitX;
  if (typeof s.splitY === "number") node._cmpSplitY = s.splitY;
  if (typeof s.sideRatio === "number") node._cmpSideRatio = s.sideRatio;
  if (typeof s.diffThreshold === "number") node._cmpDiffThreshold = s.diffThreshold;
  if (Array.isArray(s.images)) {
    if (s.images[0]) loadCmpImage(node, s.images[0], 0);
    if (s.images[1]) loadCmpImage(node, s.images[1], 1);
  }
  repaint(node);
}

// ── Repaint dispatch ──────────────────────────────────────────────────
function repaint(node) {
  if (!node) return;
  if (isVueNodes()) {
    // V2: schedule a render via rAF — never touch setDirtyCanvas
    if (node._cmpRafPending) return;
    node._cmpRafPending = true;
    requestAnimationFrame(() => {
      node._cmpRafPending = false;
      node._cmpDomRender?.();
    });
  } else {
    node.setDirtyCanvas?.(true, false);
  }
}

// ── Image-fit helper ──────────────────────────────────────────────────
function fitImage(img, areaX, areaY, areaW, areaH) {
  if (!img || !img.naturalWidth) {
    return { x: areaX, y: areaY, w: areaW, h: areaH };
  }
  const ar = img.naturalWidth / img.naturalHeight;
  const fw = areaW;
  const fh = fw / ar;
  if (fh <= areaH) {
    return { x: areaX, y: areaY + (areaH - fh) / 2, w: areaW, h: fh };
  }
  const nh = areaH;
  const nw = nh * ar;
  return { x: areaX + (areaW - nw) / 2, y: areaY, w: nw, h: nh };
}

// ── Status bar helper ──────────────────────────────────────────────────
function drawStatusBar(ctx, rect, fill, color = "rgba(255,200,0,0.95)") {
  if (!rect || rect.w <= 0) return;
  const barH = 4;
  const barY = rect.y;
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(rect.x, barY, rect.w, barH);
  ctx.fillStyle = color;
  ctx.fillRect(rect.x, barY, Math.max(0, Math.min(1, fill)) * rect.w, barH);
}

// ════════════════════════════════════════════════════════════════════
//  Paint core — draws the interactive image area
// ════════════════════════════════════════════════════════════════════
function paintCompare(ctx, node, W, H, imgTop) {
  const imgH = H - imgTop;
  if (imgH <= 0) return;

  // ── Empty state ─────────────────────────────────────────────
  if (!node._cmpImg1 && !node._cmpImg2) {
    ctx.save();
    ctx.fillStyle = "#555";
    ctx.font = "12px 'Segoe UI',sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Connect images & run to compare", W / 2, imgTop + imgH / 2);
    ctx.restore();
    return;
  }

  const m = node._cmpMode;
  const img1 = node._cmpImg1;
  const img2 = node._cmpImg2;

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, imgTop, W, imgH);
  ctx.clip();

  // ── difference (static) ────────────────────────────────────
  if (m === 3) {
    const r1 = fitImage(img1 || img2, 0, imgTop, W, imgH);
    if (img1) ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
    if (img2) {
      const r2 = fitImage(img2, 0, imgTop, W, imgH);
      ctx.globalCompositeOperation = "difference";
      ctx.drawImage(img2, r2.x, r2.y, r2.w, r2.h);
      ctx.globalCompositeOperation = "source-over";
    }
    ctx.restore();
    return;
  }

  // ── overlay (opacity) ─────────────────────────────────────
  if (m === 2) {
    const r1 = fitImage(img1 || img2, 0, imgTop, W, imgH);
    if (img1) ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
    if (img2) {
      const r2 = fitImage(img2, 0, imgTop, W, imgH);
      ctx.globalAlpha = node._cmpOpacity;
      ctx.drawImage(img2, r2.x, r2.y, r2.w, r2.h);
      ctx.globalAlpha = 1;
    }
    if (img1 || img2) drawStatusBar(ctx, r1, node._cmpOpacity);
    ctx.restore();
    return;
  }

  // ── side_by_side (draggable divider) ──────────────────────
  if (m === 4) {
    const ratio = node._cmpSideRatio;
    const halfW = W * ratio;
    if (img1) {
      const r1 = fitImage(img1, 0, imgTop, halfW, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, imgTop, halfW, imgH);
      ctx.clip();
      ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
      ctx.restore();
    }
    if (img2) {
      const r2 = fitImage(img2, halfW, imgTop, W - halfW, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(halfW, imgTop, W - halfW, imgH);
      ctx.clip();
      ctx.drawImage(img2, r2.x, r2.y, r2.w, r2.h);
      ctx.restore();
    }
    if (ratio > 0.01 && ratio < 0.99) {
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(halfW, imgTop);
      ctx.lineTo(halfW, imgTop + imgH);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  // ── left_right (left/right wipe) ────────────────────
  if (m === 0) {
    const sx = W * node._cmpSplitX;
    if (img2) {
      const r2 = fitImage(img2, 0, imgTop, W, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(sx, imgTop, W - sx, imgH);
      ctx.clip();
      ctx.drawImage(img2, r2.x, r2.y, r2.w, r2.h);
      ctx.restore();
    }
    if (img1) {
      const r1 = fitImage(img1, 0, imgTop, W, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, imgTop, sx, imgH);
      ctx.clip();
      ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
      ctx.restore();
    }
    if (node._cmpSplitX > 0.01 && node._cmpSplitX < 0.99) {
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(sx, imgTop);
      ctx.lineTo(sx, imgTop + imgH);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  // ── up_down (up/down wipe) ──────────────────────
  if (m === 1) {
    const sy = imgTop + imgH * node._cmpSplitY;
    if (img2) {
      const r2 = fitImage(img2, 0, imgTop, W, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, sy, W, imgTop + imgH - sy);
      ctx.clip();
      ctx.drawImage(img2, r2.x, r2.y, r2.w, r2.h);
      ctx.restore();
    }
    if (img1) {
      const r1 = fitImage(img1, 0, imgTop, W, imgH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, imgTop, W, sy - imgTop);
      ctx.clip();
      ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
      ctx.restore();
    }
    if (node._cmpSplitY > 0.01 && node._cmpSplitY < 0.99) {
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(W, sy);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  // ── highlight_diff (grey base, red on difference) ─────────
  if (m === 5) {
    const r1 = fitImage(img1 || img2, 0, imgTop, W, imgH);
    if (img1) {
      ctx.drawImage(img1, r1.x, r1.y, r1.w, r1.h);
      ctx.save();
      ctx.globalCompositeOperation = "saturation";
      ctx.fillStyle = "hsl(0,0%,50%)";
      ctx.fillRect(r1.x, r1.y, r1.w, r1.h);
      ctx.restore();
    }
    if (img2 && img1) {
      const threshold = node._cmpDiffThreshold ?? 38;
      const cw = Math.min(Math.round(r1.w), 512);
      const ch = Math.min(Math.round(r1.h), 512);
      const cacheKey = img1.src + "|" + img2.src + "|" + threshold + "|" + cw + "x" + ch;
      if (node._cmpDiffCacheKey !== cacheKey) {
        node._cmpDiffCacheKey = cacheKey;
        const tmpC = document.createElement("canvas");
        tmpC.width = cw;
        tmpC.height = ch;
        const tctx = tmpC.getContext("2d");
        tctx.drawImage(img1, 0, 0, cw, ch);
        const d1 = tctx.getImageData(0, 0, cw, ch);
        tctx.clearRect(0, 0, cw, ch);
        tctx.drawImage(img2, 0, 0, cw, ch);
        const d2 = tctx.getImageData(0, 0, cw, ch);
        const out = tctx.createImageData(cw, ch);
        for (let i = 0; i < d1.data.length; i += 4) {
          const dr = Math.abs(d1.data[i] - d2.data[i]);
          const dg = Math.abs(d1.data[i + 1] - d2.data[i + 1]);
          const db = Math.abs(d1.data[i + 2] - d2.data[i + 2]);
          const avg = (dr + dg + db) / 3;
          if (avg > threshold) {
            out.data[i] = 255;
            out.data[i + 1] = 0;
            out.data[i + 2] = 0;
            out.data[i + 3] = 255;
          } else {
            out.data[i + 3] = 0;
          }
        }
        tctx.putImageData(out, 0, 0);
        node._cmpDiffCanvas = tmpC;
      }
      if (node._cmpDiffCanvas) {
        ctx.drawImage(node._cmpDiffCanvas, r1.x, r1.y, r1.w, r1.h);
      }
    }
    ctx.restore();
    return;
  }

  ctx.restore();
}

// ════════════════════════════════════════════════════════════════════
//  Mouse event handlers (shared logic)
// ════════════════════════════════════════════════════════════════════
function cmpMove(node, lx, ly, W, H, imgTop) {
  const m = node._cmpMode;
  const imgH = H - imgTop;
  const img1 = node._cmpImg1;
  const img2 = node._cmpImg2;
  const ir = fitImage(img1 || img2, 0, imgTop, W, imgH);
  const inImg = img1 || img2
    ? lx >= ir.x && lx <= ir.x + ir.w && ly >= ir.y && ly <= ir.y + ir.h
    : false;

  if (m === 4 && node._cmpDragging && inImg) {
    node._cmpSideRatio = Math.max(0.02, Math.min(0.98, lx / W));
    return true;
  }
  if (m === 2 && node._cmpDragging && inImg) {
    node._cmpOpacity = Math.max(0, Math.min(1, (lx - ir.x) / ir.w));
    return true;
  }
  if (m === 0 && inImg) {
    node._cmpSplitX = Math.max(0, Math.min(1, lx / W));
    return true;
  }
  if (m === 1 && inImg) {
    node._cmpSplitY = Math.max(0, Math.min(1, (ly - imgTop) / imgH));
    return true;
  }
  return false;
}

function cmpDown(node, lx, ly, W, H, imgTop) {
  const m = node._cmpMode;
  const imgH = H - imgTop;
  const img1 = node._cmpImg1;
  const img2 = node._cmpImg2;
  const ir = fitImage(img1 || img2, 0, imgTop, W, imgH);
  const inImg = img1 || img2
    ? lx >= ir.x && lx <= ir.x + ir.w && ly >= ir.y && ly <= ir.y + ir.h
    : false;

  if (m === 4 && inImg) {
    const halfW = W * node._cmpSideRatio;
    if (Math.abs(lx - halfW) < 12) {
      node._cmpDragging = "divider";
      node._cmpSideRatio = Math.max(0.02, Math.min(0.98, lx / W));
      return true;
    }
  }
  if ((m === 0 || m === 1) && inImg) {
    node._cmpDragging = "split";
    return cmpMove(node, lx, ly, W, H, imgTop);
  }
  if (m === 2 && inImg) {
    node._cmpDragging = "opacity";
    node._cmpOpacity = Math.max(0, Math.min(1, (lx - ir.x) / ir.w));
    return true;
  }
  return false;
}

function cmpUp(node) {
  if (node._cmpDragging) saveCompareState(node);
  node._cmpDragging = false;
}

function cmpWheel(node, ly, deltaY, imgTop) {
  const m = node._cmpMode;
  if (m === 2 && ly >= imgTop) {
    node._cmpOpacity = Math.max(0, Math.min(1, node._cmpOpacity + (deltaY > 0 ? -0.05 : 0.05)));
    saveCompareState(node);
    return true;
  }
  if (m === 5 && ly >= imgTop) {
    node._cmpDiffThreshold = Math.max(5, Math.min(120, (node._cmpDiffThreshold ?? 38) + (deltaY > 0 ? -3 : 3)));
    saveCompareState(node);
    return true;
  }
  return false;
}

function cmpLeave(node) {
  if (node._cmpDragging) saveCompareState(node);
  node._cmpDragging = false;
  if (node._cmpMode === 0) node._cmpSplitX = 0.5;
  else if (node._cmpMode === 1) node._cmpSplitY = 0.5;
}

// ════════════════════════════════════════════════════════════════════
//  V2 (Vue Nodes 2.0) DOM Widget
//  ─ Canvas inside a DOM container; events are local.
//  ─ Uses rAF-throttled render, only triggered by real state changes.
//  ─ NEVER calls setDirtyCanvas.
// ════════════════════════════════════════════════════════════════════
function canvasBackingScale(cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const zoom = Math.max(1, app.canvas?.ds?.scale || 1);
  let s = dpr * zoom;
  const longCss = Math.max(cssW || 0, cssH || 0);
  if (longCss > 0 && longCss * s > CANVAS_BACKING_CAP) {
    s = CANVAS_BACKING_CAP / longCss;
  }
  return s;
}

function createCompareDOMWidget(node) {
  const root = document.createElement("div");
  root.className = "ailab-cmp-root";
  root.style.cssText =
    "position:relative;width:100%;flex:1 1 0;min-height:0;box-sizing:border-box;cursor:default;";
  root._cmpNode = node;

  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;display:block;";
  root.appendChild(canvas);

  const widget = node.addDOMWidget("compare_viewer", "compare_viewer", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => MIN_H,
  });

  widget.computeLayoutSize = () => ({ minHeight: MIN_H, minWidth: 1 });

  // In V1, make widget canvas-only so LiteGraph doesn't render as HTML
  try {
    Object.defineProperty(widget.options, "canvasOnly", {
      configurable: true,
      enumerable: true,
      get() { return !window.LiteGraph?.vueNodesMode; },
    });
  } catch (_) {
    widget.options.canvasOnly = !window.LiteGraph?.vueNodesMode;
  }

  // ── rAF-throttled render — the ONLY render path for V2 ────
  // Scheduled via repaint(); never called directly from events.
  const render = () => {
    const cssW = root.clientWidth;
    const cssH = root.clientHeight;
    if (cssW <= 0 || cssH <= 0) return;

    const s = canvasBackingScale(cssW, cssH);
    const bw = Math.round(cssW * s);
    const bh = Math.round(cssH * s);
    if (canvas.width !== bw) canvas.width = bw;
    if (canvas.height !== bh) canvas.height = bh;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    node._cmpDomW = cssW;
    node._cmpDomH = cssH;
    paintCompare(ctx, node, cssW, cssH, 0);
  };
  node._cmpDomRender = render;

  // ── Local coordinate mapping ──────────────────────────────
  const localPos = (e) => {
    const r = root.getBoundingClientRect();
    const sx = r.width ? root.clientWidth / r.width : 1;
    const sy = r.height ? root.clientHeight / r.height : 1;
    return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  };
  const W = () => node._cmpDomW || root.clientWidth;
  const H = () => node._cmpDomH || root.clientHeight;

  // ── Pointer events — only repaint when state actually changes ──
  root.addEventListener("pointerdown", (e) => {
    const [lx, ly] = localPos(e);
    if (cmpDown(node, lx, ly, W(), H(), 0)) {
      e.stopPropagation();
      if (node._cmpDragging) {
        try { root.setPointerCapture(e.pointerId); } catch (_) { }
      }
      repaint(node);
    }
  });

  root.addEventListener("pointermove", (e) => {
    const [lx, ly] = localPos(e);
    const changed = cmpMove(node, lx, ly, W(), H(), 0);
    if (changed) {
      e.stopPropagation();
      repaint(node);
    }
    // Update cursor — cheap, no canvas repaint needed
    const m = node._cmpMode;
    const ir = fitImage(node._cmpImg1 || node._cmpImg2, 0, 0, W(), H());
    const inImg = (node._cmpImg1 || node._cmpImg2) &&
      lx >= ir.x && lx <= ir.x + ir.w && ly >= ir.y && ly <= ir.y + ir.h;
    const prevCursor = root.style.cursor;
    let nextCursor = "default";
    if (inImg && (m === 0 || m === 2 || m === 4)) nextCursor = "ew-resize";
    else if (inImg && m === 1) nextCursor = "ns-resize";
    if (prevCursor !== nextCursor) root.style.cursor = nextCursor;
  });

  root.addEventListener("pointerup", (e) => {
    cmpUp(node);
    try { root.releasePointerCapture(e.pointerId); } catch (_) { }
    repaint(node);
  });

  root.addEventListener("pointerleave", () => {
    cmpLeave(node);
    repaint(node);
  });

  // ── Wheel: only intercept in overlay/highlight_diff modes ───
  // In other modes, let the event bubble so graph zoom works.
  root.addEventListener("wheel", (e) => {
    const [, ly] = localPos(e);
    if (cmpWheel(node, ly, e.deltaY, 0)) {
      e.preventDefault();
      e.stopPropagation();
      repaint(node);
    }
  }, { passive: false });

  // ── ResizeObserver: repaint when container size changes ─────
  const ro = new ResizeObserver(() => repaint(node));
  ro.observe(root);
  node._cmpDomRO = ro;

  // Initial render
  repaint(node);
  return widget;
}

// ════════════════════════════════════════════════════════════════════
//  V1 (LiteGraph) helpers
// ════════════════════════════════════════════════════════════════════
function getV1WidgetBottomY(node) {
  const widgets = node.widgets || [];
  if (widgets.length > 0) {
    const last = widgets[widgets.length - 1];
    if (last && typeof last.last_y === "number") {
      return Math.ceil(last.last_y + (last.computedHeight || 20) + 6);
    }
  }
  const titleH = 30;
  const inputH = (node.inputs?.length || 0) * 26;
  const widgetH = widgets.length * 26;
  return Math.max(titleH + inputH + widgetH + 4, 60);
}

function getV1ImgTop(node) {
  return node._cmpImgTop ?? 60;
}

// ════════════════════════════════════════════════════════════════════
//  Register Extension
// ════════════════════════════════════════════════════════════════════
app.registerExtension({
  name: "AILab.ImageCompareView",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AILab_ImageCompareView") return;

    // ── onNodeCreated ──────────────────────────────────────────
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      _origCreated?.apply(this, arguments);

      this._cmpMode = 0;
      this._cmpSplitX = 0.5;
      this._cmpSplitY = 0.5;
      this._cmpOpacity = 0.5;
      this._cmpSideRatio = 0.5;
      this._cmpDiffThreshold = 38;
      this._cmpDragging = false;
      this._cmpImg1 = null;
      this._cmpImg2 = null;
      this._cmpImgTop = 54;

      this.hideOutputImages = true;
      this.size = [INIT_W, INIT_H];

      // V2: create DOM widget
      if (isVueNodes()) {
        createCompareDOMWidget(this);
      }

      // Hook the native "mode" widget
      const syncMode = () => {
        const w = this.widgets?.find((w) => w.name === "mode");
        if (!w) return;
        const val = w.value || w.options?.default || "left_right";
        this._cmpMode = MODE_MAP[val] ?? 0;
        const origCb = w.callback;
        w.callback = (...args) => {
          const v = args[0] || w.value;
          this._cmpMode = MODE_MAP[v] ?? 0;
          repaint(this);
          if (typeof origCb === "function") return origCb.apply(w, args);
        };
      };
      queueMicrotask(syncMode);

      // Restore from saved state (only if not already restored by onConfigure)
      queueMicrotask(() => {
        if (!this._cmpRestored) {
          this._cmpRestored = true;
          restoreCompareFromProperties(this);
        }
      });
    };

    // ── onConfigure (workflow load) ────────────────────────────
    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = _origConfigure ? _origConfigure.apply(this, arguments) : undefined;
      this._cmpRestored = true;
      restoreCompareFromProperties(this);
      return r;
    };

    // ── onExecuted ────────────────────────────────────────────
    nodeType.prototype.onExecuted = function (output) {
      const imgs = output?.images || [];
      const d1 = imgs.find((i) => i.slot === 1) || null;
      const d2 = imgs.find((i) => i.slot === 2) || null;

      saveCompareImagesToProps(this, d1, d2);

      if (d1) loadCmpImage(this, d1, 0);
      else this._cmpImg1 = null;
      if (d2) loadCmpImage(this, d2, 1);
      else this._cmpImg2 = null;
      repaint(this);
    };

    // ── onDrawBackground: suppress default preview ────────────
    nodeType.prototype.onDrawBackground = function () {
      if (this.flags?.collapsed) return;
      if (this.imgs) this.imgs = null;
    };

    // ── onDrawForeground (V1 only) ─────────────────────────────
    const _origDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (_origDraw) _origDraw.call(this, ctx);
      if (this.flags?.collapsed) return;
      if (isVueNodes()) return;

      if (this.size[0] < MIN_W) this.size[0] = MIN_W;
      if (this.size[1] < MIN_H) this.size[1] = MIN_H;

      this._cmpImgTop = getV1WidgetBottomY(this);
      paintCompare(ctx, this, this.size[0], this.size[1], this._cmpImgTop);
    };

    // ── V1 Mouse hooks ────────────────────────────────────────
    const _origDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (e, pos) {
      if (isVueNodes()) return _origDown ? _origDown.call(this, e, pos) : undefined;
      const imgTop = getV1ImgTop(this);
      if (cmpDown(this, pos[0], pos[1], this.size[0], this.size[1], imgTop)) {
        this.setDirtyCanvas(true, false);
        return true;
      }
      if (_origDown) return _origDown.call(this, e, pos);
    };

    const _origMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (e, pos) {
      if (isVueNodes()) return _origMove ? _origMove.call(this, e, pos) : undefined;
      const imgTop = getV1ImgTop(this);
      if (cmpMove(this, pos[0], pos[1], this.size[0], this.size[1], imgTop)) {
        this.setDirtyCanvas(true, false);
      }
      if (_origMove) return _origMove.call(this, e, pos);
    };

    const _origUp = nodeType.prototype.onMouseUp;
    nodeType.prototype.onMouseUp = function (e, pos) {
      if (isVueNodes()) return _origUp ? _origUp.call(this, e, pos) : undefined;
      cmpUp(this);
      if (_origUp) return _origUp.call(this, e, pos);
    };

    const _origWheel = nodeType.prototype.onMouseWheel;
    nodeType.prototype.onMouseWheel = function (e, pos) {
      if (isVueNodes()) return _origWheel ? _origWheel.call(this, e, pos) : undefined;
      const imgTop = getV1ImgTop(this);
      if (cmpWheel(this, pos[1], e.deltaY, imgTop)) {
        this.setDirtyCanvas(true, false);
        return true;
      }
      if (_origWheel) return _origWheel.call(this, e, pos);
    };

    const _origLeave = nodeType.prototype.onMouseLeave;
    nodeType.prototype.onMouseLeave = function (e) {
      if (isVueNodes()) return _origLeave ? _origLeave.call(this, e) : undefined;
      cmpLeave(this);
      this.setDirtyCanvas(true, false);
      if (_origLeave) return _origLeave.call(this, e);
    };

    // ── onResize ──────────────────────────────────────────────
    const _origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (e) {
      if (_origResize) _origResize.call(this, e);
      if (isVueNodes()) return;
      this.size[0] = Math.max(this.size[0], MIN_W);
      this.size[1] = Math.max(this.size[1], MIN_H);
    };

    // ── onRemoved: cleanup ────────────────────────────────────
    const _origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try { this._cmpDomRO?.disconnect(); } catch (_) { }
      this._cmpDomRO = null;
      this._cmpDomRender = null;
      this._cmpRafPending = false;
      this._cmpDiffCanvas = null;
      this._cmpDiffCacheKey = null;
      return _origRemoved ? _origRemoved.apply(this, arguments) : undefined;
    };
  },
});
