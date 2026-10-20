import { app } from "/scripts/app.js";
import { SLOT_AFFINITY } from "./appearance.js";

const EXTENSION_NAME = "AILab.WorkflowUtils";

export function clamp(val, min, max) {
    return Math.min(Math.max(val, min), max);
}

export function debounce(fn, delay = 200) {
    let timer = null;
    return function (...args) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

export function hexToRgb(hex) {
    if (!hex) return [0, 0, 0];
    const clean = hex.replace("#", "");
    const bigint = parseInt(clean, 16);
    if (clean.length === 6) {
        return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
    }
    return [0, 0, 0];
}

const g = typeof window !== "undefined" ? window : globalThis;
g.__AILAB_WORKFLOW_UTILS__ = g.__AILAB_WORKFLOW_UTILS__ || {
    configs: [],
    register(cfg) {
        if (cfg) this.configs.push(cfg);
    },
    getMergedRules() {
        const merged = { input: {}, output: {} };
        for (const c of this.configs) {
            for (const dir of ["input", "output"]) {
                const rules = c.rules?.[dir] || {};
                for (const [slot, list] of Object.entries(rules)) {
                    merged[dir][slot] = merged[dir][slot] || [];
                    for (const name of list) {
                        if (!merged[dir][slot].includes(name)) {
                            merged[dir][slot].push(name);
                        }
                    }
                }
            }
        }
        return merged;
    },
    getMergedMetrics() {
        const merged = {};
        for (const c of this.configs) {
            Object.assign(merged, c.metrics || {});
        }
        return merged;
    },
    isMatchingNode(node) {
        if (!node) return false;
        const name = node.name || node.type || node.comfyClass || "";
        if (name.startsWith("AILab_") || name === "RMBG" || name.startsWith("SAM")) return true;
        const category = node.category || "";
        if (category.includes("AILab") || category.includes("RMBG")) return true;
        const displayName = node.display_name || "";
        return displayName.includes("(RMBG)") || displayName.includes("AILab");
    }
};

if (SLOT_AFFINITY) {
    g.__AILAB_WORKFLOW_UTILS__.register(SLOT_AFFINITY);
}

if (typeof window !== "undefined" && window.fetch && !g.__AILAB_WORKFLOW_UTILS__._fetchHooked) {
    const origFetch = window.fetch;
    window.fetch = async function (...args) {
        const res = await origFetch.apply(this, args);
        const url = typeof args[0] === "string" ? args[0] : (args[0]?.url || "");
        if (url.includes("sorted-custom-node-map.json")) {
            try {
                const clone = res.clone();
                const data = await clone.json();
                Object.assign(data, g.__AILAB_WORKFLOW_UTILS__.getMergedMetrics());
                return new Response(JSON.stringify(data), {
                    status: res.status,
                    statusText: res.statusText,
                    headers: res.headers
                });
            } catch (_) {
                return res;
            }
        }
        return res;
    };
    g.__AILAB_WORKFLOW_UTILS__._fetchHooked = true;
}

function reorderResults(results, filters, query) {
    if (!Array.isArray(results) || results.length === 0) return results;

    const trimmed = (query || "").trim().toLowerCase();
    const hasSlot = filters && filters.length > 0;

    let dir = "";
    let slotType = "";

    if (hasSlot) {
        const f = filters.find(item => {
            const fid = item.filterDef?.id || item.id || "";
            const seq = item.filterDef?.invokeSequence || "";
            return fid === "input" || fid === "output" || seq === "i" || seq === "o";
        });
        if (f) {
            const fid = f.filterDef?.id || f.id || "";
            const seq = f.filterDef?.invokeSequence || "";
            dir = (fid === "input" || seq === "i") ? "input" : "output";
            slotType = String(f.value || "").toUpperCase();
        }
    }

    const mergedRules = g.__AILAB_WORKFLOW_UTILS__.getMergedRules();

    if (dir && slotType && (!trimmed || trimmed === "rmbg" || trimmed === "ailab")) {
        const priority = mergedRules[dir]?.[slotType] || [];
        if (priority.length > 0) {
            const pinned = [];
            const rest = [];

            for (const name of priority) {
                const found = results.find(n => (n.name || n.type) === name);
                if (found) pinned.push(found);
            }

            const pinnedSet = new Set(pinned.map(p => p.name || p.type));

            for (const item of results) {
                const name = item.name || item.type;
                if (!pinnedSet.has(name)) {
                    rest.push(item);
                }
            }

            if (dir === "output") {
                const previewSet = new Set(["AILab_ImagePreview", "AILab_Preview", "AILab_MaskPreview"]);
                const normal = [];
                const previews = [];
                for (const item of rest) {
                    if (previewSet.has(item.name || item.type)) {
                        previews.push(item);
                    } else {
                        normal.push(item);
                    }
                }
                return [...pinned, ...normal, ...previews];
            }

            if (pinned.length > 0) {
                return [...pinned, ...rest];
            }
        }
    }

    if (trimmed && (trimmed === "rmbg" || trimmed === "sam" || trimmed === "ailab")) {
        const matched = [];
        const normal = [];

        for (const item of results) {
            if (g.__AILAB_WORKFLOW_UTILS__.isMatchingNode(item)) {
                matched.push(item);
            } else {
                normal.push(item);
            }
        }

        if (dir && slotType && matched.length > 0) {
            const priority = mergedRules[dir]?.[slotType] || [];
            matched.sort((a, b) => {
                const aIdx = priority.indexOf(a.name || a.type || "");
                const bIdx = priority.indexOf(b.name || b.type || "");
                return (aIdx !== -1 ? aIdx : 999) - (bIdx !== -1 ? bIdx : 999);
            });
        }

        return [...matched, ...normal];
    }

    return results;
}

function hookService(service) {
    if (!service || service._ailabHooked) return;

    const ServiceClass = service.constructor;
    const proto = (ServiceClass && ServiceClass.prototype && typeof ServiceClass.prototype.searchNode === "function")
        ? ServiceClass.prototype
        : service;

    const origSearch = proto.searchNode;
    if (typeof origSearch !== "function") return;

    proto.searchNode = function (query, filters = [], options = {}, extra = {}) {
        const origLimit = options?.limit || 64;
        const opt = { ...options };
        if (opt.limit) delete opt.limit;

        const full = origSearch.call(this, query, filters, opt, extra);
        const reordered = reorderResults(full, filters, query);
        return reordered.slice(0, origLimit);
    };

    service._ailabHooked = true;
    if (ServiceClass && ServiceClass.prototype) {
        ServiceClass.prototype._ailabHooked = true;
    }
}

function connectService() {
    try {
        const vue = document.getElementById("vue-app")?.__vue_app__;
        let pinia = vue?.config?.globalProperties?.$pinia;

        if (!pinia && vue?._context?.provides) {
            for (const sym of Object.getOwnPropertySymbols(vue._context.provides)) {
                const val = vue._context.provides[sym];
                if (val && typeof val === "object" && val._s instanceof Map) {
                    pinia = val;
                    break;
                }
            }
        }

        if (pinia && pinia._s instanceof Map) {
            const store = pinia._s.get("nodeDef");
            if (store?.nodeSearchService) {
                hookService(store.nodeSearchService);
                return true;
            }
        }
    } catch (_) { }
    return false;
}

function initLiteGraphDefaults() {
    if (typeof LiteGraph === "undefined") return;

    LiteGraph.slot_types_default_out = LiteGraph.slot_types_default_out || {};
    LiteGraph.slot_types_default_in = LiteGraph.slot_types_default_in || {};

    for (const c of g.__AILAB_WORKFLOW_UTILS__.configs) {
        if (c.defaultSlots?.out) {
            for (const [slot, list] of Object.entries(c.defaultSlots.out)) {
                const cur = LiteGraph.slot_types_default_out[slot] || [];
                const add = list.filter(n => !cur.includes(n));
                if (add.length > 0) {
                    LiteGraph.slot_types_default_out[slot] = [...add, ...cur];
                }
            }
        }
        if (c.defaultSlots?.in) {
            for (const [slot, list] of Object.entries(c.defaultSlots.in)) {
                const cur = LiteGraph.slot_types_default_in[slot] || [];
                const add = list.filter(n => !cur.includes(n));
                if (add.length > 0) {
                    LiteGraph.slot_types_default_in[slot] = [...add, ...cur];
                }
            }
        }
    }
}

app.registerExtension({
    name: EXTENSION_NAME,
    async setup() {
        initLiteGraphDefaults();

        let attempts = 0;
        const timer = setInterval(() => {
            attempts++;
            if (connectService() || attempts > 25) {
                clearInterval(timer);
            }
        }, 200);

        window.addEventListener("pointerdown", () => connectService(), { passive: true });
    }
});
