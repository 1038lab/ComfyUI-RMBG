import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    segment: { nodeColor: "#222e40", nodeBgColor: "#364254", width: 340 },
    utility: { nodeColor: "#2e3e57", nodeBgColor: "#4b5b73", width: 300 },
    canvas: { nodeColor: "#222e40", nodeBgColor: "#364254", width: 640 },
};

const NODE_COLORS = {
    // Segmentation nodes
    "RMBG": "segment",
    "Segment": "segment",
    "SegmentV2": "segment",
    "FaceSegment": "segment",
    "ClothesSegment": "segment",
    "BodySegment": "segment",
    "FacialSegment": "segment",
    "FashionSegmentAccessories": "segment",
    "FashionSegmentClothing": "segment",
    "BiRefNetRMBG": "segment",
    "SAM2Segment": "segment",
    "SAM2SegmentDiscovery": "segment",
    "SAM2SegmentDiscoveryAdv": "segment",
    "SAM3Segment": "segment",
    "AILab_Florence2": "segment",
    "AILab_Yolo": "segment",
    "AILab_VRMBG": "segment",
    "AILab_FeyNoBg": "segment",
    "AILab_FinegrainBoxSegment": "segment",

    // Utility nodes
    "AILab_LoadImage": "utility",
    "AILab_LoadImageSimple": "utility",
    "AILab_LoadImageAdvanced": "utility",
    "AILab_LoadImageBatch": "utility",
    "AILab_UnbatchImages": "utility",
    "AILab_Preview": "utility",
    "AILab_ImagePreview": "utility",
    "AILab_MaskPreview": "utility",
    "AILab_ImageMaskConvert": "utility",
    "AILab_MaskEnhancer": "utility",
    "AILab_MaskCombiner": "utility",
    "AILab_MaskOverlay": "utility",
    "AILab_ImageCombiner": "utility",
    "AILab_MaskExtractor": "utility",
    "AILab_ImageStitch": "utility",
    "AILab_ImageCrop": "utility",
    "AILab_ICLoRAConcat": "utility",
    "AILab_CropObject": "utility",
    "AILab_ImageCompare": "utility",
    "AILab_ColorInput": "utility",
    "AILab_ReferenceLatentMask": "utility",
    "AILab_LamaRemover": "utility",
    "AILab_SDMatte": "utility",
    "AILab_ImageResize": "utility",
    "AILab_ImageToList": "utility",
    "AILab_MaskToList": "utility",
    "AILab_ImageMaskToList": "utility",
    "AILab_ColorToMask": "utility",
    "AILab_Resolution_Selector": "utility",
    "AILab_ImageCompareView": "utility",
    "AILab_Canvas": "canvas",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "RMBG.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

export const SLOT_AFFINITY = {
    rules: {
        input: {
            IMAGE: [
                "AILab_ImagePreview",
                "AILab_Preview",
                "AILab_ImageCompareView",
                "RMBG"
            ],
            MASK: [
                "AILab_MaskPreview",
                "AILab_Preview",
                "AILab_MaskOverlay"
            ],
            BBOX: [
                "AILab_FinegrainBoxSegment"
            ],
            BBOXES: [
                "AILab_FinegrainBoxSegment"
            ]
        },
        output: {
            IMAGE: [
                "AILab_LoadImage",
                "AILab_LoadImageSimple",
                "AILab_LoadImageAdvanced"
            ],
            MASK: [
                "RMBG",
                "AILab_ColorToMask",
                "SAM3Segment"
            ],
            BBOX: [
                "AILab_Florence2",
                "SAM3Segment"
            ],
            BBOXES: [
                "AILab_Florence2",
                "SAM3Segment"
            ]
        }
    },
    metrics: {
        "AILab_Preview": 4600,
        "AILab_ImagePreview": 4500,
        "AILab_LoadImageSimple": 4500,
        "RMBG": 4500,
        "AILab_LoadImage": 4400,
        "AILab_LoadImageAdvanced": 4300,
        "AILab_ColorToMask": 4200,
        "SAM3Segment": 4200,
        "AILab_Florence2": 4200,
        "AILab_FinegrainBoxSegment": 4200
    },
    defaultSlots: {
        out: {
            IMAGE: ["AILab_ImagePreview"],
            MASK: ["AILab_MaskPreview"],
            BBOX: ["AILab_FinegrainBoxSegment"],
            BBOXES: ["AILab_FinegrainBoxSegment"]
        },
        in: {
            IMAGE: ["AILab_LoadImage"],
            MASK: ["RMBG"],
            BBOX: ["AILab_Florence2", "SAM3Segment"],
            BBOXES: ["AILab_Florence2", "SAM3Segment"]
        }
    }
};

app.registerExtension(ext);
