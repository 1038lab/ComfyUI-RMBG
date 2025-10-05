import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    segment: { nodeColor: "#222e40", nodeBgColor: "#364254", width: 340 },
    utility: { nodeColor: "#2e3e57", nodeBgColor: "#4b5b73", width: 300 },
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

    // Utility nodes
    "AILab_LoadImage": "utility",
    "AILab_LoadImageSimple": "utility",
    "AILab_LoadImageAdvanced": "utility",
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

app.registerExtension(ext);