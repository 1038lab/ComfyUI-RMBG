# Based on code from ComfyUI-Florence2 by Jukka Seppänen (kijai), MIT License
"""Florence-2 processor: image preprocessing, tokenization, task prompts, and post-processing."""

import re
import numpy as np
import torch


def preprocess_image(image, size=768, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """[B, C, H, W] or [C, H, W] float [0,1] -> [B, C, size, size] normalized."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=(size, size), mode='bicubic', align_corners=False).clamp(0, 1)
    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean_t) / std_t


class BoxQuantizer:
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def dequantize(self, boxes, size):
        bins_w, bins_h = self.bins
        size_w, size_h = size
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
        return torch.cat((
            (xmin + 0.5) * size_w / bins_w, (ymin + 0.5) * size_h / bins_h,
            (xmax + 0.5) * size_w / bins_w, (ymax + 0.5) * size_h / bins_h,
        ), dim=-1)


class CoordinatesQuantizer:
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def dequantize(self, coordinates, size):
        bins_w, bins_h = self.bins
        size_w, size_h = size
        x, y = coordinates.split(1, dim=-1)
        return torch.cat(((x + 0.5) * size_w / bins_w, (y + 0.5) * size_h / bins_h), dim=-1)


class PostProcessor:
    """Regex-based parsing of Florence-2 text outputs into structured results."""

    def __init__(self, tokenizer=None):
        config = {
            'NUM_BBOX_HEIGHT_BINS': 1000, 'NUM_BBOX_WIDTH_BINS': 1000, 'BOX_QUANTIZATION_MODE': 'floor',
            'COORDINATES_HEIGHT_BINS': 1000, 'COORDINATES_WIDTH_BINS': 1000, 'COORDINATES_QUANTIZATION_MODE': 'floor',
            'PARSE_TASKS': [
                {'TASK_NAME': 'od', 'PATTERN': r'([a-zA-Z0-9 ]+)<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>'},
                {'TASK_NAME': 'ocr', 'PATTERN': r'(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', 'AREA_THRESHOLD': 0.00},
                {'TASK_NAME': 'phrase_grounding', 'FILTER_BY_BLACK_LIST': True},
                {'TASK_NAME': 'pure_text'}, {'TASK_NAME': 'description_with_bboxes'},
                {'TASK_NAME': 'description_with_polygons'}, {'TASK_NAME': 'polygons'},
                {'TASK_NAME': 'bboxes'}, {'TASK_NAME': 'description_with_bboxes_or_polygons'},
            ],
        }
        self.config = config
        self.parse_tasks = [t['TASK_NAME'] for t in config['PARSE_TASKS']]
        self.parse_tasks_configs = {t['TASK_NAME']: t for t in config['PARSE_TASKS']}

        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.all_special_tokens = set(tokenizer.all_special_tokens)

        self.box_quantizer = BoxQuantizer('floor', (1000, 1000))
        self.coordinates_quantizer = CoordinatesQuantizer('floor', (1000, 1000))

        self.black_list_of_phrase_grounding = set()
        if 'phrase_grounding' in self.parse_tasks and self.parse_tasks_configs['phrase_grounding'].get('FILTER_BY_BLACK_LIST'):
            self.black_list_of_phrase_grounding = {
                'it', 'I', 'me', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his',
                'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'one', 'oneself',
                'we', 'us', 'our', 'ours', 'mine', 'yours', 'his', 'hers', 'its',
                'ours', 'yours', 'theirs', 'myself', 'yourself', 'himself', 'herself',
                'itself', 'ourselves', 'yourselves', 'themselves', 'this', 'that',
                'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'that',
                'all', 'another', 'any', 'anybody', 'anyone', 'anything',
                'each', 'everybody', 'everyone', 'everything',
                'few', 'many', 'nobody', 'none', 'one', 'several',
                'some', 'somebody', 'someone', 'something',
                'each other', 'one another',
                'the image', 'image', 'images', 'the', 'a', 'an', 'a group',
                'other objects', 'lots', 'a set',
            }

    def parse_od_from_text_and_spans(self, text, pattern, image_size, phrase_centric=False):
        parsed = list(re.finditer(pattern, text))
        instances = []
        for m in parsed:
            if phrase_centric:
                bbox_bins = [int(m.group(j)) for j in range(2, 6)]
                cat_name = m.group(1).lower().strip()
            else:
                bbox_bins = [int(m.group(j)) for j in range(1, 5)]
                cat_name = m.group(5).lower().strip()
            instances.append({
                'bbox': self.box_quantizer.dequantize(torch.tensor(bbox_bins), image_size).tolist(),
                'cat_name': cat_name,
            })
        return instances

    def parse_ocr_from_text_and_spans(self, text, pattern, image_size, area_threshold=-1.0):
        text = text.replace('<s>', '')
        parsed = re.findall(pattern, text)
        instances = []
        image_width, image_height = image_size
        for ocr_line in parsed:
            ocr_content = ocr_line[0]
            quad_box = [int(i) for i in ocr_line[1:]]
            quad_box = self.coordinates_quantizer.dequantize(torch.tensor(np.array(quad_box).reshape(-1, 2)), image_size).reshape(-1).tolist()
            if area_threshold > 0:
                x_coords, y_coords = quad_box[0::2], quad_box[1::2]
                area = 0.5 * abs(sum(x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i] for i in range(3)))
                if area < (image_width * image_height) * area_threshold:
                    continue
            instances.append({'quad_box': quad_box, 'text': ocr_content})
        return instances

    def parse_phrase_grounding_from_text_and_spans(self, text, pattern, image_size):
        text = text.replace('<s>', '').replace('</s>', '').replace('<pad>', '')
        pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)
        phrase_pattern = r'^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)'
        box_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace('<ground>', '', 1).replace('<obj>', '', 1)
            if phrase_text_strip == '':
                continue
            phrase = re.search(phrase_pattern, phrase_text_strip)
            if phrase is None:
                continue
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if not bboxes_parsed:
                continue
            phrase = phrase.group().strip()
            if phrase in self.black_list_of_phrase_grounding:
                continue
            bbox_bins = [[int(b.group(j)) for j in range(1, 5)] for b in bboxes_parsed]
            phrase = phrase.encode('ascii', errors='ignore').decode('ascii')
            instances.append({
                'bbox': self.box_quantizer.dequantize(torch.tensor(bbox_bins), image_size).tolist(),
                'cat_name': phrase,
            })
        return instances

    def parse_description_with_bboxes_from_text_and_spans(self, text, pattern, image_size, allow_empty_phrase=False):
        text = text.replace('<s>', '').replace('</s>', '').replace('<pad>', '')
        if allow_empty_phrase:
            pattern = r"(?:(?:<loc_\d+>){{4,}})"
        else:
            pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)
        phrase_pattern = r'^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)'
        box_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace('<ground>', '', 1).replace('<obj>', '', 1)
            if phrase_text_strip == '' and not allow_empty_phrase:
                continue
            phrase = re.search(phrase_pattern, phrase_text_strip)
            if phrase is None:
                continue
            phrase = phrase.group().strip()
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if not bboxes_parsed:
                continue
            bbox_bins = [[int(b.group(j)) for j in range(1, 5)] for b in bboxes_parsed]
            bboxes = self.box_quantizer.dequantize(torch.tensor(bbox_bins), image_size).tolist()
            phrase = phrase.encode('ascii', errors='ignore').decode('ascii')
            for bbox in bboxes:
                instances.append({'bbox': bbox, 'cat_name': phrase})
        return instances

    def parse_description_with_polygons_from_text_and_spans(self, text, pattern, image_size,
                                                            allow_empty_phrase=False, polygon_sep_token='<sep>',
                                                            polygon_start_token='<poly>', polygon_end_token='</poly>',
                                                            with_box_at_start=False):
        text = text.replace('<s>', '').replace('</s>', '').replace('<pad>', '')
        sep, start, end = re.escape(polygon_sep_token), re.escape(polygon_start_token), re.escape(polygon_end_token)
        if allow_empty_phrase:
            pattern = rf"(?:(?:<loc_\d+>|{sep}|{start}|{end}){{4,}})"
        else:
            pattern = rf"([^<]+(?:<loc_\d+>|{sep}|{start}|{end}){{4,}})"
        phrases = re.findall(pattern, text)
        phrase_string_pattern = r'^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)'
        box_pattern = rf'((?:<loc_\d+>)+)(?:{sep}|$)'
        polygons_instance_pattern = rf'{start}(.*?){end}'

        instances = []
        for phrase_text in phrases:
            phrase_text_strip = re.sub(r'^loc_\d+>', '', phrase_text, count=1)
            if phrase_text_strip == '' and not allow_empty_phrase:
                continue
            phrase = re.search(phrase_string_pattern, phrase_text_strip)
            if phrase is None:
                continue
            phrase = phrase.group().strip()

            if polygon_start_token in phrase_text and polygon_end_token in phrase_text:
                poly_instances = list(re.finditer(polygons_instance_pattern, phrase_text))
            else:
                poly_instances = [phrase_text]

            for pi in poly_instances:
                poly_text = pi.group(1) if not isinstance(pi, str) else pi
                polygons_parsed = list(re.finditer(box_pattern, poly_text))
                if not polygons_parsed:
                    continue

                bbox, polygons = [], []
                for pp in polygons_parsed:
                    coords = [int(m.group(1)) for m in re.finditer(r'<loc_(\d+)>', pp.group(1))]
                    if with_box_at_start and not bbox:
                        if len(coords) > 4:
                            bbox = coords[:4]
                            coords = coords[4:]
                        else:
                            bbox = [0, 0, 0, 0]
                    if len(coords) % 2 == 1:
                        coords = coords[:-1]
                    polygons.append(self.coordinates_quantizer.dequantize(
                        torch.tensor(np.array(coords).reshape(-1, 2)), image_size,
                    ).reshape(-1).tolist())

                instance = {'cat_name': phrase, 'polygons': polygons}
                if bbox:
                    instance['bbox'] = self.box_quantizer.dequantize(torch.tensor([bbox]), image_size).tolist()[0]
                instances.append(instance)
        return instances

    def __call__(self, text=None, image_size=None, parse_tasks=None):
        if parse_tasks is not None:
            if isinstance(parse_tasks, str):
                parse_tasks = [parse_tasks]
            for t in parse_tasks:
                assert t in self.parse_tasks, f'parse task {t} not supported'

        assert text is not None, 'text should be provided'
        parsed_dict = {'text': text}

        for task in self.parse_tasks:
            if parse_tasks is not None and task not in parse_tasks:
                continue
            pattern = self.parse_tasks_configs[task].get('PATTERN', None)

            if task == 'ocr':
                parsed_dict['ocr'] = self.parse_ocr_from_text_and_spans(text, pattern, image_size, self.parse_tasks_configs[task].get('AREA_THRESHOLD', 0.0))
            elif task == 'phrase_grounding':
                parsed_dict['phrase_grounding'] = self.parse_phrase_grounding_from_text_and_spans(text, pattern, image_size)
            elif task == 'pure_text':
                parsed_dict['pure_text'] = text
            elif task == 'description_with_bboxes':
                parsed_dict['description_with_bboxes'] = self.parse_description_with_bboxes_from_text_and_spans(text, pattern, image_size)
            elif task == 'description_with_polygons':
                parsed_dict['description_with_polygons'] = self.parse_description_with_polygons_from_text_and_spans(text, pattern, image_size)
            elif task == 'polygons':
                parsed_dict['polygons'] = self.parse_description_with_polygons_from_text_and_spans(text, pattern, image_size, allow_empty_phrase=True)
            elif task == 'bboxes':
                parsed_dict['bboxes'] = self.parse_description_with_bboxes_from_text_and_spans(text, pattern, image_size, allow_empty_phrase=True)
            elif task == 'description_with_bboxes_or_polygons':
                if '<poly>' in text:
                    parsed_dict['description_with_bboxes_or_polygons'] = self.parse_description_with_polygons_from_text_and_spans(text, pattern, image_size)
                else:
                    parsed_dict['description_with_bboxes_or_polygons'] = self.parse_description_with_bboxes_from_text_and_spans(text, pattern, image_size)
            else:
                raise ValueError(f"task {task} is not supported")
        return parsed_dict


class Processor:
    def __init__(self, model_path):
        from .tokenizer import Florence2Tokenizer
        self.tokenizer = Florence2Tokenizer(model_path)
        self.image_seq_length = 577

        self.tasks_answer_post_processing_type = {
            '<OCR>': 'pure_text', '<OCR_WITH_REGION>': 'ocr',
            '<CAPTION>': 'pure_text', '<DETAILED_CAPTION>': 'pure_text', '<MORE_DETAILED_CAPTION>': 'pure_text',
            '<OD>': 'description_with_bboxes', '<DENSE_REGION_CAPTION>': 'description_with_bboxes',
            '<CAPTION_TO_PHRASE_GROUNDING>': 'phrase_grounding',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'polygons', '<REGION_TO_SEGMENTATION>': 'polygons',
            '<OPEN_VOCABULARY_DETECTION>': 'description_with_bboxes_or_polygons',
            '<REGION_TO_CATEGORY>': 'pure_text', '<REGION_TO_DESCRIPTION>': 'pure_text', '<REGION_TO_OCR>': 'pure_text',
            '<REGION_PROPOSAL>': 'bboxes',
        }
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.',
        }
        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }
        self.post_processor = PostProcessor(tokenizer=self.tokenizer)

    def _construct_prompts(self, text):
        for task_token, task_prompt in self.task_prompts_without_inputs.items():
            if task_token in text:
                return task_prompt
        for task_token, task_prompt in self.task_prompts_with_input.items():
            if task_token in text:
                return task_prompt.format(input=text.replace(task_token, ''))
        return text

    def __call__(self, text, images):
        prompt = self._construct_prompts(text)
        encoded = self.tokenizer.encode(prompt)
        pixel_values = preprocess_image(images)
        return {'input_ids': encoded['input_ids'], 'pixel_values': pixel_values}

    def batch_decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def post_process_generation(self, text, task, image_size):
        pp_type = self.tasks_answer_post_processing_type.get(task, 'pure_text')
        result = self.post_processor(text=text, image_size=image_size, parse_tasks=pp_type)[pp_type]

        if pp_type == 'pure_text':
            final = result.replace('<s>', '').replace('</s>', '')
        elif pp_type in ('od', 'description_with_bboxes', 'bboxes'):
            final = {'bboxes': [i['bbox'] for i in result], 'labels': [str(i['cat_name']) for i in result]}
        elif pp_type == 'ocr':
            final = {'quad_boxes': [i['quad_box'] for i in result], 'labels': [str(i['text']) for i in result]}
        elif pp_type == 'phrase_grounding':
            bboxes, labels = [], []
            for phrase in result:
                for bbox in phrase['bbox']:
                    bboxes.append(bbox)
                    labels.append(phrase['cat_name'])
            final = {'bboxes': bboxes, 'labels': labels}
        elif pp_type in ('description_with_polygons', 'polygons'):
            final = {'polygons': [r['polygons'] for r in result], 'labels': [r['cat_name'] for r in result]}
        elif pp_type == 'description_with_bboxes_or_polygons':
            bboxes, bl, polygons, pl = [], [], [], []
            for r in result:
                if 'polygons' in r:
                    polygons.append(r['polygons'])
                    pl.append(r['cat_name'])
                else:
                    bboxes.append(r['bbox'])
                    bl.append(r['cat_name'])
            final = {'bboxes': bboxes, 'bboxes_labels': bl, 'polygons': polygons, 'polygons_labels': pl}
        else:
            raise ValueError(f'Unknown post processing type: {pp_type}')
        return {task: final}
