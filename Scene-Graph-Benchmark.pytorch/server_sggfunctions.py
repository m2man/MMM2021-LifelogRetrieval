from maskrcnn_benchmark.utils.env import setup_environment
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

from server_myfunctions import *

#torch.cuda.set_device(1)
cpu_device = torch.device('cpu')
device = torch.device('cuda')
num_gpus = 1
distributed = num_gpus > 1

config_file = 'configs/e2e_relation_X_101_32_8_FPN_1x.yaml'
cfg.merge_from_file(config_file)
cfg.local_rank = 0
# cfg.merge_from_list(args.opts)
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = 'CausalAnalysisPredictor'
cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = 'none'
cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = 'sum'
cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = 'motifs'
cfg.TEST.IMS_PER_BATCH = 1
cfg.DTYPE = "float16"
cfg.GLOVE_DIR= '/mnt/DATA/nmduy/glove' # GLOVE FILE
cfg.MODEL.PRETRAINED_DETECTOR_CKPT = '/mnt/DATA/nmduy/pretrained_faster_rcnn/pretrained_faster_rcnn/model_final.pth' # PRETRAIN FASTER RCNN
cfg.MODEL.DEVICE = 'cuda'
cfg.OUTPUT_DIR = '/home/nmduy/Scene-Graph-Benchmark.pytorch/pretrained_causal_motif_sgdet' # PRETRAIN SGG

cfg.freeze()

model = build_detection_model(cfg)
model.to(cfg.MODEL.DEVICE);

my_transform = my_build_transforms(cfg, is_train=False)

# Initialize mixed-precision if necessary
use_mixed_precision = cfg.DTYPE == 'float16'
amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

# Load model
output_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)
model.eval();

def perform_sgg_on_image(sample_img, thres_obj=0.15, thres_rel=5e-5):
    original_size = sample_img.size
    # Transform to tensor format
    sample_img_transform = my_transform(sample_img)
    # Convert to ImageList Format
    sample_img_transform_iml = to_image_list(sample_img_transform, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)

    # Run the generation
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model(sample_img_transform_iml.to(device))
        output = output[0]

    # resize to the original size and filter out low result
    prediction = output.resize(original_size)

    # Filter out low result
    refine_filter = refine_boxlist(prediction, thres_obj=thres_obj, thres_rel=thres_rel)
    refine = ranking_boxlist(refine_filter)
    pred_scores = refine.get_field('pred_scores')
    pred_labels = refine.get_field('pred_labels')
    
    # Decode to human read
    result_json = {}
    decode = decode_relation(refine, show_scores=True)
    human_read = translate_to_human_read(decode)
    result_json['sgg'] = decode
    result_json['bbox'] = {'bbox':refine.bbox.cpu().numpy().tolist(), 
                           'labels': pred_labels.tolist(), 
                           'scores': pred_scores.tolist()}
    #result_json['bbox_labels'] = pred_labels.tolist()
    #result_json['bbox_scores'] = pred_scores.tolist()
    result_json['human_read'] = human_read

    return result_json