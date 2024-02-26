import sys
import gradio as gr
import json
from types import SimpleNamespace
from ui.panels import auto_retoucher_interface
from ui.settings import RetoucherSettings, Settings
from scripts.utils import get_checkpoints_list
from scripts.imageutils import get_custom_tile, image_rotate, image_ss_resize
from scripts.autoretoucher import AutoRetoucher
from scripts.pipelines import Mask2FormerSegmentation, SDXLPipeLine, ClipSegmentation
from scripts.remote_pipeline import RemoteSDXLPipeLine
from scripts.face_detector import FaceDetector

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

def create_ui():
    mask_predict = Mask2FormerSegmentation(
        config.mask2former_model,
        config.mask2former_processor
        )

    # clip_predict = ClipSegmentation(
    #     config.clip_segmentation_model,
    #     config.clip_segmentation_processor
    #     )

#    mask_predict = Mask2FormerSegmentation(
#        'facebook/mask2former-swin-base-IN21k-ade-semantic',
#        'facebook/mask2former-swin-base-IN21k-ade-semantic'
#        )

    clip_predict = ClipSegmentation(
        config.clip_segmentation_model,
        config.clip_segmentation_processor
        )

    face_detector = FaceDetector(clip_predict)
    # sdxl_pipe = SDXLPipeLine(f'{config.path_to_checkpoints}/{get_checkpoints_list(config.path_to_checkpoints)[0]}')
    sdxl_pipe = RemoteSDXLPipeLine()  # For debuggin purposes only
    
    
    retoucher = AutoRetoucher(config, mask_predict, clip_predict, sdxl_pipe, face_detector)
    
    with gr.Blocks() as demo:
        with gr.Row():
            auto_retoucher_interface(config,
                                     invert_mask=retoucher.invert_mask,
                                     input_image_load=retoucher.load_image,
                                     generate_mask=retoucher.generate_mask,
                                     remesh_grid=retoucher.generate_grid,
                                     generate_image=retoucher.generate_image,
                                     image_rotate=image_rotate,
                                     resize_image=retoucher.resize_image,
                                     stop_generation=retoucher.stop_generation,
                                     load_checkpoint=retoucher.load_checkpoint,
                                     batch_generate=retoucher.batch_generation,
                                     get_custom_tile_and_mask=retoucher.get_custom_tile_and_mask,
                                     apply_custom_tile=retoucher.apply_custom_tile,
                                     generate_custom_tile=retoucher.generate_custom_tile,
                                     reset_all=retoucher.reset_all,
                                     save_output_image=retoucher.save_output_image,
                                     generate_clip_mask=retoucher.generate_clip_mask,
                                     test_button=retoucher.test_cfg
                                     )

    return demo


def main_ui(ui):
    ui.launch()


if __name__ == '__main__':
    with open('config.json') as cfg:
        try:
            config_data = cfg.read()
            config = json.loads(config_data, object_hook=lambda d: SimpleNamespace(**d))
        except Exception:
            print('Can not read config.json file')
            raise SystemExit
    sys.stdout = Logger("output.log")
    settings = Settings()
    ui = create_ui()
    
    main_ui(ui)
