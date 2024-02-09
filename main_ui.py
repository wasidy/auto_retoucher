# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:57:03 2024

@author: Pussy
"""
import gradio as gr
import json
from types import SimpleNamespace
from ui.panels import auto_retoucher_interface
from scripts.utils import get_checkpoints_list
from scripts.imageutils import get_custom_tile, image_rotate, image_ss_resize
from scripts.autoretoucher import AutoRetoucher
from scripts.pipelines import Mask2FormerSegmentation, SDXLPipeLine


def create_ui():
    mask_predict = Mask2FormerSegmentation(config.mask2former_model, config.mask2former_processor)
    # sdxl_pipe = SDXLPipeLine(f'{config.path_to_checkpoints}/{get_checkpoints_list(config.path_to_checkpoints)[0]}')
    sdxl_pipe = SDXLPipeLine()
    retoucher = AutoRetoucher(config, mask_predict, sdxl_pipe)

    with gr.Blocks() as demo:
        with gr.Row():
            auto_retoucher_interface(config,
                                     fill_mask=retoucher.fill_mask,
                                     invert_mask=retoucher.invert_mask,
                                     input_image_change=retoucher.generate_mask_and_grid,
                                     generate_mask=retoucher.generate_mask,
                                     remesh_grid=retoucher.remesh_grid,
                                     generate_image=retoucher.generate_image,
                                     image_rotate=image_rotate,
                                     resize_image=image_ss_resize,
                                     stop_generation=retoucher.stop_generation,
                                     load_checkpoint=retoucher.load_checkpoint,
                                     select_folder_for_batch=retoucher.select_folder_for_batch,
                                     generate_batch_files=retoucher.generate_batch_files,
                                     get_custom_tile=get_custom_tile,
                                     apply_custom_tile=retoucher.apply_custom_tile,
                                     generate_custom_tile=retoucher.generate_custom_tile,
                                     reset_all=retoucher.reset_all,
                                     save_output_image=retoucher.save_output_image,
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

    ui = create_ui()
    main_ui(ui)
