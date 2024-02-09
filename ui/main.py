# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:57:03 2024

@author: Pussy
"""
import re
import os

import gradio as gr

OUTPUT_PATH = 'fucj you'

def image_mask_grid_processing():
    
    return

def generation_settings():
    return


def input_images_panel():
    with gr.Tabs('Preprocess') as input_tabs:
        with gr.TabItem('Input image', id=0):
            input_image_gr = gr.Image(
                sources='upload',
                show_download_button=False,
                container=False,
                label='Input image'
                )
        with gr.TabItem('Composite mask', id=1):
            mask_composite_gr = gr.Image(
                sources='upload',
                label='Composite mask',
                show_download_button=False,
                container=False,
                interactive=False
                )
        with gr.TabItem('Mask'):
            mask_gr = gr.Image(
                visible=True,
                sources='upload',
                label='Mask',
                show_download_button=False,
                container=False,
                interactive=False
                )
        with gr.TabItem('Grid', id=2):
            grid_image_gr = gr.Image(
                sources='upload', label='Grid',
                show_download_button=False, container=False,
                show_label=False, interactive=False
                )
        with gr.TabItem('Batch processing'):
            batch_files_gr = gr.File(
                file_count='multiple',
                file_types=['image']
                )
            batch_files_bt = gr.Button(
                'Batch processing with current setting',
                elem_id='generate'
                )
            with gr.Row():
                with gr.Column():
                    out_file_format_gr = gr.Radio(
                        choices=['JPG', 'PNG'],
                        value='PNG',
                        show_label=False,
                        interactive=True
                        )
                with gr.Column():
                    select_folder_gr = gr.Button(
                        'Select outputs folder'
                        )
                    cwd = re.sub('\\\\', '/', os.getcwd() + '/' + OUTPUT_PATH + '/')
                select_folder_show_gr = gr.Text(
                    label='Outputs folder:',
                    value=cwd,
                    interactive=False,
                    show_label=True
                    )
    
    gradio_components: List[gr.components.Component] = [
        input_image_gr,
        mask_composite_gr,
        mask_gr,
        grid_image_gr,
        batch_files_gr,
        batch_files_bt,
        out_file_format_gr,
        select_folder_gr,
        select_folder_show_gr,
        ]
    
    return gradio_components

def create_ui():
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_images_panel()
            with gr.Column():
                pass
    return demo

def main_ui(ui):
    ui.launch()
    
    
        
if __name__ == '__main__':
    ui = create_ui()
    main_ui(ui)