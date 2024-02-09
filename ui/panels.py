# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:37 2024

@author: Wasidy
"""
import re
import os
import gradio as gr
from typing import List

from scripts.utils import get_checkpoints_list


def auto_retoucher_interface(config, **kwargs) -> List[gr.components.Component]:

    # %% Left panel
    with gr.Column():
        ''' Show image, mask, and preprocess image for generation '''
        with gr.Tabs('Preprocess'):
            with gr.TabItem('Input image', id=0):
                input_image = gr.Image(
                    sources='upload',
                    show_download_button=False,
                    container=False,
                    label='Input image'
                    )
            with gr.TabItem('Composite mask', id=1):
                composite_mask = gr.Image(
                    sources='upload',
                    label='Composite mask',
                    show_download_button=False,
                    container=False,
                    interactive=False
                    )
            with gr.TabItem('Mask'):
                mask_preview = gr.Image(
                    visible=True,
                    sources='upload',
                    label='Mask',
                    show_download_button=False,
                    container=False,
                    interactive=False
                    )
            with gr.TabItem('Grid', id=2):
                grid_image = gr.Image(
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
                            choices=config.file_formats,
                            value=config.default_file_format,
                            show_label=False,
                            interactive=True
                            )
                        resize_batch_images = gr.Checkbox(
                            label='Resize batch',
                            value=False)
                    with gr.Column():
                        select_folder_gr = gr.Button(
                            'Select outputs folder'
                            )

                    cwd = re.sub('\\\\', '/', os.getcwd() + '/' + config.path_to_outputs + '/')
                    select_folder_show_gr = gr.Text(
                        label='Outputs folder:',
                        value=cwd,
                        interactive=False,
                        show_label=True
                        )

        # %% GENERATION CONTROL
        with gr.Row():
            with gr.Column():
                generate_button = gr.Button(
                    'GENERATE',
                    interactive=True,
                    )
                stop_button_bt = gr.Button(
                    'STOP',
                    interactive=True,
                    )
            with gr.Column():
                reset_all_bt = gr.Button(
                    'RESET ALL PARAMETERS',
                    )
        with gr.Row():
            with gr.Column():
                base_prompt = gr.Textbox(
                    label='Prompt',
                    lines=2,
                    interactive=True,
                    value=config.prompt
                    )
            with gr.Column():
                base_negative_prompt = gr.Textbox(
                    label='Negative prompt',
                    lines=2,
                    interactive=True,
                    value=config.negative_prompt
                    )

        # %% IMAGE AND MASK PROCESSING
        with gr.Accordion(label='Image and mask processing', open=True):
            with gr.Row():
                rotate_left_button = gr.Button('◀ Rotate left')
                rotate_left_value = gr.State(value=270)
                rotate_right_button = gr.Button('Rotate Right ▶')
                rotate_right_value = gr.State(value=90)
            with gr.Row():
                resize_image_button = gr.Button(
                    'Resize Image'
                    )
                copy_input_to_output = gr.Button(
                    'Copy input to output'
                    )
            with gr.Row():
                with gr.Column():
                    resize_image_size = gr.Slider(
                        minimum=config.image_minimum_size,
                        maximum=config.image_maximum_size,
                        value=config.image_resize_size,
                        step=64, label='Shortest side',
                        show_label=True,
                        interactive=True
                        )
                    mask_blur_value = gr.Slider(
                        minimum=0,
                        maximum=256,
                        value=config.figure_mask_blur,
                        step=1,
                        label='Mask blur',
                        show_label=True,
                        interactive=True
                        )
                    mask_expand_value = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=config.figure_mask_expand,
                        label='Mask expand',
                        show_label=True,
                        interactive=True
                        )
                with gr.Column():
                    current_image_size = gr.Text(
                        show_label=False,
                        interactive=False
                        )
            
                    label_id = gr.Dropdown(
                        choices=list(x for x in range(100)),
                        value=0,
                        show_label=False,
                        interactive=True
                        )
            
                    
            with gr.Row():
                with gr.Column():
                    fill_mask_button = gr.Button(
                        'Fill mask',
                        )
                with gr.Column():
                    make_mask_bt = gr.Button(
                        'Remake figure mask',
                        )
            with gr.Row():
                with gr.Column():
                    minimum_overlap = gr.Slider(
                        minimum=32,
                        maximum=256,
                        value=config.tile_default_minimum_overlap,
                        step=8,
                        label='Minimum overlap',
                        show_label=True,
                        interactive=True
                        )
                    minimum_density = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=config.tile_minimum_density,
                        step=1,
                        label='Minumum density',
                        show_label=True,
                        interactive=True
                        )
                with gr.Column():
                    tile_size = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=config.default_tile_size,
                        step=128,
                        label='Tile size',
                        show_label=True,
                        interactive=True
                        )
                    remesh_grid_bt = gr.Button(
                        'Remesh grid'
                        )

        # %% GENERATION SETTINGS
        with gr.Accordion(label='Generation settings', open=True):
            with gr.Row():
                with gr.Column():
                    strength = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=config.default_strength,
                        label='Denoise strength', show_label=True,
                        interactive=True
                        )
                    steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=config.default_steps,
                        step=1,
                        label='Steps',
                        show_label=True,
                        interactive=True
                        )
                with gr.Column():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=config.default_batch_size,
                        step=1,
                        label='Batch size',
                        show_label=True,
                        interactive=True
                        )
            with gr.Row():
                checkpoints_list = get_checkpoints_list(config.path_to_checkpoints)
                checkpoints_dropdown = gr.Dropdown(
                    choices=checkpoints_list,
                    value=checkpoints_list[0],
                    label='SDXL checkpoint',
                    interactive=True
                    )

    # %%  Right panel
    with gr.Column():
        with gr.Tabs('Output'):
            with gr.TabItem('Output image'):
                out = gr.Image(
                    show_label=False,
                    interactive=False,
                    show_download_button=True
                    )

        # %% CUSTOM TILE GENERATION
        with gr.Accordion('ReGEN custom tile', open=True):
            with gr.Tabs():
                with gr.TabItem('Before', id='custom_tile') as custom_tile:
                    custom_tile = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False
                        )
                    custom_tile_coords = gr.State()
                with gr.TabItem('After'):
                    tile_after_gr = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False
                        )
            with gr.Row():
                with gr.Column():
                    apply_mask_for_tile_gr = gr.Checkbox(
                        label='Apply generated mask',
                        value=True
                        )
                with gr.Column():
                    custom_tmask_value_gr = gr.Slider(
                        label='Border tile masking',
                        value=config.custom_tile_border_mask_value,
                        minimum=0,
                        maximum=256,
                        show_label=True,
                        interactive=True
                        )
            with gr.Row():
                with gr.Column():
                    regen_custom_tile_bt = gr.Button(
                        'GENERATE TILE'
                        )
                    tile_coordinates_gr = gr.State()
                with gr.Column():
                    apply_custom_tile_bt = gr.Button(
                        'APPLY TO IMAGE'
                        )
                with gr.Row():
                    with gr.Column():
                        ct_prompt = gr.Textbox(
                            label='Prompt',
                            lines=2,
                            value=config.custom_tile_prompt
                            )
                    with gr.Column():
                        ct_negative_prompt = gr.Textbox(
                            label='Negative prompt',
                            lines=2,
                            value=config.custom_tile_negative_prompt
                            )

        # %% CLIP MASK PROCESSING
        with gr.Accordion('CLIP mask processing'):
            auto_masked_image_gr = gr.Image()
            with gr.Row():
                with gr.Column():
                    auto_process_clip_masking_gr = gr.Checkbox(
                        label='Automatic apply after generate',
                        value=config.clip_tile_auto_apply
                        )
                    scale_to_tile_gr = gr.Checkbox(
                        label='Scale to tile size',
                        value=config.clip_tile_scale_to_tile_size
                        )
                    different_prompts_gr = gr.Checkbox(
                        label='Different prompts for objects',
                        value=config.clip_tile_different_prompts_for_multiple_objects
                        )
                with gr.Column():
                    select_object_gr = gr.Dropdown(
                        choices=[],
                        label='Select object',
                        interactive=True
                        )
                    preview_image_clipmask_gr = gr.Button(
                        'Generate masks'
                        )
            keyword_for_mask_gr = gr.Textbox(
                label='Keyword for automatic mask',
                value=config.keyword_for_clip_masking,
                lines=1,
                interactive=True
                )
            with gr.Row():
                with gr.Column():
                    automask_prompt = gr.Textbox(
                        label='Prompt',
                        lines=2,
                        value=config.clip_tile_prompt
                        )
                with gr.Column():
                    automask_negative_prompt = gr.Textbox(
                        label='Negative prompt',
                        lines=2,
                        value=config.clip_tile_negative_prompt
                        )

        # %% SAVING OPTIONS
        with gr.Accordion('Saving options'):
            with gr.Row():
                with gr.Column():
                    save_file_after_generation = gr.Checkbox(
                        label='Auto save outputs',
                        value=config.auto_save_outputs,
                        interactive=True
                        )
                    save_file_format = gr.Radio(
                        choices=config.file_formats,
                        value=config.default_file_format,
                        show_label=False,
                        interactive=True
                        )
                with gr.Column():
                    add_model_name_to_filename = gr.Checkbox(
                        label='Add model name',
                        value=config.add_checkpoint_name_to_file,
                        interactive=True
                        )
                    save_current_out_gr = gr.Button(
                        'SAVE CURRENT OUTPUT'
                        )

    # %% Events
    input_image.change(fn=kwargs.get('input_image_change'),
                       inputs=[input_image,
                               minimum_overlap,
                               tile_size,
                               minimum_density,
                               mask_blur_value,
                               mask_expand_value],
                       outputs=[composite_mask,
                                mask_preview,
                                grid_image,
                                current_image_size],
                       concurrency_id='fn'
                       )

    copy_input_to_output.click(fn=lambda x: x, inputs=[input_image], outputs=[out])

    rotate_left_button.click(fn=kwargs.get('image_rotate'),
                             inputs=[input_image, rotate_left_value],
                             outputs=[input_image])

    rotate_right_button.click(fn=kwargs.get('image_rotate'),
                              inputs=[input_image, rotate_right_value],
                              outputs=[input_image])

    resize_image_button.click(fn=kwargs.get('resize_image'),
                              inputs=[input_image, resize_image_size],
                              outputs=[input_image])
    
    fill_mask_button.click(fn=kwargs.get('fill_mask'),
                           outputs=[mask_preview])

    generate_button.click(fn=kwargs.get('generate_image'),
                          inputs=[input_image,
                                  strength,
                                  base_prompt,
                                  base_negative_prompt,
                                  steps,
                                  batch_size,
                                  save_file_after_generation,
                                  save_file_format,
                                  add_model_name_to_filename
                                  ],
                          outputs=[out],
                          concurrency_id='fn'
                          )

    def get_custom_tile(ev: gr.SelectData, image, size):
        ''' Get custom tile from input or output '''
        fn = kwargs.get('get_custom_tile')
        tile, x1, y1, x2, y2 = fn(image, size, (ev.index[0], ev.index[1]))
        return tile, (x1, y1, x2, y2)

    input_image.select(fn=get_custom_tile,
                       inputs=[input_image, tile_size],
                       outputs=[custom_tile, custom_tile_coords]
                       )
    return
