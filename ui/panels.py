# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:37 2024

@author: Wasidy
"""
import re
import os
import sys
import gradio as gr
from typing import List
from tkinter import Tk, filedialog
import numpy as np
from PIL import Image
from scripts.utils import get_checkpoints_list

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        logs = f.readlines()
        lines = logs if len(logs) <= 10 else logs[-10:]
        out = ''.join(line for line in lines)
        return out

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
                composite_mask_preview = gr.Image(
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
                grid_image_preview = gr.Image(
                    sources='upload', label='Grid',
                    show_download_button=False, container=False,
                    show_label=False, interactive=False
                    )
            with gr.TabItem('Batch processing'):
                batch_file_list = gr.File(
                    file_count='multiple',
                    file_types=['image']
                    )
                process_batch_files_button = gr.Button(
                    'Batch processing with current setting',
                    elem_id='generate'
                    )
                with gr.Row():
                    with gr.Column():
                        batch_output_file_format = gr.Radio(
                            choices=config.file_formats,
                            value=config.default_file_format,
                            show_label=False,
                            interactive=True
                            )
                        resize_batch_images = gr.Checkbox(
                            label='Resize batch',
                            value=False)
                    with gr.Column():
                        mask_batch_mode = gr.Radio(
                            choices = ['Mask2Former', 'Fill'],
                            value = 'Mask2Former',
                            label = 'Mask type in batch',
                            show_label=True,
                            interactive=True
                            )
                        fill_mask_if_not_detected = gr.Checkbox(
                            label='Fill mask if object not detected',
                            show_label=True,
                            value=True)
                        select_folder_button = gr.Button(
                            'Select outputs folder'
                            )

                    cwd = re.sub('\\\\', '/', os.getcwd() + '/' + config.path_to_outputs + '/')
                    output_folder = gr.Text(
                        label='Output folder:',
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
                stop_button = gr.Button(
                    'STOP',
                    interactive=True,
                    )
            with gr.Column():
                reset_all_button = gr.Button(
                    'RESET ALL PARAMETERS',
                    )
        with gr.Row():
            with gr.Column():
                base_prompt = gr.Textbox(
                    label='Prompt',
                    lines=4,
                    interactive=True,
                    value=config.prompt
                    )
            with gr.Column():
                base_negative_prompt = gr.Textbox(
                    label='Negative prompt',
                    lines=4,
                    interactive=True,
                    value=config.negative_prompt
                    )

        # %% IMAGE AND MASK PROCESSING
        with gr.Accordion(label='Image and mask processing', open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion('Image size and orientation'):
                        rotate_left_button = gr.Button('◀ Rotate left')
                        rotate_left_value = gr.State(value=270)
                        rotate_right_button = gr.Button('Rotate Right ▶')
                        rotate_right_value = gr.State(value=90)
                        
                        resize_image_size = gr.Slider(
                            minimum=config.image_minimum_size,
                            maximum=config.image_maximum_size,
                            value=config.image_resize_size,
                            step=64, label='Shortest side',
                            show_label=True,
                            interactive=True
                            )
                        resize_image_at_load = gr.Checkbox(
                            label='Auto resize image at load',
                            value=False)
                        current_image_size_value = gr.Text(
                                label='Current image size (h:w)',
                                show_label=True,
                                interactive=False
                                )
                        resize_image_button = gr.Button('Resize Image')
                with gr.Column():
                    with gr.Accordion('Grid settings'):
                        minimum_overlap_value = gr.Slider(
                            minimum=32,
                            maximum=256,
                            value=config.tile_default_minimum_overlap_value,
                            step=8,
                            label='Minimum overlap',
                            show_label=True,
                            interactive=True
                            )
                        minimum_density_value = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=config.tile_minimum_density_value,
                            step=1,
                            label='Minumum density',
                            show_label=True,
                            interactive=True
                            )
                        tile_size = gr.Slider(
                            minimum=512,
                            maximum=1536,
                            value=config.default_tile_size,
                            step=128,
                            label='Tile size',
                            show_label=True,
                            interactive=True
                            )
                        remesh_grid_button = gr.Button(
                            'REMESH GRID'
                            )
            with gr.Row():
                with gr.Column():
                    with gr.Accordion('Skintone mask'):
                        skin_mask_checkbox = gr.Checkbox(
                            label='Skin mask only',
                            value=True,
                            interactive=True
                            )
                        standart_skin_tones = gr.Checkbox(
                            label='Use standart skintones if face not found',
                            value=True,
                            interactive=True
                            )
                        
                        skin_mask_hue_threshold = gr.Slider(
                            label='HUE threshold',
                            value=2,
                            minimum=0,
                            maximum=100,
                            show_label=True,
                            interactive=True
                            )
                        skin_mask_sat_threshold = gr.Slider(
                            label='Saturation threshold',
                            value=2,
                            minimum=0,
                            maximum=100,
                            show_label=True,
                            interactive=True
                            )
                        skin_mask_val_threshold = gr.Slider(
                            label='Value threshold',
                            value=2,
                            minimum=0,
                            maximum=100,
                            show_label=True,
                            interactive=True
                            )
                with gr.Column():
                    with gr.Accordion('Mask settings'):
                        mask_mode = gr.Dropdown(
                            label='Mask mode',
                            show_label=True,
                            choices=['Mask2Former', 'Fill', 'Faces'],
                            value='Mask2Former',
                            interactive=True)
                        label_id = gr.Dropdown(
                            label='Mask2Former Label ID',
                            choices=list(x for x in range(100)),
                            value=0,
                            show_label=True,
                            interactive=True
                            )
                        mask_smooth_value = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=config.mask_smooth,
                            label='Mask smooth',
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
                        mask_blur_value = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=config.figure_mask_blur,
                            step=1,
                            label='Mask blur',
                            show_label=True,
                            interactive=True
                            )
                        face_mask_threshold = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=config.face_mask_threshold,
                            step=1,
                            label='Face mask threshold',
                            show_label=True,
                            interactive=True)
            with gr.Row():
                generate_mask_button = gr.Button('GENERATE MASK')
                    
                    
                    
                        
                
            

                    

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
                        label='Batch_size',
                        show_label=True,
                        interactive=True,
                        info='batchSize',
                        
                        )
            with gr.Row():
                checkpoints_list = get_checkpoints_list(config.path_to_checkpoints)
                checkpoints_dropdown = gr.Dropdown(
                    choices=checkpoints_list,
                    value=checkpoints_list[0],
                    label='SDXL checkpoint',
                    interactive=True
                    )
                
        with gr.Accordion('Face postprocessing'):
            with gr.Row():
                with gr.Column():
                    
                    postprocess_face_checkbox = gr.Checkbox(
                        label='Auto generate face',
                        value=True,
                        interactive=True,
                        )
                    postprocess_all_faces_checkbox = gr.Checkbox(
                        label='Generate all faces',
                        value=False,
                        interactive=True,
                        )
                with gr.Column():
                    upscale_face_to_tile_size = gr.Checkbox(
                        label='Upscale face to tile size',
                        value=False,
                        interactive=True,
                        )
                    downscale_face_to_tile_size = gr.Checkbox(
                        label='Downscale face to tile size',
                        value=True,
                        interactive=True)
            with gr.Row():
                with gr.Column():
                    use_face_mask = gr.Checkbox(
                        label='Paste with face mask',
                        value=True,
                        interactive=True)
                with gr.Column():
                    face_tile_masking = gr.Slider(
                        label='Mask smoothing',
                        minimum=1,
                        maximum=128,
                        interactive=True)
            with gr.Row():
                face_prompt = gr.Textbox(
                    label='Positive face prompt',
                    lines=2)
                face_negative_prompt = gr.Textbox(
                    label='Negative face prompt',
                    lines=2)
                

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
                with gr.TabItem('Image', id='custom_tile'):
                    custom_tile_source = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False
                        )
                    custom_tile_coordinates = gr.State()
                with gr.TabItem('Composite'):
                    custom_tile_composite = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False
                        )
                with gr.TabItem('Mask'):
                    custom_tile_mask = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False,
                        image_mode='L'
                        )
                with gr.TabItem('Editor'):
                    
                    custom_tile_mask_editor = gr.ImageEditor(
                        #brush = gr.Brush(colors=['#FFFFFF'], color_mode='fixed'),
                        image_mode='L',
                        #transforms=()
                        )
                    load_editor_background = gr.Button('Load background')
                    editor_apply_mask = gr.Button('Apply')
                    
                    def load_editor_background_fn(image, mask, editor):
                        #d = dict(background=image)
                        editor['background'] = np.zeros((image.shape[0],image.shape[1],3))
                        mask = mask[:, :, np.newaxis]
                        comp_image = np.concatenate((image, mask), axis=2)
                        editor['layers'][0] = comp_image
                        print(len(editor['layers']))
                        #layers = []
                        #layers.append(mask)
                        #mask = mask[:, :, np.newaxis]
                        #mask = mask.repeat(4, axis=2)
                        
                        #mask = Image.fromarray(mask, mode='L')
                        #print(mask)
                        #print(f'Mask shape: {mask.shape}')
                        #editor['layers'][0]=mask
                        #editor['composite'] = mask
                        #print(type(editor))
                        return editor
                    
                    def delete_layers(editor):
                        print(editor['layers'])
                        return 
                        
                    
                    load_editor_background.click(
                        fn=load_editor_background_fn,
                        inputs=[custom_tile_source,
                                custom_tile_mask,
                                custom_tile_mask_editor],
                        outputs=[custom_tile_mask_editor]) 
                    '''.then(
                            fn=delete_layers,
                            inputs=[custom_tile_mask_editor],
                            outputs=[custom_tile_mask_editor])'''
   
                    
                    
                    def editor_apply_mask_fn(editor):
                        return editor['layers'][0]
                    
                    editor_apply_mask.click(fn=delete_layers,
                                            inputs=[custom_tile_mask_editor],
                                            )
                    
                    # editor_apply_mask.click(fn=editor_apply_mask_fn,
                    #                         inputs=[custom_tile_mask_editor],
                    #                         outputs=[custom_tile_mask])
                    
                with gr.TabItem('Generation'):
                    custom_tile_generated = gr.Image(
                        show_label=False,
                        sources=None,
                        interactive=False
                        )
            with gr.Row():
                with gr.Column():
                    apply_base_mask_for_custom_tile = gr.Checkbox(
                        label='Apply generated mask',
                        value=True
                        )
                with gr.Column():
                    custom_tile_border_mask_value = gr.Slider(
                        label='Border tile masking',
                        value=config.custom_tile_border_mask_value,
                        minimum=0,
                        maximum=256,
                        show_label=True,
                        interactive=True
                        )
            with gr.Row():
                with gr.Column():
                    generate_custom_tile_button = gr.Button(
                        'GENERATE TILE'
                        )
                    custom_tile_coordinates = gr.State()
                with gr.Column():
                    apply_custom_tile_button = gr.Button(
                        'APPLY TO IMAGE'
                        )
                    copy_input_to_output_button = gr.Button(
                        'Copy input to output'
                        )
                with gr.Row():
                    with gr.Column():
                        custom_tile_prompt = gr.Textbox(
                            label='Prompt',
                            lines=2,
                            value=config.custom_tile_prompt
                            )
                    with gr.Column():
                        custom_tile_negative_prompt = gr.Textbox(
                            label='Negative prompt',
                            lines=2,
                            value=config.custom_tile_negative_prompt
                            )

        # %% CLIP MASK PROCESSING
        # with gr.Accordion('CLIP mask processing'):
        #     with gr.Tabs():
        #         with gr.TabItem('CLIP Tile'):
        #             clip_mask_tile_source = gr.Image()
                    
        #         with gr.TabItem('Mask'):
        #             clip_mask_preview = gr.Image(image_mode='L')
                
        #         with gr.TabItem('Result'):
        #             clip_tile_preview = gr.Image()
                    
        #     with gr.Row():
        #         with gr.Column():
        #             auto_process_clip_masking_gr = gr.Checkbox(
        #                 label='Automatic apply after generate',
        #                 value=config.clip_tile_auto_apply
        #                 )
        #             scale_clip_mask_to_tile = gr.Checkbox(
        #                 label='Scale to tile size',
        #                 value=config.clip_tile_scale_to_tile_size
        #                 )
        #             different_prompts_gr = gr.Checkbox(
        #                 label='Different prompts for objects',
        #                 value=config.clip_tile_different_prompts_for_multiple_objects
        #                 )
        #             clip_tile_coordinates = gr.State()
        #         with gr.Column():
        #             select_object_gr = gr.Dropdown(
        #                 choices=[],
        #                 label='Select object',
        #                 interactive=True
        #                 )
        #             generate_clip_mask_button = gr.Button(
        #                 'Generate masks'
        #                 )
        #             generate_clip_tile_button = gr.Button(
        #                 'Generate tile'
        #                 )
        #     keyword_for_clip_mask = gr.Textbox(
        #         label='Keyword for automatic mask',
        #         value=config.keyword_for_clip_masking,
        #         lines=1,
        #         interactive=True
        #         )
        #     with gr.Row():
        #         with gr.Column():
        #             clip_mask_prompt = gr.Textbox(
        #                 label='Prompt',
        #                 lines=2,
        #                 value=config.clip_tile_prompt
        #                 )
        #         with gr.Column():
        #             clip_mask_negative_prompt = gr.Textbox(
        #                 label='Negative prompt',
        #                 lines=2,
        #                 value=config.clip_tile_negative_prompt
        #                 )

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
                    test_button = gr.Button('For debug')
        logs = gr.Textbox(value=read_logs, every=1, lines=10, label="Logs", autoscroll=True)
        
    # %% Events
    # input_image.upload(fn=kwargs.get('input_image_change'),
    #                    inputs=[input_image,
    #                            tile_size,
    #                            minimum_overlap_value,
    #                            minimum_density_value,
    #                            mask_blur_value,
    #                            mask_expand_value,
    #                            label_id,
    #                            mask_mode,
    #                            skin_mask_checkbox,
    #                            resize_image_size,
    #                            resize_image_at_load,
    #                            ],
    #                    outputs=[input_image,
    #                             composite_mask_preview,
    #                             mask_preview,
    #                             grid_image_preview,
    #                             current_image_size_value],
    #                    concurrency_id='fn'
    #                    )

    input_image.upload(fn=kwargs.get('input_image_load'),
                       inputs=[input_image,
                               resize_image_at_load,
                               resize_image_size],
                       outputs=[input_image,
                                current_image_size_value],
                       concurrency_id='fn'
                       )

    # generate_mask_button.click(fn=kwargs.get('generate_mask'),
    #                            inputs=[input_image,
    #                                    tile_size,
    #                                    minimum_overlap_value,
    #                                    minimum_density_value,
    #                                    mask_blur_value,
    #                                    mask_expand_value,
    #                                    label_id,
    #                                    mask_mode,
    #                                    skin_mask_checkbox,
    #                                    resize_image_size,
    #                                    resize_image_at_load,
    #                                    skin_mask_hue_threshold,
    #                                    skin_mask_sat_threshold,
    #                                    skin_mask_val_threshold
    #                                    ],
    #                            outputs=[input_image,
    #                                     composite_mask_preview,
    #                                     mask_preview,
    #                                     grid_image_preview,
    #                                     current_image_size_value],
    #                            concurrency_id='fn'
    #                            )

    generate_mask_button.click(fn=kwargs.get('generate_mask'),
                               inputs=[
                                       mask_mode,
                                       label_id,
                                       mask_smooth_value,
                                       mask_expand_value,
                                       mask_blur_value,
                                       face_mask_threshold,
                                       skin_mask_checkbox,
                                       standart_skin_tones,
                                       skin_mask_hue_threshold,
                                       skin_mask_sat_threshold,
                                       skin_mask_val_threshold
                                       ],
                               outputs=[composite_mask_preview,
                                        mask_preview],
                               concurrency_id='fn'
                               )



    # Set parameters
    def skin_mask_checkbox_change(value):
        cfg.auto_generate_skin_mask = value
    skin_mask_checkbox.change(fn=skin_mask_checkbox_change, inputs=skin_mask_checkbox)
           
    

    copy_input_to_output_button.click(fn=lambda x: x, inputs=[input_image],
                                      outputs=[out])

    rotate_left_button.click(fn=kwargs.get('image_rotate'),
                             inputs=[input_image, rotate_left_value],
                             outputs=[input_image])

    rotate_right_button.click(fn=kwargs.get('image_rotate'),
                              inputs=[input_image, rotate_right_value],
                              outputs=[input_image])
    
    # check imputs all 
    r_args = [resize_image_size, minimum_overlap_value, tile_size,
              minimum_density_value]
    
    resize_image_button.click(fn=kwargs.get('resize_image'),
                              inputs=r_args,
                              outputs=[input_image,
                                       grid_image_preview,
                                       current_image_size_value])

    remesh_grid_button.click(fn=kwargs.get('remesh_grid'),
                             inputs=[minimum_overlap_value,
                                     tile_size,
                                     minimum_density_value,
                                     input_image],
                             outputs=[grid_image_preview])

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

    def get_custom_tile_and_mask(ev: gr.SelectData, size):
        ''' Get custom tile from input or output '''
        fn = kwargs.get('get_custom_tile_and_mask')
        tile, mask, x1, y1, x2, y2 = fn(size, (ev.index[0], ev.index[1]))
        return tile, mask, (x1, y1, x2, y2)

    input_image.select(fn=get_custom_tile_and_mask,
                       inputs=[
                           tile_size
                           ],
                       outputs=[
                           custom_tile_source,
                           custom_tile_mask,
                           custom_tile_coordinates
                           ]
                       )
    
    def select_folder_fn(folder):
        ''' Select folder with TK function '''
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        save_file_path = filedialog.askdirectory()
        root.destroy()
        if os.path.isdir(save_file_path):
            return str(save_file_path) + '/'
        return folder

    select_folder_button.click(fn=select_folder_fn,
                               inputs=[output_folder],
                               outputs=[output_folder]
                               )
    
    process_batch_files_button.click(fn=kwargs.get('batch_generate'),
                                     inputs=[
                                         batch_file_list,
                                         strength,
                                         base_prompt,
                                         base_negative_prompt,
                                         steps,
                                         batch_size,
                                         minimum_overlap_value,
                                         tile_size,
                                         minimum_density_value,
                                         batch_output_file_format,
                                         output_folder,
                                         resize_batch_images,
                                         resize_image_size,
                                         mask_blur_value,
                                         mask_expand_value,
                                         label_id,
                                         mask_batch_mode,
                                         fill_mask_if_not_detected
                                         ],
                                     outputs=[out])
    
    generate_custom_tile_button.click(fn=kwargs.get('generate_custom_tile'),
                                      inputs=[
                                          custom_tile_source,
                                          strength,
                                          custom_tile_prompt,
                                          custom_tile_negative_prompt,
                                          steps,
                                          tile_size],
                                      outputs=[custom_tile_generated]
                                      )
    
    apply_custom_tile_button.click(fn=kwargs.get('apply_custom_tile'),
                                   inputs=[
                                       out,
                                       custom_tile_generated,
                                       custom_tile_coordinates,
                                       custom_tile_border_mask_value,
                                       apply_base_mask_for_custom_tile],
                                   outputs=[out])
    
    # generate_clip_mask_button.click(fn=kwargs.get('generate_clip_mask'),
    #                                 inputs=[
    #                                     input_image,
    #                                     keyword_for_clip_mask,
    #                                     tile_size,
    #                                     scale_clip_mask_to_tile],
    #                                 outputs=[
    #                                     clip_mask_tile_source,
    #                                     clip_mask_preview,
    #                                     clip_tile_coordinates
    #                                     ]
    #                                 )
    
    # generate_clip_tile_button.click(fn=kwargs.get('generate_custom_tile'),
    #                                 inputs=[
    #                                     clip_mask_tile_source,
    #                                     strength,
    #                                     clip_mask_prompt,
    #                                     clip_mask_negative_prompt,
    #                                     steps,
    #                                     tile_size,
    #                                     clip_mask_preview],
    #                                 outputs=[clip_tile_preview])
    
    # iii = {minimum_overlap_value, minimum_density_value, batch_size}
    

    return
