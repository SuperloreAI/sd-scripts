# Beta V0.73
# Update by 0xnewton
import numpy as np
from tqdm import trange
from PIL import Image, ImageSequence, ImageDraw, ImageOps
import math
import os

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from modules import deepbooru


# Define a key function to extract the numerical part of the filename
# Important: filename should be like /someFile-00123.png
# Example usage:
# sorted_files = sorted(files, key=numerical_part)
def numerical_part(filename):
    # Find the index of the first digit in the filename
    index = next((i for i, c in enumerate(filename) if c.isdigit()), None)
    if index is not None:
        # Extract the numerical part of the filename
        return int(filename[index:].split(".")[0])
    else:
        # If the filename doesn't contain any digits, return 0
        return 0


def index_exists(lst, indexes):
    """
    Check if an arbitrary nested index exists in a list.

    Args:
        lst (list): The list to check.
        indexes (list): A list of indexes to check, where each index corresponds to a nested level in the list.

    Returns:
        bool: True if the index exists, False otherwise.
    """
    for index in indexes:
        if isinstance(lst, list) and index < len(lst):
            lst = lst[index]
        else:
            return False
    return True


class Script(scripts.Script):
    def title(self):
        return "(Beta) Multi-frame Video rendering - V0.73 - 0xnewton"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        first_denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label='Initial Denoise Strength', value=1,
                                  elem_id=self.elem_id("first_denoise"))
        append_interrogation = gr.Dropdown(label="Append interrogated prompt at each iteration",
                                           choices=["None", "CLIP", "DeepBooru"], value="None")
        third_frame_image = gr.Dropdown(label="Third Frame Image",
                                        choices=["None", "FirstGen", "GuideImg", "Historical"], value="None")
        # reference_imgs = gr.UploadButton(label="Upload Guide Frames", file_types = ['.png','.jpg','.jpeg'], live=True, file_count = "multiple")
        reference_img_folder = gr.Textbox(label="Reference images",
                                          placeholder="The frames you want to animate")
        control_net_folder1 = gr.Textbox(label="Controlnet 1 Input Directory",
                                         placeholder="A directory for input images to controlnet unit 1 (if applicable)")
        control_net_folder1_masks = gr.Textbox(label="Controlnet 1 Masks (optional)",
                                               placeholder="A directory for masks to be applied to control net 1 source images (if applicable)")
        control_net_folder2 = gr.Textbox(label="Controlnet 2 Input Directory",
                                         placeholder="A directory for input images to controlnet unit 2 (if applicable)")
        control_net_folder2_masks = gr.Textbox(label="Controlnet 2 Masks (optional)",
                                               placeholder="A directory for masks to be applied to control net 2 source images (if applicable)")
        control_net_folder3 = gr.Textbox(label="Controlnet 3 Input Directory",
                                         placeholder="A directory for input images to controlnet unit 3 (if applicable)")
        control_net_folder3_masks = gr.Textbox(label="Controlnet 3 Masks (optional)",
                                               placeholder="A directory for masks to be applied to control net 3 source images (if applicable)")
        control_net_folder4 = gr.Textbox(label="Controlnet 4 Input Directory",
                                         placeholder="A directory for input images to controlnet unit 4 (if applicable)")
        control_net_folder4_masks = gr.Textbox(label="Controlnet 4 Masks (optional)",
                                               placeholder="A directory for masks to be applied to control net 4 source images (if applicable)")
        outputFolder = gr.Textbox(label="Output Directory",
                                  placeholder="Where to save data to")
        color_correction_enabled = gr.Checkbox(label="Enable Color Correction", value=False,
                                               elem_id=self.elem_id("color_correction_enabled"))
        unfreeze_seed = gr.Checkbox(label="Unfreeze Seed", value=False, elem_id=self.elem_id("unfreeze_seed"))
        loopback_source = gr.Dropdown(label="Loopback Source", choices=["PreviousFrame", "InputFrame", "FirstGen"],
                                      value="PreviousFrame")
        n_frames = gr.Number(label="Number of frames to use for comparison", value=3, min=1, max=20)

        return [append_interrogation, reference_img_folder, control_net_folder1, control_net_folder1_masks,
                control_net_folder2, control_net_folder2_masks,
                control_net_folder3, control_net_folder3_masks, control_net_folder4, control_net_folder4_masks,
                outputFolder, first_denoise, third_frame_image,
                color_correction_enabled, unfreeze_seed, loopback_source, n_frames]

    def run(self, p, append_interrogation, reference_img_folder, control_net_folder1, control_net_folder1_masks,
            control_net_folder2, control_net_folder2_masks,
            control_net_folder3, control_net_folder3_masks, control_net_folder4, control_net_folder4_masks,
            outputFolder, first_denoise, third_frame_image,
            color_correction_enabled, unfreeze_seed, loopback_source, n_frames):

        freeze_seed = not unfreeze_seed

        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)
        p.outpath_samples = outputFolder
        p.outpath_grid = outputFolder

        reference_imgs = [os.path.join(reference_img_folder, x) for x in
                          sorted(os.listdir(reference_img_folder), key=numerical_part)]
        cn_files = []  # list of controlnet files
        cn_masks = []  # list of masks to apply controlnet to
        if control_net_folder1:
            cn_files.append([os.path.join(control_net_folder1, x) for x in
                             sorted(os.listdir(control_net_folder1), key=numerical_part)])
            if control_net_folder1_masks:
                cn_masks.append([os.path.join(control_net_folder1_masks, x) for x in
                                 sorted(os.listdir(control_net_folder1_masks), key=numerical_part)])
            else:
                cn_masks.append([])

        if control_net_folder2:
            cn_files.append([os.path.join(control_net_folder2, x) for x in
                             sorted(os.listdir(control_net_folder2), key=numerical_part)])
            if control_net_folder2_masks:
                cn_masks.append([os.path.join(control_net_folder2_masks, x) for x in
                                 sorted(os.listdir(control_net_folder2_masks), key=numerical_part)])
            else:
                cn_masks.append([])
        if control_net_folder3:
            cn_files.append([os.path.join(control_net_folder3, x) for x in
                             sorted(os.listdir(control_net_folder3), key=numerical_part)])
            if control_net_folder3_masks:
                cn_masks.append([os.path.join(control_net_folder3_masks, x) for x in
                                 sorted(os.listdir(control_net_folder3_masks), key=numerical_part)])
            else:
                cn_masks.append([])
        if control_net_folder4:
            cn_files.append([os.path.join(control_net_folder4, x) for x in
                             sorted(os.listdir(control_net_folder4), key=numerical_part)])
            if control_net_folder4_masks:
                cn_masks.append([os.path.join(control_net_folder4_masks, x) for x in
                                 sorted(os.listdir(control_net_folder4_masks), key=numerical_part)])
            else:
                cn_masks.append([])

        loops = len(reference_imgs)

        processing.fix_seed(p)
        batch_count = p.n_iter

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        initial_width = p.width
        initial_img = p.init_images[0]

        grids = []
        all_images = []
        original_init_image = p.init_images
        original_prompt = p.prompt
        original_denoise = p.denoising_strength
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        for n in range(batch_count):
            history = []
            frames = []
            third_image = None
            third_image_index = 0
            frame_color_correction = None

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image
            p.width = initial_width

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True
                p.control_net_input_image = []
                for iii, cn_files_unit in enumerate(cn_files):
                    cn_file = cn_files_unit[i]
                    image = Image.open(cn_file).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS)
                    if index_exists(cn_masks, [iii, i]):
                        mask_image = Image.open(cn_masks[iii][i]).convert("L").resize((initial_width, p.height))
                        mask_thing = ImageOps.fit(image, image.size, centering=(0.5, 0.5)).convert("RGBA")
                        mask_thing.putalpha(mask_image)

                        # Create a new image with only the masked pixels
                        masked_image = Image.new("RGBA", (initial_width, p.height), color=(0, 0, 0, 0))
                        masked_image.paste(image, mask=mask_thing)
                        image = masked_image

                    p.control_net_input_image.append(image)

                ref_image = Image.open(reference_imgs[i]).convert("RGB").resize((initial_width, p.height),
                                                                                Image.ANTIALIAS)
                if (i > 0):
                    loopback_image = p.init_images[0]
                    if loopback_source == "InputFrame":
                        # loopback_image = p.control_net_input_image
                        loopback_image = ref_image
                    elif loopback_source == "FirstGen":
                        loopback_image = history[0]

                    if third_frame_image != "None" and i > 1:
                        n_frames_in_iteration = (i + 1) if (i + 1) < n_frames else n_frames
                        p.width = initial_width * n_frames_in_iteration
                        img = Image.new("RGB", (initial_width * n_frames_in_iteration, p.height))
                        for i in range(n_frames_in_iteration):
                            if i == 1:
                                img.paste(loopback_image, (initial_width * i, 0))
                            elif (i == n_frames_in_iteration - 1): 
                                img.paste(third_image, (initial_width * i, 0))
                            else:
                                img.paste(history[len(history) - 1 - i], (initial_width * i, 0))
                            
                        img.paste(p.init_images[0], (0, 0))
                        # img.paste(p.init_images[0], (initial_width, 0))
                        img.paste(loopback_image, (initial_width, 0))
                        img.paste(third_image, (initial_width * 2, 0))
                        p.init_images = [img]
                        if color_correction_enabled:
                            p.color_corrections = [processing.setup_color_correction(img)]

                        for control_idx, control_img in enumerate(p.control_net_input_image):
                            msk = Image.new("RGB", (initial_width * n_frames_in_iteration, p.height))
                            for i in range(n_frames_in_iteration):
                                if i == 1:
                                    msk.paste(control_img, (initial_width * i, 0))
                                elif (i == n_frames_in_iteration - 1): 
                                    msk.paste(reference_imgs[third_image_index], (initial_width * i, 0))
                                else:
                                    msk.paste(reference_imgs[len(history) - 1 - i], (initial_width * i, 0))
                            # msk.paste(Image.open(reference_imgs[i - 1]).convert("RGB").resize((initial_width, p.height),
                            #                                                                   Image.ANTIALIAS), (0, 0))
                            # msk.paste(control_img, (initial_width, 0))
                            # msk.paste(Image.open(reference_imgs[third_image_index]).convert("RGB").resize(
                            #     (initial_width, p.height), Image.ANTIALIAS), (initial_width * 2, 0))

                            p.control_net_input_image[control_idx] = msk

                        latent_mask = Image.new("RGB", (initial_width * n_frames_in_iteration, p.height), "black")
                        latent_draw = ImageDraw.Draw(latent_mask)
                        latent_draw.rectangle((initial_width, 0, initial_width * 2, p.height), fill="white")
                        p.image_mask = latent_mask
                        p.denoising_strength = original_denoise
                    else:
                        p.width = initial_width * 2
                        img = Image.new("RGB", (initial_width * 2, p.height))
                        img.paste(p.init_images[0], (0, 0))
                        # img.paste(p.init_images[0], (initial_width, 0))
                        img.paste(loopback_image, (initial_width, 0))
                        p.init_images = [img]
                        if color_correction_enabled:
                            p.color_corrections = [processing.setup_color_correction(img)]

                        cn_imgs_copy = p.control_net_input_image.copy()
                        for control_idx, control_img in enumerate(cn_imgs_copy):
                            msk = Image.new("RGB", (initial_width * 2, p.height))
                            msk.paste(Image.open(reference_imgs[i - 1]).convert("RGB").resize((initial_width, p.height),
                                                                                              Image.ANTIALIAS), (0, 0))
                            msk.paste(control_img, (initial_width, 0))

                            p.control_net_input_image[control_idx] = msk
                            frames.append(msk)

                        # latent_mask = Image.new("RGB", (initial_width*2, p.height), "white")
                        # latent_draw = ImageDraw.Draw(latent_mask)
                        # latent_draw.rectangle((0,0,initial_width,p.height), fill="black")
                        latent_mask = Image.new("RGB", (initial_width * 2, p.height), "black")
                        latent_draw = ImageDraw.Draw(latent_mask)
                        latent_draw.rectangle((initial_width, 0, initial_width * 2, p.height), fill="white")

                        # p.latent_mask = latent_mask
                        p.image_mask = latent_mask
                        p.denoising_strength = original_denoise
                else:
                    latent_mask = Image.new("RGB", (initial_width, p.height), "white")
                    # p.latent_mask = latent_mask
                    p.image_mask = latent_mask
                    p.denoising_strength = first_denoise
                    # p.control_net_input_image = p.control_net_input_image.resize((initial_width, p.height))
                    cn_imgs_copy = p.control_net_input_image.copy()
                    for control_idx, control_img in enumerate(cn_imgs_copy):
                        p.control_net_input_image[control_idx] = control_img.resize((initial_width, p.height))

                        # Should this write every time?
                        frames.append(p.control_net_input_image[control_idx])

                if append_interrogation != "None":
                    p.prompt = original_prompt + ", " if original_prompt != "" else ""
                    if append_interrogation == "CLIP":
                        p.prompt += shared.interrogator.interrogate(p.init_images[0])
                    elif append_interrogation == "DeepBooru":
                        p.prompt += deepbooru.model.tag(p.init_images[0])

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                processed = processing.process_images(p)
                print(f'finished iteration processing, recieved {len(processed.images)} processed imgs')

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = processed.images[0]
                if (i > 0):
                    init_img = init_img.crop((initial_width, 0, initial_width * 2, p.height))

                if third_frame_image != "None":
                    if third_frame_image == "FirstGen" and i == 0:
                        print('using first gen third frame')
                        third_image = init_img
                        third_image_index = 0
                    elif third_frame_image == "GuideImg" and i == 0:
                        print('using guide img 3rd frame')
                        third_image = original_init_image[0]
                        third_image_index = 0
                    elif third_frame_image == "Historical":
                        print('using historical 3rd frame')
                        third_image = processed.images[0].crop((0, 0, initial_width, p.height))
                        third_image_index = (i - 1)

                p.init_images = [init_img]
                if (freeze_seed):
                    p.seed = processed.seed
                else:
                    p.seed = processed.seed + 1

                history.append(init_img)
                if opts.samples_save:
                    images.save_image(init_img, p.outpath_samples, "Frame", p.seed, p.prompt, opts.grid_format,
                                      info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                for img in processed.images:
                    frames.append(img)

            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info,
                                  short_filename=not opts.grid_extended_filename, grid=True, p=p)

            grids.append(grid)
            # all_images += history + frames
            all_images += history

            p.seed = p.seed + 1

        if opts.return_grid:
            all_images = grids + all_images

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
