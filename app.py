import os
import natsort
import gradio as gr
from eval import util, interpolator
from tqdm import tqdm
import tensorflow as tf
import numpy as np

def process_directory(input_dir, suffix=".png"):

    print(f'processing directory: {input_dir}')
    print(f'matching files with suffix: {suffix}')

    if not os.path.exists(input_dir):
        return '', f'Path not found: {input_dir}', '', None, None

    input_path = os.path.abspath(input_dir)

    matched_files = natsort.natsorted([x for x in os.listdir(input_path) if
                           os.path.isfile(os.path.join(input_path, x)) and os.path.splitext(x)[-1] == suffix])

    info = f"Input Directory: {input_path}\nNumber of Files: {len(matched_files)}"
    matched_files_str = f"{os.linesep}".join(matched_files)

    # Your interpolation logic here
    output_dir = os.path.join(input_dir, 'interpolated_frames')

    first_image_path = os.path.join(input_path, matched_files[0])
    first_img = util.read_image(first_image_path)

    last_image_path = os.path.join(input_path, matched_files[-1])
    last_img = util.read_image(last_image_path)

    return output_dir, info, matched_files_str, first_img, last_img


def run_film(input_dir, matched_files_info, output_dir, model_path, times_to_interpolate):

    from eval import interpolator

    matched_files_info = matched_files_info.split(os.linesep)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    interpolator = interpolator.Interpolator(model_path, None)

    # Batched time.
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    input_frames = [os.path.join(input_dir, x) for x in matched_files_info]

    num_frames = len(input_frames)
    print(f'Generating {num_frames} in-between frames for {input_dir}')

    print(f'Making directory {output_dir}')
    tf.io.gfile.makedirs(output_dir)

    print(input_frames[0], matched_files_info[0])

    print(os.path.abspath(input_frames[0]))

    for i, frame in tqdm(enumerate(util.interpolate_recursively_from_files(
            input_frames, times_to_interpolate, interpolator))):
        util.write_image(f'{output_dir}/frame_{i:05d}.png', frame)

    return '', ''

with gr.Blocks(title='FILM Frame Interpolation') as demo:

    with gr.Row():
        with gr.Column():

            input_dir = gr.Textbox(label="Input Directory")
            filetype = gr.Textbox(value=".png", label="Filetype (.png, .jpg, .jpeg)", interactive=True)
            process_dir = gr.Button(value="Read Input Directory")

            output_dir = gr.Textbox(label="Output Directory", interactive=True)
            with gr.Row():
                input_dir_info = gr.Textbox(label="Input Directory Info", max_lines=10)
                matched_files_info = gr.Textbox(label="Matched Files", max_lines=10)

            with gr.Row():
                first_image = gr.Image(label='First Image').style(height=256, width=256)
                last_image = gr.Image(label='Last Image').style(height=256, width=256)

        with gr.Column():

            model_path = gr.Textbox(value='pretrained_models/film_net/Style/saved_model',
                                    label="Model Path", interactive=True)

            # TODO: Make this a slider (1-10)
            times_to_interpolate = gr.Number(value=5, label="Times to Interpolate", interactive=True)


            exec_film = gr.Button(value="Run FILM")

            film_details = gr.Textbox(label="Details", interactive=False)

            # TODO: Make this an iterative output
            film_output = gr.Textbox(label="FILM progress", interactive=False)


    process_dir.click(process_directory,
                      inputs=[input_dir, filetype],
                      outputs=[output_dir, input_dir_info, matched_files_info, first_image, last_image],
                      api_name="process-directory")

    exec_film.click(run_film,
                    inputs=[input_dir, matched_files_info, output_dir, model_path, times_to_interpolate],
                    outputs=[film_details, film_output],
                    api_name='exec-film')

if __name__ == "__main__":
    demo.launch()