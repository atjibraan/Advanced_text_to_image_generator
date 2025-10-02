import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
import random

# -----------------------
# Configuration
# -----------------------
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_steps = 35
    default_guidance = 9
    default_size = (512, 512)
    model_id = "stabilityai/stable-diffusion-2"

# -----------------------
# Load Model
# -----------------------
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
revision = "fp16" if torch.cuda.is_available() else None

pipe = StableDiffusionPipeline.from_pretrained(
    CFG.model_id,
    torch_dtype=dtype,
    revision=revision,
)
pipe = pipe.to(CFG.device)

# -----------------------
# Image Generation Function
# -----------------------
def generate_image(prompt, steps, guidance, width, height, num_images, seed):
    if not prompt or prompt.strip() == "":
        return [], "⚠️ Please enter a valid prompt."

    if seed == -1:
        seed = random.randint(0, 999999)
    generator = torch.Generator(CFG.device).manual_seed(seed)

    try:
        images = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            width=width,
            height=height,
            num_images_per_prompt=num_images
        ).images
        return images, f"✅ Used Seed: {seed}"
    except Exception as e:
        return [], f"❌ Error: {str(e)}"

# -----------------------
# Gradio UI (Gradio 4.x)
# -----------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Advanced Text-to-Image Generator (Stable Diffusion 2)")
    gr.Markdown("Generate AI art with customizable settings. Powered by Hugging Face 🤗 Diffusers.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Enter your prompt", placeholder="e.g. futuristic city at sunset")
            steps = gr.Slider(10, 75, value=CFG.default_steps, step=1, label="Steps")
            guidance = gr.Slider(1, 15, value=CFG.default_guidance, step=0.5, label="Guidance Scale")
            width = gr.Dropdown([256, 384, 512, 768], value=CFG.default_size[0], label="Width")
            height = gr.Dropdown([256, 384, 512, 768], value=CFG.default_size[1], label="Height")
            num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            btn = gr.Button("Generate")

        with gr.Column(scale=2):
            output = gr.Gallery(label="Generated Images", show_label=True, columns=2, height="auto")
            seed_out = gr.Textbox(label="Generation Info", interactive=False)

    btn.click(fn=generate_image,
              inputs=[prompt, steps, guidance, width, height, num_images, seed],
              outputs=[output, seed_out])

# -----------------------
# Launch App
# -----------------------
if __name__ == "__main__":
    demo.launch()
