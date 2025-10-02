# Advanced_text_to_image_generator

# 🎨 Advanced Text-to-Image Generator (Stable Diffusion 2)

A simple yet powerful **Gradio app** for generating AI art using [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2).
This project leverages **Hugging Face Diffusers**, **PyTorch**, and **Gradio** to create a customizable text-to-image interface.

---

## 🚀 Features

* Generate images from **text prompts**.
* Control **inference steps** (quality vs. speed trade-off).
* Adjust **guidance scale** for creativity vs. prompt adherence.
* Choose image **resolution** (256–768 px).
* Generate **1–4 images** at once.
* Use **fixed or random seeds** for reproducibility.
* Interactive **Gradio web UI** with gallery output.

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/stable-diffusion-gradio.git
cd stable-diffusion-gradio

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121  # CUDA version if supported
pip install diffusers==0.24.0 transformers accelerate safetensors
pip install gradio==4.29.0
```

---

## ▶️ Usage

Run the app:

```bash
python app.py
```

The Gradio interface will launch, and you can access it in your browser at:

```
http://127.0.0.1:7860
```

---

## ⚙️ Configuration

* **Device**: Automatically uses GPU (`cuda`) if available, otherwise falls back to CPU.
* **Default settings**:

  * Steps: `35`
  * Guidance: `9`
  * Image size: `512x512`
  * Model: `stabilityai/stable-diffusion-2`

---

## 📂 Project Structure

```
.
├── app.py          # Main Gradio app script
├── README.md       # Project documentation
```

---

## 🖼️ Example

Prompt: *"A futuristic city at sunset, cinematic lighting"*

<img src="example.png" width="500" />

---

## 💡 Notes

* Ensure you have a GPU with at least **6GB VRAM** for faster generation. CPU fallback works but is slower.
* You may need to log in with a [Hugging Face account](https://huggingface.co) to access the Stable Diffusion model.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgments

* [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* [Stable Diffusion](https://stability.ai)
* [Gradio](https://www.gradio.app/)

---
