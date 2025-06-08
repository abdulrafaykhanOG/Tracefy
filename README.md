# Tracefy

**Tracefy** is an advanced AI-powered system that transforms rough criminal sketches and descriptive text prompts into realistic facial images. Built for forensic and law enforcement applications, Tracefy leverages cutting-edge generative models to modernize suspect identification and accelerate investigations.

---

## Project Overview

Tracefy bridges the gap between victim/witness memory and actionable visual evidence using a multi-stage AI pipeline involving sketch refinement and text-guided image generation.

**Input:** User-drawn outline sketch + natural language description  
**Output:** High-resolution photorealistic facial image
---

## Demo Video

https://github.com/user-attachments/assets/135e0008-9b91-470f-ba7c-ac142a291200

---

## Key Features

- **Sketch-to-Image Translation**  
  Converts rough outlines into high-resolution face images.

- **Dual Conditioning**  
  Fuses geometry (sketch) with semantic cues (text) to guide image generation.

- **Real-Time Inference**  
  GPU-accelerated pipeline for fast, web-based generation.

- **User Interface**  
  React.js frontend with secure, role-based access (RBAC).

- **Bias Mitigation**  
  Built-in fairness monitoring to ensure demographic balance.

- **Law Enforcement Tools**  
  Secure dashboard to manage sketches, generation history, and retrievals.

---

## Technical Architecture

### Pipeline Stages

1. **Sketch Refinement**  
   Uses a CycleGAN-based model for line cleanup and style normalization.

2. **Image Generation**  
   Integrates refined sketch and user prompt using:
   - FLUX.1-dev (Latent Diffusion Model)
   - ControlNet-Union-Pro-2.0 (structure-aware synthesis)

3. **Prompt Iteration**  
   Interactive session flow for refining and improving generated faces.

### Core Models

| Task               | Model                                  |
|--------------------|-----------------------------------------|
| Sketch Cleanup     | CycleGAN                                |
| Image Generation   | FLUX.1-dev + ControlNet                 |
| Fine-tuning        | LoRA (Low-Rank Adaptation)              |

---

## Tech Stack

| Layer          | Technology                                 |
|----------------|---------------------------------------------|
| Frontend       | React.js                                    |
| Backend        | Flask (Python)                              |
| ML Framework   | PyTorch, TensorFlow                         |
| Data Storage   | MySQL, MongoDB                              |
| Model Infra    | Google Colab, Hugging Face                  |

---

## Functional Modules

- User Authentication (RBAC)
- Sketch Upload & Prompt Input
- Sketch Refinement Preview
- Face Generation & Iteration
- Image History & Download
- Law Enforcement Dashboard

---

## Evaluation Metrics

| Metric                  | Purpose                             |
|-------------------------|--------------------------------------|
| Canny Edge Similarity   | Structural accuracy measurement      |
| Perceptual Hash (pHash) | Visual resemblance scoring           |
| Hamming Distance        | Quantifies binary differences in hashes |


## Business Model

Tracefy is a **B2G (Business-to-Government)** solution. Target sectors include:

- Police departments and forensic teams  
- Investigative journalism units  
- Security and surveillance technology providers

---

## Future Roadmap

- Mobile App Deployment  
- Acquisition of South Asian Facial Dataset  
- Academic & Institutional Collaborations  
- Integration with National Criminal Databases (e.g. NADRA)

---

## Project Poster

![Poster](https://github.com/user-attachments/assets/d0a2fb86-ee27-46ed-9685-ab83ed78cb43)

## License

Tracefy is developed for academic and public safety purposes.  For licensing or usage inquiries, please contact the authors.
