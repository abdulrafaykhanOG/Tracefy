Tracefy
Tracefy is an advanced AI-powered system that transforms rough criminal sketches and descriptive text prompts into realistic facial images. Built for forensic and law enforcement use, Tracefy leverages cutting-edge generative models to modernize suspect identification and support efficient investigations.

🎥 Demo Video
Watch the full walkthrough on YouTube:
📺 Tracefy Demonstration Video
(Includes user interface, sketch upload, prompt iteration, and image generation pipeline)

🔍 Project Overview
Tracefy bridges the gap between victim/witness memory and actionable visual evidence using a multi-stage pipeline involving sketch refinement and text-guided image generation. Users can upload an outline sketch and describe the suspect in natural language; the system then generates a photorealistic face image combining both inputs.

✨ Key Features
Sketch-to-Image Translation: Converts user-drawn outlines into high-resolution images.

Dual Conditioning: Merges sketch geometry with semantic guidance from text prompts.

Real-Time Generation: Fast, web-based inference pipeline using GPU-accelerated models.

User Interface: Intuitive frontend built in React.js for secure, role-based interaction.

Bias Mitigation: Actively monitors fairness and demographic balance in outputs.

Law Enforcement Tools: Secure dashboard for managing investigations and retrieval.

🧠 Technical Architecture
Pipeline Stages
Sketch Refinement
Refines rough line art using a CycleGAN-based model trained on unpaired sketches.

Image Generation
Combines refined sketch + text using a FLUX.1-dev latent diffusion model and ControlNet-Union-Pro-2.0 for structure-aware generation.

Prompt Iteration
Allows multiple refinements per session to perfect suspect likeness.

Core Models
CycleGAN – For sketch cleanup and style normalization.

FLUX.1-dev + ControlNet – For generating realistic face images with dual inputs.

LoRA Fine-Tuning – For efficient, domain-specific model adaptation.

🛠 Tech Stack
Layer	Technology
Frontend	React.js
Backend	Flask (Python)
ML Framework	PyTorch, TensorFlow
Data Storage	MySQL, MongoDB
Model Infra	Google Colab, Hugging Face

📦 Functional Modules
User Authentication (RBAC, secure sessions)

Sketch Upload + Prompt Input

Sketch Refinement Preview

Face Generation

Image History + Download

Law Enforcement Dashboard

📊 Evaluation Metrics
Canny Edge Similarity – Measures structural accuracy.

Perceptual Hash (pHash) – Assesses visual similarity.

Hamming Distance – Quantifies binary differences between image hashes.

📈 Business Model
Tracefy is a B2G (Business-to-Government) solution with potential use cases in:

Police departments and forensic units

Investigative journalism

Security and surveillance tech firms

🚀 Future Roadmap
📱 Mobile App Deployment

🌍 South Asian Facial Dataset Acquisition

🤝 Institutional Collaborations

🔁 Integration with National Criminal Databases (e.g., NADRA)

⚙️ Deployment & CI/CD
Containerized with Docker for backend and frontend services

GitHub Actions used for automated testing and deployment

WebSockets for real-time progress updates during image generation

🤝 Contributors
Mahad Mohtashim

Mehar Ali Ahmed

Abdul Rafay Khan
Supervised by Dr. Rafia Mumtaz & Dr. Muhammad Daud Abdullah Asif

📜 License
Tracefy is developed for academic and public safety purposes. For licensing inquiries, please contact the authors.

