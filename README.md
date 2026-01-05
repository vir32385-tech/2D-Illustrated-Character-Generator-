# 2D Illustrated Character Generator

**2D Illustrated Character Generator** is a professional AI-powered application designed to generate high-quality 2D illustrated character images from short text descriptions. The tool allows users to choose between **Normal** and **Horror** character styles while maintaining strong artistic consistency and controlled visual output.

This project is built with a focus on clarity, stability, and controlled generation rather than random or overly generic AI outputs. It is intended to function as a reliable character creation tool rather than a simple image generation demo.

---

## Overview

The application provides a clean and minimal interface where users can describe a character using visual details such as facial features, hair, clothing, and expression. Based on the selected style, the system internally applies predefined artistic rules before generating the final character image.

The internal prompt system is deliberately abstracted and not exposed to the user. This ensures consistent results, protects internal logic, and prevents style degradation caused by unrestricted user inputs.

---

## Key Features

- Text-to-image character generation
- Supports **Normal** and **Horror** character styles
- Controlled and consistent 2D illustration output
- Internal rule-based prompt system
- Clean, minimal, and mobile-friendly UI
- Free-tier compatible deployment using Streamlit
- No prompt leakage or user-side prompt manipulation

---

## How It Works

1. The user enters a short character description focused on visual traits.
2. The user selects a character style (Normal or Horror).
3. The system applies internal illustration and style rules based on the selected mode.
4. A Stable Diffusion-based model generates the final character image.
5. The generated image is displayed directly in the application interface.

The internal prompt logic is handled entirely within the backend and is never shown or editable through the user interface.

---

## Usage Guidelines

- Descriptions should be concise and visual (e.g., facial structure, hair, mood).
- Avoid technical or quality-related keywords such as “8K,” “ultra realistic,” or “photorealistic.”
- Image generation on free-tier infrastructure may take between 1–3 minutes.

For best results, keep descriptions simple and focused.

---

## Technology Stack

- **Python**
- **Streamlit**
- **Stable Diffusion (Diffusers)**
- **PyTorch**

The application is designed to remain lightweight and stable within free hosting limitations.

---

## Project Structure

The repository intentionally uses a minimal structure to reduce deployment issues:

- `app.py` – Main application logic and UI
- `requirements.txt` – Dependency list
- `README.md` – Project documentation

No additional configuration or runtime files are required.

---

## Disclaimer

This project is intended for educational, experimental, and prototyping purposes. Performance and output quality may vary depending on system resources and hosting limitations. The application does not guarantee consistent generation speed or identical outputs across different runs.

---

## License

This project is provided as-is without any warranty. Usage and modification are permitted for learning and experimentation purposes.
