# CSE465 Voice-to-Voice LLM Project

Welcome to the CSE465 Voice-to-Voice Project repository. This project explores and implements speech-to-speech models, primarily focusing on variations of Moshi and related models.

## Repository Structure

The project has been organized into the following directories for clarity and modularity:

- **`notebooks/`**: Contains the Jupyter notebooks used for experiments and modeling.
  - **`active/`**: Current, working notebooks (e.g., `Moshi_Colab_Notebook_v3_FREE_COLAB.ipynb`, `moshiko2_0.ipynb`).
  - **`archive/`**: Deprecated or older test notebooks from previous iterations.
- **`docs/`**: Documentation, proposals, and reports.
  - **`proposals/`**: Initial project proposals and planning documents.
  - **`reports/`**: LaTeX files and project reports.
  - **`resources/`**: Reference materials, research papers, project guides, and paper reading summaries (e.g., AlexNet, Voice Deepfake Detectors, Speech-to-Speech Translation).
- **`data/`**: Datasets and test files.
  - **`audio/`**: Audio files (`.wav`, `.mp3`) used as input or output for the models.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/RRM006/cse465.git
   ```
2. Navigate to the `notebooks/active/` folder to explore the primary Voice-to-Voice models and Colab notebooks.
3. Review the `docs/resources/` folder for related reading material and architectural context.

## Note on Credentials
Please ensure that you do not commit active Hugging Face tokens or other secrets within the Jupyter Notebooks. Always use environment variables or local secret managers.
