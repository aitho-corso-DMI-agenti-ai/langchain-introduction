# LLMs and LangChain introduction

This repository contains example notebooks for the 2025 course [Agenti Intelligenti e Machine Learning (AiTHO)](https://web.dmi.unict.it/it/corsi/l-31/agenti-intelligenti-e-machine-learning-aitho), focusing on the LangChain framework.

## Tech Stack

- **Python**
- **[Marimo](https://marimo.io/)** – A modern alternative to Jupyter for interactive notebooks
- **[LangChain](https://github.com/langchain-ai/langchain)** – A framework for building AI-based conversational chains
- **[LangGraph](https://github.com/langchain-ai/langgraph)** – A framework for building AI agent workflows

## Project Structure

All the example notebooks and code are located in the `notebook/` directory.

All the slides are located in the `slide/` directory.

## Setup Instructions

### 1. Install Poetry

Poetry is the dependency manager used in this project. Follow the [official installation guide](https://python-poetry.org/docs/#installation) to set it up on your system.

### 2. Install Project Dependencies

```bash
poetry install
```

### 3. Setup Anthropic key

Copy the file `.env.example` as `.env` and put your key in the `MISTRAL_API_KEY` field.

### 4. Launch the Notebooks
Use Marimo to edit and run the notebooks:

```bash
poetry run marimo edit notebook/<notebook_name>.py
```
