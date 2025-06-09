FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file
COPY conda.yaml /app/conda.yaml

# Create conda environment
RUN conda env create -f conda.yaml

# Activate environment
SHELL ["conda", "run", "-n", "penguins-ml-env", "/bin/bash", "-c"]

# Copy project files
COPY . /app/

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV PYTHONPATH=/app

# Expose MLflow UI port
EXPOSE 5000

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "penguins-ml-env", "python", "modelling.py"]