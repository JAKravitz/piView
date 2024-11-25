# Base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /home

# Copy environment file and install dependencies
COPY requirements.yml .
RUN conda env create -f requirements.yml

# Ensure the environment is activated and accessible
SHELL ["conda", "run", "-n", "piView_environment", "/bin/bash", "-c"]

# Install Jupyter and clean up
RUN conda install -n piView_environment jupyterlab -y && conda clean -a -y
RUN apt-get update && apt-get install -y wget

# Copy the notebook and lib files into the container
COPY piView.ipynb .
COPY lib2.py .
 
# Expose port for Jupyter Notebook
EXPOSE 8888
EXPOSE 5006

# Run JupyterLab within the environment
CMD ["conda", "run", "-n", "piView_environment", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
