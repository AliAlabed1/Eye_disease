# Use Arch Linux base image
FROM archlinux:latest

# Set the working directory in the container
WORKDIR /workspaces

# Update Arch and install necessary packages
RUN pacman -Syu --noconfirm && pacman -S --noconfirm python python-pip git python-virtualenv

# Clone the Git repository each time the container is started
CMD ["sh", "-c", "git clone https://github.com/AliAlabed1/Eye_disease.git && \
    cd Eye_disease && \
    python -m venv venv && \
    source venv/bin/activate && \
    ls && \
    pip install -r requirements.txt && \
    uvicorn src.api.main:APP --host 0.0.0.0 --port 8000 --reload"]