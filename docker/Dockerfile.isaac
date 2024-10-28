## -----------------------------------------
## Stage 1: Base dependencies installation.
## -----------------------------------------
FROM nvcr.io/nvidia/isaac-sim:4.1.0 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Update system packages and install the basic ones
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
    apt-utils \
    software-properties-common \
    build-essential \
    sudo

# Copy apt dependencies file
COPY ./docker/deps/apt_requirements.txt /docker/deps/apt_requirements.txt

# Install additional dependencies from requirements file
RUN xargs -d '\n' -a /docker/deps/apt_requirements.txt \
    apt-get -y install \
    --no-install-recommends

# Clean up to reduce image size
RUN apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*


## -----------------------------------------
## Stage 2: User setup.
## -----------------------------------------
FROM base AS setup

# User and group ids to map the docker user to the host user.
ARG USER_ID
ARG GROUP_ID
ARG SHARED_GROUP_ID

ARG DOCKER_USERNAME="user"
ENV USERNAME $DOCKER_USERNAME

# Set default shell to bash.
SHELL ["/bin/bash", "-c"]

# Add a non-root docker user with same user and group id as the host user
# so files created in the docker container are owned by the host.
RUN if ! id $USERNAME >/dev/null 2>&1; then \
        adduser \
            --uid $USER_ID \
            --ingroup $( \
                (addgroup --gid $GROUP_ID $USERNAME >/dev/null && echo $USERNAME) || \
                (getent group $GROUP_ID | cut -d: -f1) ) \
            --home /home/$USERNAME \
            --shell /bin/bash \
            --disabled-password \
            --gecos "" \
            $USERNAME; \    
    fi

# Give the user passwordless sudo permissions
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Create shared group with same GID as a shared host group (e.g. vip_user) and
# add docker user to this secondary group so that they can access any shared
# dirs that might need to be mounted.
ENV SHARED_GROUP_NAME="host_shared"
RUN if [ -n $SHARED_GROUP_ID ]; then \
        groupadd -g $SHARED_GROUP_ID $SHARED_GROUP_NAME && \
        usermod -a -G $SHARED_GROUP_NAME $USERNAME; \
    fi

# MUCH FASTER than local recursive chown. see https://github.com/docker/for-linux/issues/388
COPY --from=base --chown=${USER_ID}:${GROUP_ID} /isaac-sim /isaac-sim

# Add local user binary folder to PATH variable
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"


## -----------------------------------------
## Stage 3: Setting Up Isaac-Sim.
## -----------------------------------------
FROM setup AS isaac-sim

ARG PROJECT_NAME

# Copy dependency files
COPY ./docker/deps/entrypoint.sh /home/${USERNAME}/entrypoint.sh
COPY ./docker/deps/setup_docker_env.sh /home/${USERNAME}/setup_docker_env.sh
COPY ./docker/deps/pip_requirements.txt /home/${USERNAME}/pip_requirements.txt

USER root
RUN chmod +x /home/$USERNAME/entrypoint.sh && \
    chown $USER_ID:$GROUP_ID /home/$USERNAME/entrypoint.sh && \
    chmod +x /home/$USERNAME/setup_docker_env.sh && \
    chown $USER_ID:$GROUP_ID /home/$USERNAME/setup_docker_env.sh

USER $USERNAME

# we insert our script source to the start of our .bashrc, so it runs even when bash is non-interactive
RUN sed -i '1s/^/source \/home\/'"$USERNAME"'\/setup_docker_env.sh\n/' /home/$USERNAME/.bashrc && \
    source /home/$USERNAME/setup_docker_env.sh && \
    pip install --upgrade pip && \
    pip install -r /home/$USERNAME/pip_requirements.txt

WORKDIR /$PROJECT_NAME
ENTRYPOINT /bin/bash -i /home/$USERNAME/entrypoint.sh


## -----------------------------------------
## Stage 4: Setting Up Isaac-Lab.
## -----------------------------------------
FROM isaac-sim AS isaac-lab

USER root

ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} && \
    ln -sf ${ISAACSIM_PATH} ${ISAACLAB_PATH}/_isaac_sim && \
    cd ${ISAACLAB_PATH} && \
    source /home/$USERNAME/setup_docker_env.sh && \
    ./isaaclab.sh --install && \
    chown -R $USER_ID:$GROUP_ID ${ISAACLAB_PATH}

# Install apt dependencies for extensions that declare them in their extension.toml
RUN /home/$USERNAME/setup_docker_env.sh && \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source/extensions && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${ISAACLAB_PATH}

USER $USERNAME
