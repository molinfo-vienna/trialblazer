#
# BUILD APPLICATION
#
FROM mambaorg/micromamba:2.0.5 AS build

# necessary to display the image on Github
LABEL org.opencontainers.image.source="https://github.com/molinfo-vienna/trialblazer"

# using the root user during the build stage
USER root

# keep Docker from buffering the output so we can see the output of the application in real-time
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy package files first (for caching docker layers)
COPY nerdd-requirements.txt ./

# create environment
# -p /env forces the environment to be created in /env so we don't have to know the env name
RUN micromamba create --copy -p /env -c conda-forge python=3.12 rdkit=2025.09.1 && \
    # install the pip dependencies
    micromamba run -p /env pip install -r nerdd-requirements.txt

# copy the rest of the source code directory and install the main package
COPY . .
RUN micromamba run -p /env pip install --no-deps . && \
    # download model files
    micromamba run -p /env trialblazer-download

#
# RUN APPLICATION
#
FROM gcr.io/distroless/base-debian12

# copy the environment from the build stage
COPY --from=build /env /env
COPY --from=build /root/.trialblazer /root/.trialblazer

ENTRYPOINT ["/env/bin/trialblazer"]