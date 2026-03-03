FROM rocker/r-ver:4.3.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libfftw3-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    libcurl4-openssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN R -q -e "options(repos = c(CRAN='https://cloud.r-project.org')); install.packages(c('jpeg','tiff','logging','BiocManager'))"
RUN R -q -e "BiocManager::install('EBImage', ask = FALSE, update = FALSE)"
