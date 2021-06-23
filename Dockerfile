FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1:latest
COPY Anaconda3-2021.05-Linux-x86_64.sh /root/
RUN bash /root/Anaconda3-2021.05-Linux-x86_64.sh -b -p &&\
    export PATH="/root/anaconda3/bin/:"$PATH &&\
    conda init &&\
    conda create -n p2 python=3.7.10
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate p2" >> ~/.bashrc
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html &&\
    conda install -c psi4 gcc-5 &&\
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-geometric &&\
    pip install nltk &&\
    pip install pytorch_pretrained_bert &&\
    pip install tensorboardX &&\
    conda list