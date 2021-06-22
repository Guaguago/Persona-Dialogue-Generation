FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1:latest

COPY Anaconda3-2021.05-Linux-x86_64.sh /root/
RUN wget http://100.97.11.243:10086/proxy/enable_internet_proxy.sh &&\
    bash enable_internet_proxy.sh

RUN bash /root/Anaconda3-2021.05-Linux-x86_64.sh -b -p &&\
    export PATH="/root/anaconda3/bin/:"$PATH &&\
    export http_proxy="http://9.131.211.13:3128" &&\
    export https_proxy="http://9.131.211.13:3128" &&\
    export no_proxy="tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,mirrors-tlinux.tencentyun.com,.oa.com,.local" &&\
    conda init &&\
    conda create -n p2 python=3.7.10

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate p2" >> ~/.bashrc

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    conda install -c psi4 gcc-5 && \
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html && \
    pip install torch-geometric && \
    pip install nltk && \
    conda list