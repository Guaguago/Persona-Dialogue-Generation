FROM mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1:latest
COPY Anaconda3-2021.05-Linux-x86_64.sh /root/
RUN bash /root/Anaconda3-2021.05-Linux-x86_64.sh -b -p
COPY p2-env.zip /root/anaconda3/envs
RUN cd /root/anaconda3/envs/ &&\
    unzip p2-env.zip
RUN export PATH="/root/anaconda3/bin/:"$PATH &&\
    conda init &&\
    conda create --name p2 --clone p2-env
RUN rm -rf /root/anaconda3/envs/p2-env &&\
    rm /root/anaconda3/envs/p2-env.zip
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate p2" >> ~/.bashrc



