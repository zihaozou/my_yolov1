FROM nvidia/cuda:10.1-runtime-ubuntu18.04
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.9
RUN apt-get install -y python3.9-distutils
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg libgl1-mesa-glx
RUN pip install opencv-python
RUN pip install tensorboard