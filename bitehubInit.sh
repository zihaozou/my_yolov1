apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
python newTrain.py -c True -n 4 -b 32 -e 10 -l 1e-4 -w base.pth -ci /data/zihaozou/voc/ -co /data/ -g True