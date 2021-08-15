FROM 10.11.3.8:5000/user-images/my_py_env_zihaozou
RUN pip install jupyter
RUN apt-get install -y openssh-server