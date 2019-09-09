FROM pytorch/pytorch
FROM pytorch/pytorch
RUN # Update list of available packages, then upgrade them
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
RUN # Utility and support packages
RUN apt-get install -y screen terminator tmux vim wget
RUN apt-get install -y aptitude build-essential cmake g++ gfortran git \
pkg-config software-properties-common
RUN apt-get install -y unrar
RUN apt-get install -y ffmpeg
RUN apt-get install -y gnuplot-x11
RUN apt-get install -y vim
RUN pip install pandas
RUN pip install scipy
RUN # Clean up
RUN apt-get -y autoremove
RUN rm -rvf /var/lib/apt/lists/*
CMD exec /bin/bash "$@"
