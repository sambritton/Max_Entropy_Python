Bootstrap: docker
From: pytorch/pytorch

%post

    # Update list of available packages, then upgrade them
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
    
    # Utility and support packages
    apt-get install -y screen terminator tmux vim wget 
    apt-get install -y aptitude build-essential cmake g++ gfortran git \
        pkg-config software-properties-common
    apt-get install -y unrar
    apt-get install -y ffmpeg
    apt-get install -y gnuplot-x11
	
	apt-get install -y vim
	pip install pandas
	pip install scipy
    
	# Clean up
    apt-get -y autoremove
    rm -rvf /var/lib/apt/lists/*