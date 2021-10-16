# Copy sintel from raid array to local disk for faster training
mkdir /playpen-nvme/tgreer/Sintel
cp /playpen-raid1/Data/MPI-Sintel-complete.zip /playpen-nvme/tgreer/Sintel

cd /playpen-nvme/tgreer/Sintel
unzip MPI-Sintel-complete.zip
