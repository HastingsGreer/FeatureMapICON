mkdir data_storage
cd data_storage

if stat ~/.davis.zip 
    then
        echo "used cached dataset"
        cp ~/.davis.zip DAVIS-2017-trainval-480p.zip
    else
        echo "got new copy"
        ls -la ~
        wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip 
        cp DAVIS-2017-trainval-480p.zip ~/.davis.zip
fi

unzip -q DAVIS-2017-trainval-480p.zip


