mkdir data_storage
cd data_storage

if [ -f "~/.davis.zip" ]
    then
        cp ~/.davis.zip DAVIS-2017-trainval-480p.zip
    else
        wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip 
        cp DAVIS-2017-trainval-480p.zip ~/.davis.zip
fi

unzip -q DAVIS-2017-trainval-480p.zip

