unzip nips_task34.zip -d nips_task34
unzip assist2009.zip -d assist2009
unzip peiyou.zip -d peiyou
cd ../examples
python data_preprocess.py -d assist2009
python data_preprocess.py -d nips_task34
python data_preprocess.py -d peiyou