# Download bactria biotope corpus and format for model input.
# From https://sites.google.com/view/bb-2019/dataset

out_dir=data/bb
mkdir -p $out_dir

# Download and unzip.
mkdir $out_dir/BB-rel+ner
wget https://drive.google.com/uc?id=1fUz3hHJoiEQzdYjxN5nlfl3o9Gv98zz9&export=download -P $out_dir/BB-rel+ner
wget https://drive.google.com/uc?id=1gQ0mDwHyv79isLmFZ2DWHwe-BwaKiNm_&export=download -P $out_dir/BB-rel+ner
wget https://drive.google.com/uc?id=1W-a1gZoxUG1I-2ttfAr991WaznFSdWtk&export=download -P $out_dir/BB-rel+ner
unzip $out_dir/BB-rel+ner/BioNLP-OST-2019_BB-rel+ner_train.zip -d $out_dir/BB-rel+ner
unzip $out_dir/BB-rel+ner/BioNLP-OST-2019_BB-rel+ner_dev.zip -d $out_dir/BB-rel+ner
unzip $out_dir/BB-rel+ner/BioNLP-OST-2019_BB-rel+ner_test.zip -d $out_dir/BB-rel+ner

rm -r $out_dir/BB-rel+ner/*.zip

# merge all abstracts on each set
mkdir -p $out_dir/processed_data/merge_data
python scripts/data/bb/01_merge_file.py

# Run formatting
mkdir $out_dir/processed_data/json
# Format labelled files (training and development sets)
python scripts/data/bb/02_final_bb_to_input.py
# Format unlabelled file (test set)
python scripts/data/bb/04-test_set_process.py




