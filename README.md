# Fully convolutional geolocation
This is a tensorflow implementation of the fully convolutional geolocation.
I trained and evaluated this on streetview data.

## Dependencies
 * python (tested with python3)
 * numpy
 * sklearn (`aptitude install python3-sklearn`)
 * matplotlib + basemap (`aptitude install python3-matplotlib python3-mpltoolkits.basemap`)
 * tensorflow v0.8+ (http://www.tensorflow.org/)

## Training
### Data setup
First you'll need to setup your data.
Put all the files you want to train on in a single directory and resize them to an appropriate size.
Let's call the newly created data directory `DATA_DIR` (e.g. `/fastdata/finder/`) and the finder image location `INPUT_DIR` (e.g. `/media/philkr/Elements/`).
To setup the finder streetview data use (note this will take about a day and use up 500G of disk space):
```bash
I=$INPUT_DIR/TaiwanStreetView/imgs/
O=$DATA_DIR/streetview/
mkdir $O
for j in $I/*/; do
  for i in $j/*.jpg; do
    if [ ! -f $O/$(basename $i) ]; then 
      convert $i -resize 320x200 -quality 99 $O/$(basename $i);
    fi
  done
done
```
for flickr use (this doesn't store all the image for some reason):
```bash
I=$INPUT_DIR/TaiwanFlickr/
O=$DATA_DIR/flickr/
for D in $I/meta/*/; do
  DD=${D/meta/imgs}
  for i in $D/*; do
    IM_N=$DD/$(cut -d , -f 1 $i | tail -n1 | sed -e 's/"//g')_z.jpg
    OUT_N=$(basename ${i/-meta.csv/_img.jpg})
    if [ -e $IM_N ]; then
      convert $IM_N -resize 320x320 -quality 99 $O/$OUT_N;
    fi
  done
done
```
Once this is complete we should split the dataset into training and testing.
```bash
N_TEST=65536
ls $DATA_DIR/streetview/ > $DATA_DIR/streetview.txt
shuf $DATA_DIR/streetview.txt > $DATA_DIR/streetview_shuf.txt
head -n -$N_TEST $DATA_DIR/streetview_shuf.txt > $DATA_DIR/streetview_train.txt
tail -n $N_TEST $DATA_DIR/streetview_shuf.txt > $DATA_DIR/streetview_test.txt
```
### Cluster setup
With this all the files are setup. We can now start by clustering the coordinates
```bash
python3 src/cluster.py -n 100 --file-list $DATA_DIR/streetview_train.txt $DATA_DIR/clusters.npy
```
### Training
To train a model use
```bash
train.py --clusters $DATA_DIR/clusters.npy --file-list $DATA_DIR/streetview_train.txt --file-base-dir $DATA_DIR/streetview/
```
Optional arguments:
 * `--initial-weights path/to/VGG16.caffemodel.h5` (can be downloaded from TODO)
 * `--num-gpu -1` for multi-gpu training

Then the only thing left to do is wait for a few days and monitor the training in tensorboard.