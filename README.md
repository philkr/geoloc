# Fully convolutional geolocation
This is a tensorflow implementation of the fully convolutional geolocation.
I trained and evaluated this on streetview data.

## Dependencies
 * python (tested with python3)
 * numpy
 * sklearn (`aptitude install python3-sklearn`)
 * matplotlib + basemap (`aptitude install python3-matplotlib python3-mpltoolkits.basemap`)
 * tensorflow v0.8+ (http://www.tensorflow.org/)

Tested on Ubuntu 16.04 and Mac OS X 10.10. Windows is not recommended.

## Evaluation
### Data setup
Follow the training data setup detailed below, or download just the test data from [here](https://drive.google.com/folderview?id=0B9Rfwa3xKC_rWDVDZGZmWFJNbDA) and store it in `$DATA_DIR`.

### Models
Feel free to train your own geolocation models (as shown below), or simply download the pre-trained ones from [here](https://drive.google.com/folderview?id=0B9Rfwa3xKC_rWDVDZGZmWFJNbDA).

### Running the evaluation
```bash
python3 src/eval.py --file-list $DATA_DIR/streetview_test.txt --file-base-dir $DATA_DIR/streetview/ path/to/model/
```

You should see an output as follows (for the 100 class model)
```
...
   13760, top1 = 0.50    top5 = 0.84  (102.5 im/sec)
...
   65534, top1 = 0.50    top5 = 0.84  (101.9 im/sec)
[0.4968719482421875, 0.6625518798828125, 0.74810791015625, 0.80029296875, 0.8364105224609375, 0.8622283935546875, 0.8816680908203125, 0.89739990234375, 0.910430908203125, 0.92047119140625, 0.92950439453125, 0.9364166259765625, 0.9422149658203125, 0.947357177734375, 0.952484130859375, 0.95635986328125, 0.9600982666015625, 0.9631500244140625, 0.96588134765625, 0.96826171875, 0.9705047607421875, 0.9724578857421875, 0.974365234375, 0.9763031005859375, 0.977813720703125, 0.9791412353515625, 0.9808197021484375, 0.9821319580078125, 0.983489990234375, 0.984466552734375, 0.9853973388671875, 0.9863433837890625, 0.9870758056640625, 0.988006591796875, 0.9886627197265625, 0.9892425537109375, 0.989776611328125, 0.9903106689453125, 0.990966796875, 0.991546630859375, 0.992095947265625, 0.99249267578125, 0.9929046630859375, 0.99334716796875, 0.9937591552734375, 0.99407958984375, 0.9943084716796875, 0.994659423828125, 0.9948883056640625, 0.9951629638671875, 0.9954986572265625, 0.9957275390625, 0.9959716796875, 0.996124267578125, 0.996307373046875, 0.9965667724609375, 0.9967803955078125, 0.99700927734375, 0.9973602294921875, 0.9975738525390625, 0.9977569580078125, 0.9980010986328125, 0.998077392578125, 0.99822998046875, 0.9982757568359375, 0.9984283447265625, 0.99859619140625, 0.9987335205078125, 0.9988250732421875, 0.9989166259765625, 0.9990234375, 0.9991302490234375, 0.999267578125, 0.9993438720703125, 0.999359130859375, 0.999420166015625, 0.999481201171875, 0.99951171875, 0.999542236328125, 0.99957275390625, 0.9996490478515625, 0.99969482421875, 0.9997100830078125, 0.999755859375, 0.9998016357421875, 0.9998321533203125, 0.999847412109375, 0.9998931884765625, 0.9998931884765625, 0.9999237060546875, 0.999969482421875, 0.999969482421875, 0.999969482421875, 0.9999847412109375, 0.9999847412109375, 0.9999847412109375, 0.9999847412109375, 0.9999847412109375, 0.9999847412109375, 1.0]
```
The list in the end measures the top-n accuracy for n from 1 to `n_clusters`.
## Web UI
Follow the data and model setup for evaluation.
### Running the web ui
```bash
python3 www/server.py --file-list $DATA_DIR/streetview_test.txt --file-base-dir $DATA_DIR/streetview/ path/to/model/ --use-gpu
```
Remove the `--use-gpu` flag if you want all computations to be on the CPU.

Open a browser on `localhost:8000`.

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
 * `--initial-weights path/to/VGG16.caffemodel.h5` (can be downloaded from [here](https://drive.google.com/folderview?id=0B9Rfwa3xKC_rWDVDZGZmWFJNbDA))
 * `--num-gpu -1` for multi-gpu training

Then the only thing left to do is wait for a few days and monitor the training in tensorboard.
