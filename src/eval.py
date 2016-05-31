from train import *

def youngest(files):
	from os import path
	ts_files = sorted([(path.getmtime(f),f) for f in files if path.exists(f)])
	if not ts_files:
		return None
	return ts_files[-1][1]
	

def main():
	from argparse import ArgumentParser
	from glob import glob
	from time import time
	import random
	import tensorflow as tf

	parser = ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('-n', type=int, default=None, help='Number of images to evaluate on')
	parser.add_argument('--file-list', type=str, default='/fastdata/finder/streetview_test.txt', help='path to the streetview test file')
	parser.add_argument('--file-base-dir', type=str, default='/fastdata/finder/streetview/', help='directory of the test images')
	parser.add_argument('train_dir', help='training directory')
	args = parser.parse_args()

	cluster_file = args.train_dir+'clusters.npy'
	files = [os.path.join(args.file_base_dir,l.strip()) for l in open(args.file_list,'r')]

	random.shuffle(files)
	if args.n:
		files = files[:args.n]
	args.n = len(files)
	
	# Setup the graph
	data,gt = glocData(files, cluster_file, batch_size=args.batch_size)
	n_clusters = getNCluster(cluster_file)

	vgg = vgg16(data, n_out=n_clusters)
	avg_vgg = tf.reduce_mean(tf.reduce_mean(vgg,1),1)
	
	pred = tf.argmax(avg_vgg,dimension=1)

	# Initialize ops
	saver = tf.train.Saver(tf.all_variables())
	tf.get_default_graph().finalize()

	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))) as sess:
		# Initialize stuff
		tf.train.start_queue_runners(sess=sess)
		snap = youngest([f for f in glob(os.path.join(args.train_dir, 'final.ckpt*')) + glob(os.path.join(args.train_dir, 'snap.ckpt*')) if '.meta' not in f])
		saver.restore(sess, snap)
		
		# Eval
		top1_acc, top5_acc = [], []
		id = []
		for it in range(0,args.n,args.batch_size):
			t0 = time()
			gt_value, pred_value, score = sess.run([gt, pred, avg_vgg])
			t1 = time()
			
			for g, p, s in zip(gt_value, pred_value, score):
				top1_acc.append( g==p )
				ss = sorted( s )
				top5_acc.append( ss[-5] <= s[g] )
				id.append( np.where(s[g] == ss[::-1])[0] )
			if it % (10*args.batch_size) == 0:
				print('%8d, top1 = %0.2f    top5 = %0.2f  (%0.1f im/sec)'%(it, np.mean(top1_acc), np.mean(top5_acc), args.batch_size / (t1-t0)))
		print('%8d, top1 = %0.2f    top5 = %0.2f  (%0.1f im/sec)'%(args.n, np.mean(top1_acc), np.mean(top5_acc), args.batch_size / (t1-t0)))
		print( [np.mean(np.array(id) <= r) for r in range(100)] )

if __name__ == "__main__":
	main()