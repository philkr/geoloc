import tensorflow as tf
import os, numpy as np, random


def vgg16(inputs, n_out=100):
	from slim import slim
	with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
		net = slim.ops.repeat_op(2, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
		net = slim.ops.max_pool(net, [2, 2], scope='pool1')
		net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 128, [3, 3], scope='conv2')
		net = slim.ops.max_pool(net, [2, 2], scope='pool2')
		net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
		net = slim.ops.max_pool(net, [2, 2], scope='pool3')
		net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv4')
		net = slim.ops.max_pool(net, [2, 2], scope='pool4')
		net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv5')
		# Add a 1x1 conv as a classifier
		net = slim.ops.conv2d(net, n_out, [1,1], scope='fc', activation=None)
	return net


def imageInput(files):
	filename_queue = tf.train.string_input_producer(files,shuffle=True)
	reader = tf.WholeFileReader()
	name, value = reader.read(filename_queue)
	return name, tf.image.decode_jpeg(value, channels=3)


def parseCoord(filename):
	import re
	r = re.compile('GSV_640_400_(.*)_(.*)_(.*)_(.*)_(.*).jpg')
	def _parser(fn):
		m = r.search(fn.decode())
		if m:
			return np.float32(m.group(1)),np.float32(m.group(2))
		return np.float32(0),np.float32(0)
	lt,ln = tf.py_func(_parser, [filename], [tf.float32, tf.float32])
	lt.set_shape(())
	ln.set_shape(())
	return lt,ln


def coordToCluster(lat,lon,cluster_file):
	# Setup the basemap and cluster
	from mpl_toolkits.basemap import Basemap
	from sklearn import cluster
	bm_param, km_param = np.load(cluster_file)
	m = Basemap(**bm_param)
	km = cluster.MiniBatchKMeans(n_clusters=km_param.shape[0])
	km.cluster_centers_ = km_param
	# Cluster function
	def _cluster(lt,ln):
		return np.int32(km.predict(np.array(m(ln,lt))[None])[0]),
	r = tf.py_func(_cluster, [lat,lon], [tf.int32])[0]
	r.set_shape(())
	return r


def glocData(files, cluster_file, batch_size=32):
    fn, data = imageInput(files)
    w_data = tf.cast(data,tf.float32) - tf.constant([123,117,103],dtype=tf.float32,shape=(1,1,3))
    ln,lt = parseCoord(fn)
    gt = coordToCluster(ln,lt, cluster_file=cluster_file)
    if batch_size > 1:
        return tf.train.batch([w_data,gt], batch_size=batch_size, capacity=10*batch_size, dynamic_pad=True)
    return tf.expand_dims(w_data,0), tf.expand_dims(gt,0)

def main():
	from argparse import ArgumentParser
	from time import time

	parser = ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--nit', type=int, default=1000000, help='number of iterations')
	parser.add_argument('--log_device_placement', action='store_true')
	parser.add_argument('train_dir', help='training directory')
	args = parser.parse_args()

	try: os.makedirs(args.train_dir)
	except: pass

	FINDER_DIR='/fastdata/finder/'
	IMAGE_PATH=FINDER_DIR+'streetview/'
	files = [IMAGE_PATH+l.strip() for l in open(FINDER_DIR+'/images.txt','r')]

	# Setup the graph
	data,gt = glocData(files, FINDER_DIR+'cluster.npy', batch_size=args.batch_size)

	vgg = vgg16(data)
	avg_vgg = tf.reduce_mean(tf.reduce_mean(vgg,1),1)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(avg_vgg, gt))
	# solver = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9)
	solver = tf.train.AdamOptimizer(learning_rate=args.lr)
	grads_and_vars = solver.compute_gradients(loss)
	solver_step = solver.apply_gradients(grads_and_vars)

	# Create some summaries
	loss_avg = tf.train.ExponentialMovingAverage(0.9, name='avg')
	tf.scalar_summary('loss', loss)
	loss_avg_op = loss_avg.apply([loss])
	tf.scalar_summary('loss(avg)', loss_avg.average(loss))
	with tf.control_dependencies([loss_avg_op]):
		loss = tf.identity(loss)


	for grad,var in grads_and_vars:
	# 	tf.histogram_summary(var.op.name+'/activations', var)
	# 	tf.histogram_summary(var.op.name+'/sparsity', tf.nn.zero_fraction(var))
		tf.scalar_summary(var.op.name+'/norm', tf.reduce_mean(var*var))
		tf.scalar_summary(var.op.name+'/gradient_norm', tf.reduce_mean(grad*grad))
		tf.scalar_summary(var.op.name+'/gradient_ratio', tf.reduce_mean(grad*grad) / tf.reduce_mean(var*var))
	summary_op = tf.merge_all_summaries()

	# Initialize ops
	saver = tf.train.Saver(tf.all_variables())
	init_op = tf.initialize_all_variables()
	from slim import load
	load_op = load.loadH5('/data/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel.h5')

	tf.get_default_graph().finalize()

	with tf.Session(config=tf.ConfigProto(log_device_placement=args.log_device_placement, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))) as sess:
		# Initialize stuff
		summary_writer = tf.train.SummaryWriter(args.train_dir, sess.graph)
		tf.train.start_queue_runners(sess=sess)
		sess.run(init_op)
		sess.run(load_op)
	
		# Train
		loss_values = []
		for it in range(args.nit):
			t0 = time()
			_, loss_value = sess.run([solver_step, loss])
			t1 = time()
			loss_values.append( loss_value )
		
			if it % 10 == 0:
				print('%8d, loss = %0.2f [%0.2f] (%0.1f im/sec)'%(it, loss_value, np.mean(loss_values), args.batch_size / (t1-t0)))
				loss_values = loss_values[-20:]
			if it % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, it)
			if it % 1000 == 0:
				saver.save(sess, os.path.join(args.train_dir, 'snap.ckpt'), global_step=it)
		saver.save(sess, os.path.join(args.train_dir, 'final.ckpt'))

if __name__ == "__main__":
	main()
