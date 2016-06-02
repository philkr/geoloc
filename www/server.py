try:
	from http.server import SimpleHTTPRequestHandler
	import socketserver
except:
	from SimpleHTTPServer import SimpleHTTPRequestHandler
	import SocketServer as socketserver
import matplotlib as mpl
mpl.use('agg')

from PIL import Image
import sys, os, inspect
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(current_dir,'..','src'))
from eval import *

def pm(m):
	mm = m - np.max(m)
	r = np.exp(mm)
	return r / np.sum(r)

class GeoLoc:
	parser = CoordParser()
	def _load_cluster(self, cluster_file):
		from mpl_toolkits.basemap import Basemap
		from sklearn import cluster
		bm_param, km_param = np.load(cluster_file)
		self.m = Basemap(resolution='h', **bm_param)
		self.km = cluster.MiniBatchKMeans(n_clusters=km_param.shape[0])
		self.km.cluster_centers_ = km_param
	
	def __init__(self, model_dir, files, use_gpu=False):
		import tensorflow as tf
		from glob import glob
		self.files = files
		
		cluster_file = model_dir+'clusters.npy'
		self.n_clusters = getNCluster(cluster_file)
		self._load_cluster(cluster_file)
		
		if not use_gpu:
			d = tf.device('/cpu:0')
		data = tf.placeholder(tf.uint8, shape=(None,None,3))
		w_data = tf.expand_dims(tf.cast(data,tf.float32),0) - tf.constant([123,117,103],dtype=tf.float32,shape=(1,1,3))
		vgg = vgg16(w_data, n_out=self.n_clusters)
		avg_vgg = tf.reduce_mean(tf.reduce_mean(vgg,1),1)
		pred = tf.argmax(avg_vgg,dimension=1)
		self.vgg, self.pred, self.data = vgg, pred, data
		if not use_gpu:
			del d
		
		saver = tf.train.Saver(tf.all_variables())
		tf.get_default_graph().finalize()
		
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)))
		snap = youngest([f for f in glob(os.path.join(model_dir, 'final.ckpt*')) + glob(os.path.join(model_dir, 'snap.ckpt*')) if '.meta' not in f])
		saver.restore(self.sess, snap)
	
	def drawMap(self, gt_ll=None, pr_ll=None):
		from pylab import scatter, subplots
		f,ax = subplots(figsize=(5,8))
		ax.axis('off')
		f.patch.set_facecolor('white')
		self.m.drawcoastlines()
		self.m.drawcountries()
		if gt_ll is not None:
			scatter( *self.m(*gt_ll[::-1]), color='b', s=100 )
		if pr_ll is not None:
			scatter( *self.m(*pr_ll[::-1]), color='r', s=100, marker='x' )
		f.tight_layout(pad=0)
		f.canvas.draw()
		buf = np.fromstring( f.canvas.tostring_rgb(), dtype=np.uint8 )
		w,h = f.canvas.get_width_height()
		buf.shape = (h,w,3)
		return buf
	
	def drawHeat(self, m, cmap=None):
		from pylab import scatter, figure, subplots
		f = figure(figsize=(6,4))
		ax = f.add_axes([0,0,1,1])
		ax.axis('off')
		f.patch.set_facecolor('white')
		ax.imshow(m, cmap=cmap)
		#f.tight_layout(pad=0)
		f.canvas.draw()
		buf = np.fromstring( f.canvas.tostring_rgb(), dtype=np.uint8 )
		w,h = f.canvas.get_width_height()
		buf.shape = (h,w,3)
		return buf
	
	def predict(self, im):
		pred_value, vgg_value = self.sess.run([self.pred,self.vgg], {self.data:im})
		pred_value = int(pred_value[0])
		m = pm(0.05*vgg_value[0,:,:,pred_value])
		return pred_value, m
	
	def __getitem__(self, i):
		f = self.files[i]
		ln,lt = self.parser(f)
		print( 'loading ', f )
		return np.asarray(Image.open(f)), ln, lt
	
	def __len__(self):
		return len(self.files)
	
	def __call__(self, im):
		pass
	
	def cToLL(self, c):
		return self.m(*self.km.cluster_centers_[c], inverse=True)[::-1]
	
	def llToC(self, ln, lt):
		return np.int32(self.km.predict(np.array(self.m(lt,ln))[None])[0])


def haversine(lon1, lat1, lon2, lat2):
	from math import radians, cos, sin, asin, sqrt
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # Radius of earth in kilometers. Use 3956 for miles
	return c * r

class GeoLocRequest(SimpleHTTPRequestHandler):
	def __init__(self, gl, *args, **kwargs):
		self.gl = gl
		super().__init__(*args, **kwargs)
	
	def translate_path(self, path):
		"""Translate a /-separated PATH to the local filename syntax.

		Components that mean special things to the local file system
		(e.g. drive or directory names) are ignored.  (XXX They should
		probably be diagnosed.)

		"""
		words = path.split('/')
		words = filter(None, words)
		path = current_dir
		for word in words:
			drive, word = os.path.splitdrive(word)
			head, word = os.path.split(word)
			if word in (os.curdir, os.pardir): continue
			path = os.path.join(path, word)
		return path
	
	def do_GET(self):
		try:
			from urlparse import urlparse, parse_qs
		except:
			from urllib.parse import urlparse, parse_qs
		# Parse the request
		path = urlparse(self.path)
		self.url = parse_qs(path.query)
		file_path = self.translate_path(self.path)
		if os.path.isdir(file_path):
			for index in "index.html", "index.htm":
				index = os.path.join(file_path, index)
				if os.path.exists(index):
					file_path = index
		ctype = self.guess_type(file_path)
		try:
			f = open(file_path, 'rb')
		except IOError:
			self.send_error(404, "File not found")
			return None
		
		page_str = f.read()
		if ctype == 'text/html':
			page_str = self.replace_str(page_str)
		
		self.send_response(200)
		self.send_header("Content-type", ctype)
		self.send_header("Content-Length", str(len(page_str)))
		self.send_header("Last-Modified", self.date_time_string())
		self.end_headers()
		self.wfile.write(page_str)
	
	
	def encodeImage(self, im):
		from io import BytesIO
		import base64
		
		buf = BytesIO()
		Image.fromarray(im).save(buf, format="PNG")
		imgStr = base64.b64encode(buf.getvalue())
		
		return b'data:image/png;base64,%s'%imgStr
	
	
	def replace_str(self, s):
		import random
		
		id = random.randint(0,len(self.gl)-1)
		if 'id' in self.url:
			id = int(self.url['id'][0])
		
		im, ln, lt = self.gl[id]
		
		s = s.replace(b'<!--ID-->', b'%d'%id)
		s = s.replace(b'<!--IM_LON-->', b'%0.3f'%ln)
		s = s.replace(b'<!--IM_LAT-->', b'%0.3f'%lt)
		s = s.replace(b'<!--IM_C-->', b'%d'%self.gl.llToC(ln,lt))
		s = s.replace(b'<!--IM_DIST-->', b'%0.3f km'%haversine(*self.gl.cToLL( self.gl.llToC(ln,lt) ), ln, lt))
		
		# Predict the cluster
		p_c, m = self.gl.predict(im)
		hm = self.gl.drawHeat(m, 'Blues')
		
		p_ln, p_lt = self.gl.cToLL( p_c )
		mp = self.gl.drawMap((ln,lt), (p_ln, p_lt))
		
		s = s.replace(b'<!--PR_LON-->', b'%0.3f'%p_ln)
		s = s.replace(b'<!--PR_LAT-->', b'%0.3f'%p_lt)
		s = s.replace(b'<!--PR_C-->', b'%d'%p_c)
		
		s = s.replace(b'<!--PR_DIST-->', b'%0.3f km'%haversine(p_ln, p_lt, ln, lt))
		
		s = s.replace(b'<!--IM-->', b'<img class="im" src="%s"/>'%self.encodeImage(im))
		s = s.replace(b'<!--MAP-->', b'<img class="map" src="%s"/>'%self.encodeImage(mp))
		s = s.replace(b'<!--HEAT-->', b'<img class="heat" src="%s"/>'%self.encodeImage(hm))
		return s
	
	def copyfile(self, source, outputfile):
		page_str = source.read()
		page_str = self.replace_str(page_str)
		outputfile.write( page_str )

def main():
	from argparse import ArgumentParser
	from time import time
	
	parser = ArgumentParser()
	parser.add_argument('--file-list', type=str, default='/fastdata/finder/streetview_test.txt', help='path to the streetview test file')
	parser.add_argument('--file-base-dir', type=str, default='/fastdata/finder/streetview/', help='directory of the test images')
	parser.add_argument('-p','--port', type=int, default=8000, help='port number')
	parser.add_argument('-g','--use-gpu', action='store_true', help='use the gpu')
	parser.add_argument('train_dir', help='training directory')
	args = parser.parse_args()
	
	files = [os.path.join(args.file_base_dir,l.strip()) for l in open(args.file_list,'r')]
	
	gl = GeoLoc(args.train_dir, files, use_gpu=args.use_gpu)
	
	socketserver.TCPServer.allow_reuse_address = True
	httpd = socketserver.TCPServer(("", args.port), lambda *a, **k: GeoLocRequest(gl,*a,**k))
	httpd.serve_forever()
	

if __name__ == "__main__":
	main()
