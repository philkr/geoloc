from train import CoordParser

def main():
	from argparse import ArgumentParser
	from time import time
	from mpl_toolkits.basemap import Basemap
	import numpy as np
	
	parser = ArgumentParser()
	parser.add_argument('--file-list', type=str, default='/fastdata/finder/streetview_train.txt', help='path to the streetview training file')
	parser.add_argument('-n', '--n-clusters', type=int, default=100, help='number of cluster')
	parser.add_argument('--max-files', type=int, help='maximum number of files to cluster')
	parser.add_argument('output', type=str, help='output file (e.g. clusters.npy)')
	args = parser.parse_args()
	
	# Parse the coordinates
	parser = CoordParser()
	c = np.array([parser(l) for l in open(args.file_list,'r')])

	# Create the basemap parameters
	bnd = 0
	basemap_params = dict(projection='merc',llcrnrlat=np.min(c[:,0])-bnd,urcrnrlat=np.max(c[:,0])+bnd, llcrnrlon=np.min(c[:,1])-bnd,urcrnrlon=np.max(c[:,1])+bnd)
	
	# Select a subset of the coordinates to cluster
	np.random.shuffle(c)
	if args.max_files is not None:
		c = c[:args.max_files]
	
	# Project the coordinates into x, y coordinates
	m = Basemap(basemap_params)
	x,y = m(c[:,1],c[:,0])
	
	from sklearn import cluster
	km = cluster.MiniBatchKMeans(n_clusters=args.n_clusters).fit(np.concatenate((x[:,None],y[:,None]),axis=1))
	
	np.save(args.output,(basemap_params,km.cluster_centers_))

if __name__ == "__main__":
	main()
