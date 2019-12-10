
# srun -n 20 -p veryshort -A COMS30005 --reservation COMS30005 ./stencil 1024 1024 100 >> test.txt
import os
import argparse
import re
import csv

parser = argparse.ArgumentParser(description = 'get timings for MPI')
parser.add_argument('min', type=int, help='min no of cores')
parser.add_argument('max', type=int, help='max no of cores')
parser.add_argument('dim',type=int, help='dimensions of stencil file you want to generate')
parser.add_argument('output_file',help = 'csv file to store output')
args = parser.parse_args()
csv_file = csv.writer(open(args.output_file,"a"),delimiter = ',')
csv_file.writerow(['no of cores','r1','r2','r3','r_avg'])
for i in range(args.min,args.max+1):
    cmd = 'srun -n %d -p veryshort -A COMS30005 --reservation COMS30005 ./stencil %d %d 100 > temp.txt'
    print("no of cores " + str(i))
    cmd = (cmd % (i,args.dim,args.dim))
    runtimes = []
    for j in range(3):
        os.system(cmd)
	#parse string to get runtime
	f = open('temp.txt',"r")
	lines = f.readlines()
	x = float(lines[1][10:18])
	runtimes.append(x)
    avg = sum(runtimes)/len(runtimes)
    runtimes.append(avg)
    csv_file.writerow([i] + runtimes)

