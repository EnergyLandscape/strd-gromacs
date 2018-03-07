#!/usr/bin/env python

import sys, getopt, random

gro_filename = ''
out_filename = 'solvated.gro'
srdDensity = 0.0
example = 'EXAMPLE: add_srd_uniform.py -f <input gro name> -d <SRD number density>'

try:
	opts, args = getopt.getopt(sys.argv[1:],"hf:d:",["density="])
except getopt.GetoptError:
	print 'ERROR: invalid command line argument'
	print example
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print 'Required arguments:'
		print example
		sys.exit()
	elif opt in ("-f"):
		gro_filename = arg
		if gro_filename == out_filename:
			print("Input filename not allowed: %s" % out_filename)
			sys.exit(2)
	elif opt in ("-d", "--density"):
		srdDensity = float(arg)
		if srdDensity > 50 or srdDensity < 0:
			print 'Invalid SRD density.'
			sys.exit(2)

if gro_filename == '':
	print example
	sys.exit(2)

try:
	gro_in = open(gro_filename,"r")
except IOError as e:
	print("Couldn't open gro input file (%s)." % e)

gro_out = open(out_filename, "w+")

title = gro_in.readline()
natoms = int(gro_in.readline())

# count water atoms and get box size
numWater = 0
for i in xrange(natoms):
	line = gro_in.readline()
	resname = line[5:10]
	if resname == "Water":
		numWater += 1

box = gro_in.readline()
sx, sy, sz = float(box[0:10]), float(box[10:20]), float(box[20:30])

numSRD = int(sx * sy * sz * srdDensity)
lastResID = 0
nextResID = 1
nextAtomID = 1

# remove water
gro_in.seek(0)
gro_out.write(gro_in.readline())
gro_out.write("%5d\n"%(int(gro_in.readline()) + numSRD - numWater))
for i in xrange(natoms):
	line = gro_in.readline()
	resname = line[5:10]
	if resname == "Water":
		continue
	
	nextAtomID += 1
	if nextAtomID >= 100000:
		nextAtomID = 1
		
	resID = int(line[0:5])
	if lastResID != resID:
		lastResID = resID
		nextResID += 1
		if nextResID >= 100000:
			nextResID = 1

	gro_out.write(line)
	
# add SRD
random.seed()
for i in xrange(numSRD):
	gro_out.write(
		"%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(
		nextResID, "SOL", "SRD", nextAtomID,
		random.uniform(0, sx), random.uniform(0, sy), random.uniform(0, sz)))
	nextAtomID += 1
	if nextAtomID >= 100000:
		nextAtomID = 1
	nextResID += 1
	if nextResID >= 100000:
		nextResID = 1
	
gro_out.write(box)

gro_in.close()
gro_out.close()

print "Removed", numWater, "water particles."
print "Added", numSRD, "SRD particles."
print "Don't forget to update your topology with 'SOL   %d'" % numSRD
