#!/bin/sh

# Adapt this part of script for your computer cluster.

NNODES=XXX

# Choose a 2D domain decomposition for NNODES
# This is optional, but will provide the best performance for bilayer systems.

DDZ=1
if ((DDZ >= $NNODES || $NNODES % DDZ)); then
	echo "STRD Error: Cannot make domain decomposition for $N nodes with $DDZ cells along z."
	exit 1
fi

let M=$NNODES/DDZ
C=$(echo "sqrt ( $M ) / 1" | bc)

let DDY=1
for (( K=2; K<=$C; K++ )); do
	if ! ((M % K)); then
		let DDY=K
	fi
done
let DDX=M/DDY

# Run STRD Gromacs with chosen DD grid
# This must also be adapted for your computer cluster.

mpirun \
	-np $NNODES \
	PATH_TO_MDRUN \
	-dlb no -dd $DDX $DDY $DDZ \
	OTHER_MDRUN_OPTIONS # -deffnm XXX, etc
