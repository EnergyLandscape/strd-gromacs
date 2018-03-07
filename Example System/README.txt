
Follow the standard Gromacs build instructions to create a STRD Gromacs executable. The following build options are recommended to distinguish STRD Gromacs from any other build you may have on your system:

	-DGMX_DEFAULT_SUFFIX=OFF
	-DGMX_BINARY_SUFFIX=_strd
	-DGMX_LIBS_SUFFIX=_strd

After compiling STRD Gromacs, prepare the simulation with grompp:

gmx_strd grompp -f run.mdp \
	-c strd_popc_20nm.gro \
	-p strd_popc_20nm.top \
	-n strd_popc_20nm.ndx \
	-o ./out/run

Then start mdrun:

gmx_strd mdrun -deffnm ./out/run -dlb no

A sample job submission script is provided.
