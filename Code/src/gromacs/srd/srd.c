/*
 *	This file is a part of STRD Martini, an efficient implementation
 *	of multi-particle collision dynamics for membrane simulations.
 *	Copyright (C) 2018 Andrew Zgorski
 *
 *	This library is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	This library is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *	Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with this library; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include "gromacs/srd/srd.h"
#include "mtop_util.h"
#include "sim_util.h"
#include "physics.h"
#include "copyrite.h"

#include "domdec.h"
#include "gromacs/timing/wallcycle.h"

#include "vec.h"
#include "macros.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

// experimental communication scheme
void init_srd_EXP(t_srd *srd, t_commrec *cr);

// experimental communication scheme
void update_srd_EXP(
		t_srd				*srd,
		gmx_int64_t			 step,
		t_state				*state_local,
		gmx_unused t_state	*state_global,
		t_mdatoms			*mdatoms,
		t_commrec			*cr,
		gmx_wallcycle_t		 wcycle);

void getCellIndicies(t_srd *srd, int cellID, int *ci, int *cj, int *ck)
{
	int cc = cellID;
	*ci = cc / (srd->cellCount[YY] * srd->cellCount[ZZ]);
	cc -= *ci * (srd->cellCount[YY] * srd->cellCount[ZZ]);
	*cj = cc / srd->cellCount[ZZ];
	*ck = cc - *cj * srd->cellCount[ZZ];
}

static void ERROR_LOST_PARTICLE(t_srd *srd, t_commrec *cr, t_state *state_local, int ai, int cellID)
{
	int ci, cj, ck;
	rvec xmin, xmax;

	getCellIndicies(srd, cellID, &ci, &cj, &ck);
	get_dd_bounds(cr, xmin, xmax);

	gmx_fatal(FARGS, "<Step %"GMX_PRId64"> [Node %d] SRD particle does not belong to a local collision cell!\n"
			"Grid shift is (%8.4f %8.4f %8.4f)\n"
			"Shifted position is (%8.4f %8.4f %8.4f), this process has (%8.4f - %8.4f) (%8.4f - %8.4f) (%8.4f - %8.4f)\n"
			"Assigned to %d (%d %d %d), this process covers (%d - %d) (%d - %d) (%d - %d)",
			srd->step, srd->nodeID,
			srd->gridShift[XX], srd->gridShift[YY], srd->gridShift[ZZ],
			state_local->x[ai][XX] - srd->gridShift[XX],
			state_local->x[ai][YY] - srd->gridShift[YY],
			state_local->x[ai][ZZ] - srd->gridShift[ZZ],
			xmin[XX] - srd->gridShift[XX],
			xmax[XX] - srd->gridShift[XX],
			xmin[YY] - srd->gridShift[YY],
			xmax[YY] - srd->gridShift[YY],
			xmin[ZZ] - srd->gridShift[ZZ],
			xmax[ZZ] - srd->gridShift[ZZ],
			cellID, ci, cj, ck,
			srd->localCells[XX][0], srd->localCells[XX][srd->localCellCount[XX] - 1],
			srd->localCells[YY][0], srd->localCells[YY][srd->localCellCount[YY] - 1],
			srd->localCells[ZZ][0], srd->localCells[ZZ][srd->localCellCount[ZZ] - 1]);
}

// Used for testing and debugging
static gmx_inline gmx_bool float_float_eq(float a, float b, float tolerance)
{
	return fabs(a - b) < tolerance;
}

// Used for testing and debugging
static gmx_inline gmx_bool rvec_eq(rvec a, rvec b, float tolerance)
{
	return	float_float_eq(a[XX], b[XX], tolerance) &&
			float_float_eq(a[YY], b[YY], tolerance) &&
			float_float_eq(a[ZZ], b[ZZ], tolerance);
}

static gmx_inline void clear_dmat(dmatrix a)
{
    const double nul = 0.0;

    a[XX][XX] = a[XX][YY] = a[XX][ZZ] = nul;
    a[YY][XX] = a[YY][YY] = a[YY][ZZ] = nul;
    a[ZZ][XX] = a[ZZ][YY] = a[ZZ][ZZ] = nul;
}

static gmx_inline double ddet(dmatrix a)
{
    return ( a[XX][XX]*(a[YY][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[YY][ZZ])
             -a[YY][XX]*(a[XX][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[XX][ZZ])
             +a[ZZ][XX]*(a[XX][YY]*a[YY][ZZ]-a[YY][YY]*a[XX][ZZ]));
}

static gmx_inline gmx_bool dm_inv(dmatrix src, dmatrix dest)
{
    const double smallnum = 1.0e-24;
    const double largenum = 1.0e24;
    double       deter, c, fc;

    deter = ddet(src);
    c     = 1.0 / deter;
    fc    = fabs(c);

    if ((fc <= smallnum) || (fc >= largenum))
    	return FALSE;

    dest[XX][XX] = c*(src[YY][YY]*src[ZZ][ZZ]-src[ZZ][YY]*src[YY][ZZ]);
    dest[XX][YY] = -c*(src[XX][YY]*src[ZZ][ZZ]-src[ZZ][YY]*src[XX][ZZ]);
    dest[XX][ZZ] = c*(src[XX][YY]*src[YY][ZZ]-src[YY][YY]*src[XX][ZZ]);
    dest[YY][XX] = -c*(src[YY][XX]*src[ZZ][ZZ]-src[ZZ][XX]*src[YY][ZZ]);
    dest[YY][YY] = c*(src[XX][XX]*src[ZZ][ZZ]-src[ZZ][XX]*src[XX][ZZ]);
    dest[YY][ZZ] = -c*(src[XX][XX]*src[YY][ZZ]-src[YY][XX]*src[XX][ZZ]);
    dest[ZZ][XX] = c*(src[YY][XX]*src[ZZ][YY]-src[ZZ][XX]*src[YY][YY]);
    dest[ZZ][YY] = -c*(src[XX][XX]*src[ZZ][YY]-src[ZZ][XX]*src[XX][YY]);
    dest[ZZ][ZZ] = c*(src[XX][XX]*src[YY][YY]-src[YY][XX]*src[XX][YY]);

    return TRUE;
}

static gmx_inline void dmvmul(dmatrix a, const rvec src, rvec dest)
{
    dest[XX] = a[XX][XX]*src[XX]+a[XX][YY]*src[YY]+a[XX][ZZ]*src[ZZ];
    dest[YY] = a[YY][XX]*src[XX]+a[YY][YY]*src[YY]+a[YY][ZZ]*src[ZZ];
    dest[ZZ] = a[ZZ][XX]*src[XX]+a[ZZ][YY]*src[YY]+a[ZZ][ZZ]*src[ZZ];
}

/*
 * Assigns cellIDs and cell-relative positions for local atoms.
 */
static void assign_atoms_to_cells(t_srd *srd, t_state *state_local)
{
	int d, ai;
	ivec cv;

	realloc_srd_int_buffer(srd, srd->cellID, srd->homenr);
	if(srd->bCommunicateX)
		realloc_srd_rvec_buffer(srd, srd->relXBuf, srd->homenr);

	for(ai = 0; ai < srd->homenr; ai++)
	{
		int globalIndex = (srd->gatindex == NULL) ? ai : srd->gatindex[ai];
		if (!srd->isSRD_gl[globalIndex])
		{
			srd->cellID->buf[ai] = -1;
			continue;
		}

		// Work in the frame of collision cell grid, which extends from 0 to boxSize
		// in each dimension. In this frame, atoms need to be shifted and may fall
		// outside the periodic box.
		for (d = 0; d < DIM; d++)
		{
			real boxSize = srd->boxSize[d];
			real shifted = state_local->x[ai][d] - srd->gridShift[d];
			while (shifted < 0.0)
				shifted += boxSize;
			while (shifted >= boxSize)
				shifted -= boxSize;

			// protect against corner cases:
			// * shifted = -0.0
			// * shifted / cellSize > numCells, ex: 9.999999 / 10.000000 => 1.0
			cv[d] = (int)floor(shifted / srd->cellSize[d]);
			if (cv[d] == srd->cellCount[d])
				cv[d] = 0;

			if (srd->bCommunicateX)
				srd->relXBuf->buf[ai][d] = shifted - (cv[d] * srd->cellSize[d]);
		}

		srd->cellID->buf[ai] = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * cv[XX]) + cv[YY])) + cv[ZZ];
	}
}

/**
 * Rotates a vector around an unit axis.
 */
static gmx_inline void unitaxis_angle_rotate(rvec vec, const rvec axis, const real sinAngle, const real cosAngle)
{
	rvec vpara, vperp, cross;

	real ip = iprod(vec, axis);
	vpara[XX] = axis[XX] * ip;
	vpara[YY] = axis[YY] * ip;
	vpara[ZZ] = axis[ZZ] * ip;

	vperp[XX] = vec[XX] - vpara[XX];
	vperp[YY] = vec[YY] - vpara[YY];
	vperp[ZZ] = vec[ZZ] - vpara[ZZ];

	cprod(axis, vec, cross);

	vec[XX] = vperp[XX] * cosAngle + cross[XX] * sinAngle + vpara[XX];
	vec[YY] = vperp[YY] * cosAngle + cross[YY] * sinAngle + vpara[YY];
	vec[ZZ] = vperp[ZZ] * cosAngle + cross[ZZ] * sinAngle + vpara[ZZ];
}

/**
 * Creates a spatially averaged sample of the cell occupancy and velocities,
 * adding it to the sum of other samples for the next output.
 */
static inline void do_output_sample(t_srd *srd)
{
	int ci, cj, ck;	// cell indices
	int i, j, k;	// offsets
	int si, sj, sk;	// shifted cell indices (with pbc)

	double weights[3][3][3];
	double weightSum;
	double var = srd->cellSize[XX] / 3;

	// precompute weights for nearest neighbors
	for(i = -1; i <= 1; i++)
	for(j = -1; j <= 1; j++)
	for(k = -1; k <= 1; k++)
	{
		double dx = i * srd->cellSize[XX] + srd->gridShift[XX];
		double dy = j * srd->cellSize[YY] + srd->gridShift[YY];
		double dz = k * srd->cellSize[ZZ] + srd->gridShift[ZZ];
		double dist2 = dx*dx + dy*dy + dz*dz;
		double weight = exp(-dist2 / (2*var*var));
		weights[i+1][j+1][k+1] = weight;
		weightSum += weight;
	}

	// normalize weights
	for(i = -1; i <= 1; i++)
	for(j = -1; j <= 1; j++)
	for(k = -1; k <= 1; k++)
	{
		weights[i+1][j+1][k+1] /= weightSum;
	}

	// iterate over all cells
	int cc = 0;
	for(ci = 0; ci < srd->cellCount[XX]; ci++)
	for(cj = 0; cj < srd->cellCount[YY]; cj++)
	for(ck = 0; ck < srd->cellCount[ZZ]; ck++)
	{
		real weightSum;

		// iterate over nearest neighbors
		for(i = -1; i <= 1; i++)
		{
			si = ci + i;
			if (ci + i < 0)
				si += srd->cellCount[XX];
			else if (ci + i >= srd->cellCount[XX])
				si -= srd->cellCount[XX];

			for(j = -1; j <= 1; j++)
			{
				sj = cj + j;
				if (cj + j < 0)
					sj += srd->cellCount[YY];
				else if (cj + j >= srd->cellCount[YY])
					sj -= srd->cellCount[YY];

				for(k = -1; k <= 1; k++)
				{
					sk = ck + k;
					if (ck + k < 0)
						sk += srd->cellCount[ZZ];
					else if (ck + k >= srd->cellCount[ZZ])
						sk -= srd->cellCount[ZZ];

					double weight = weights[i+1][j+1][k+1];
					int nc = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * si) + sj)) + sk;

					if(srd->writeCellVectors)
					{
						srd->outVelSum[cc][XX] += srd->outVelBuf[nc][XX] * weight;
						srd->outVelSum[cc][YY] += srd->outVelBuf[nc][YY] * weight;
						srd->outVelSum[cc][ZZ] += srd->outVelBuf[nc][ZZ] * weight;
					}

					if(srd->writeCellOccupancy)
						srd->outOccSum[cc] += srd->outOccBuf[nc] * weight;
				}
			}
		}
		cc++;
	}

	srd->outSizeSum[XX] += srd->cellSize[XX];
	srd->outSizeSum[YY] += srd->cellSize[YY];
	srd->outSizeSum[ZZ] += srd->cellSize[ZZ];
}

static inline void write_output_file_header(t_srd *srd)
{
	gmx_fio_write_int(srd->binaryOutputFile, 2016); // magic number, 0x7E0
	gmx_fio_write_int(srd->binaryOutputFile, 1);

	int vectorsSize = sizeof(rvec) * srd->numCells;
	int occupancySize = sizeof(int) * srd->numCells;
	gmx_fio_write_int(srd->binaryOutputFile, srd->writeCellOccupancy ? occupancySize : 0);
	gmx_fio_write_int(srd->binaryOutputFile, srd->writeCellVectors ? vectorsSize : 0);

	gmx_fio_write_int(srd->binaryOutputFile, srd->cellCount[XX]);
	gmx_fio_write_int(srd->binaryOutputFile, srd->cellCount[YY]);
	gmx_fio_write_int(srd->binaryOutputFile, srd->cellCount[ZZ]);

	srd->outFrameCountPos = gmx_fio_ftell(srd->binaryOutputFile);
	gmx_fio_write_int(srd->binaryOutputFile, 0);

	gmx_fio_write_int(srd->binaryOutputFile, srd->collisionFrequency);
	gmx_fio_write_int(srd->binaryOutputFile, srd->samplingFrequency);
	gmx_fio_write_int(srd->binaryOutputFile, srd->outputFrequency);
	gmx_fio_write_real(srd->binaryOutputFile, srd->outputFrequency * srd->delta_t);
}

static inline void write_frame_header(t_srd *srd)
{
	gmx_fio_write_int(srd->binaryOutputFile, srd->step);
	gmx_fio_write_real(srd->binaryOutputFile, srd->step * srd->delta_t);
	gmx_fio_write_real(srd->binaryOutputFile, srd->gridShift[XX]);
	gmx_fio_write_real(srd->binaryOutputFile, srd->gridShift[YY]);
	gmx_fio_write_real(srd->binaryOutputFile, srd->gridShift[ZZ]);
	gmx_fio_write_real(srd->binaryOutputFile, (real)srd->outSizeSum[XX]);
	gmx_fio_write_real(srd->binaryOutputFile, (real)srd->outSizeSum[YY]);
	gmx_fio_write_real(srd->binaryOutputFile, (real)srd->outSizeSum[ZZ]);
}

/**
 * Writes a frame to the output file and clears averaging sums.
 */
static void write_output_frame(t_srd *srd)
{
	int ci;
	double invNumSamples = (double)srd->samplingFrequency / srd->outputFrequency;

	srd->outFrameCount++;

	srd->outSizeSum[XX] *= invNumSamples;
	srd->outSizeSum[YY] *= invNumSamples;
	srd->outSizeSum[ZZ] *= invNumSamples;

	write_frame_header(srd);

	clear_dvec(srd->outSizeSum);

	if(srd->writeCellOccupancy)
	{
		for(ci = 0; ci < srd->numCells; ci++)
		{
			gmx_fio_write_real(srd->binaryOutputFile, (real)(srd->outOccSum[ci] * invNumSamples));
			srd->outOccSum[ci] = 0.0;
		}
	}

	if(srd->writeCellVectors)
	{
		for(ci = 0; ci < srd->numCells; ci++)
		{
			gmx_fio_write_real(srd->binaryOutputFile, (real)(srd->outVelSum[ci][XX] * invNumSamples));
			gmx_fio_write_real(srd->binaryOutputFile, (real)(srd->outVelSum[ci][YY] * invNumSamples));
			gmx_fio_write_real(srd->binaryOutputFile, (real)(srd->outVelSum[ci][ZZ] * invNumSamples));
			clear_dvec(srd->outVelSum[ci]);
		}
	}
}

static inline void initialize_output_file(t_srd *srd)
{
	srd->binaryOutputFile = gmx_fio_open("cells.srd", "wb");
	write_output_file_header(srd);
}

static inline void finalize_output_file(t_srd *srd)
{
	gmx_fio_seek(srd->binaryOutputFile, srd->outFrameCountPos);
	gmx_fio_write_int(srd->binaryOutputFile, srd->outFrameCount);
	gmx_fio_close(srd->binaryOutputFile);
}

/**
 * Initializes a communication buffer with an initial capacity.
 */
static gmx_inline void init_srd_comm_buffer(t_srd *srd, srd_comm_buf *buffer, char *name, int capacity, int maxCapacity)
{
	capacity = min(capacity, maxCapacity);
	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Creating %s buffer with initial capacity %d\n", srd->nodeID, name, capacity);
	#endif
	buffer->name = name;
	buffer->maxCapacity = maxCapacity;

	if(srd->bCommunicateX)
		snew(buffer->X, capacity);
	snew(buffer->V, capacity);
	snew(buffer->M, capacity);
	buffer->capacity = capacity;
}

/**
 * Initializes an rvec buffer with an initial capacity.
 */
static gmx_inline void init_srd_rvec_buffer(gmx_unused t_srd *srd, srd_rvec_buf *buffer, char *name, int capacity, int maxCapacity)
{
	capacity = min(capacity, maxCapacity);
	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Creating %s buffer with initial capacity %d\n", srd->nodeID, name, capacity);
	#endif
	buffer->name = name;
	buffer->maxCapacity = maxCapacity;
	snew(buffer->buf, capacity);
	buffer->capacity = capacity;
}

/**
 * Initializes an integer buffer with an initial capacity.
 */
static gmx_inline void init_srd_int_buffer(gmx_unused t_srd *srd, srd_int_buf *buffer, char *name, int capacity, int maxCapacity)
{
	capacity = min(capacity, maxCapacity);
	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Creating %s buffer with initial capacity %d\n", srd->nodeID, name, capacity);
	#endif
	buffer->name = name;
	buffer->maxCapacity = maxCapacity;
	snew(buffer->buf, capacity);
	buffer->capacity = capacity;
}

/**
 * Initializes a real buffer with an initial capacity.
 */
static gmx_inline void init_srd_real_buffer(gmx_unused t_srd *srd, srd_real_buf *buffer, char *name, int capacity, int maxCapacity)
{
	capacity = min(capacity, maxCapacity);
	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Creating %s buffer with initial capacity %d\n", srd->nodeID, name, capacity);
	#endif
	buffer->name = name;
	buffer->maxCapacity = maxCapacity;
	snew(buffer->buf, capacity);
	buffer->capacity = capacity;
}

/**
 * Initializes a double buffer with an initial capacity.
 */
static gmx_inline void init_srd_double_buffer(gmx_unused t_srd *srd, srd_double_buf *buffer, char *name, int capacity, int maxCapacity)
{
	capacity = min(capacity, maxCapacity);
	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Creating %s buffer with initial capacity %d\n", srd->nodeID, name, capacity);
	#endif
	buffer->name = name;
	buffer->maxCapacity = maxCapacity;
	snew(buffer->buf, capacity);
	buffer->capacity = capacity;
}

/**
 * Checks the current capacity of a communication buffer and reallocates if there
 * is not enough room. Returns TRUE if the buffer was resized, FALSE if not.
 */
gmx_bool realloc_srd_comm_buffer(gmx_unused t_srd *srd, srd_comm_buf *buffer, int size)
{
	if(size <= buffer->capacity)
		return FALSE;

	// overallocate a bit
	size = ceil(size * 1.25);

	if(size > buffer->maxCapacity)
		size = buffer->maxCapacity;

	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Resizing %s buffer from %d to %d\n", srd->nodeID, buffer->name, buffer->capacity, size);
	#endif

	if(srd->bCommunicateX)
		srenew(buffer->X, size);
	srenew(buffer->V, size);
	srenew(buffer->M, size);
	buffer->capacity = size;

	return TRUE;
}

/**
 * Checks the current capacity of an rvec buffer and reallocates if there
 * is not enough room. Returns TRUE if the buffer was resized, FALSE if not.
 */
gmx_bool realloc_srd_rvec_buffer(gmx_unused t_srd *srd, srd_rvec_buf *buffer, int size)
{
	if(size <= buffer->capacity)
		return FALSE;

	// overallocate a bit
	size = ceil(size * 1.25);

	if(size > buffer->maxCapacity)
		size = buffer->maxCapacity;

	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Resizing %s buffer from %d to %d\n", srd->nodeID, buffer->name, buffer->capacity, size);
	#endif

	srenew(buffer->buf, size);
	buffer->capacity = size;

	return TRUE;
}

/**
 * Checks the current capacity of an integer buffer and reallocates if there
 * is not enough room. Returns TRUE if the buffer was resized, FALSE if not.
 */
gmx_bool realloc_srd_int_buffer(gmx_unused t_srd *srd, srd_int_buf *buffer, int size)
{
	if(size <= buffer->capacity)
		return FALSE;

	// overallocate a bit
	size = ceil(size * 1.25);

	if(size > buffer->maxCapacity)
		size = buffer->maxCapacity;

	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Resizing %s buffer from %d to %d\n", srd->nodeID, buffer->name, buffer->capacity, size);
	#endif

	srenew(buffer->buf, size);
	buffer->capacity = size;

	return TRUE;
}

/**
 * Checks the current capacity of an integer buffer and reallocates if there
 * is not enough room. Returns TRUE if the buffer was resized, FALSE if not.
 */
gmx_bool realloc_srd_real_buffer(gmx_unused t_srd *srd, srd_real_buf *buffer, int size)
{
	if(size <= buffer->capacity)
		return FALSE;

	// overallocate a bit
	size = ceil(size * 1.25);

	if(size > buffer->maxCapacity)
		size = buffer->maxCapacity;

	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Resizing %s buffer from %d to %d\n", srd->nodeID, buffer->name, buffer->capacity, size);
	#endif

	srenew(buffer->buf, size);
	buffer->capacity = size;

	return TRUE;
}

/**
 * Checks the current capacity of an integer buffer and reallocates if there
 * is not enough room. Returns TRUE if the buffer was resized, FALSE if not.
 */
gmx_bool realloc_srd_double_buffer(gmx_unused t_srd *srd, srd_double_buf *buffer, int size)
{
	if(size <= buffer->capacity)
		return FALSE;

	// overallocate a bit
	size = ceil(size * 1.25);

	if(size > buffer->maxCapacity)
		size = buffer->maxCapacity;

	#ifdef SRD_PRINT_BUFFER_INFO
	printf("[node %d] Resizing %s buffer from %d to %d\n", srd->nodeID, buffer->name, buffer->capacity, size);
	#endif

	srenew(buffer->buf, size);
	buffer->capacity = size;

	return TRUE;
}

#ifdef SRD_ANGULAR_MOMENTUM_TEST
/**
 * Calculates the angular momentum for each cell, storing the results in an
 * rvec array. Uses positions and velocities relative to each cell.
 */
static void calculate_angular_momentum(t_srd *srd, t_state *state_local, rvec *result)
{
	int ai;
	rvec xrel, vrel, L;

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);
		rvec_sub(state_local->v[ai], srd->cellMeanV[cellID], vrel);

		// L = r x p
		cprod(xrel, vrel, L);
		rvec_inc(result[cellID], L);
	}

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		rvec_sub(srd->adoptionBuffer->X[ai], srd->cellMeanX[cellID], xrel);
		rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], vrel);

		// L = r x p
		cprod(xrel, vrel, L);
		rvec_inc(result[cellID], L);
	}
}

/**
 * Compares angular momentum for each cell calculated before and after collisions.
 */
static void check_angular_momentum(t_srd *srd)
{
	int cc, ci, cj, ck;
	cc = -1;

	for(ci = 0; ci < srd->cellCount[XX]; ci++)
	for(cj = 0; cj < srd->cellCount[YY]; cj++)
	for(ck = 0; ck < srd->cellCount[ZZ]; ck++)
	{
		if (!srd->isLocalCell[++cc])
			continue;

		if(!rvec_eq(srd->initialAngularMomentum[cc], srd->finalAngularMomentum[cc], 1e-4))
			fprintf(srd->localLogFP, "<Step %"GMX_PRId64"> Angular momentum in cell %d (%d %d %d): (%f %f %f) %f --> %f (%f %f %f)\n",
				srd->step, cc, ci, cj ,ck,
				srd->initialAngularMomentum[cc][XX],
				srd->initialAngularMomentum[cc][YY],
				srd->initialAngularMomentum[cc][ZZ],
				norm(srd->initialAngularMomentum[cc]),
				norm(srd->finalAngularMomentum[cc]),
				srd->finalAngularMomentum[cc][XX],
				srd->finalAngularMomentum[cc][YY],
				srd->finalAngularMomentum[cc][ZZ]);
	}
}

/**
 * Calculates the angular momentum for each cell, storing the results in an
 * rvec array. Uses positions and velocities relative to each cell.
 */
static void calculate_angular_momentum_EXP(t_srd *srd, t_state *state_local, rvec *result)
{
	int ai;
	rvec xrel, vrel, L;

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);
		rvec_sub(state_local->v[ai], srd->cellMeanV[cellID], vrel);

		// L = r x p
		cprod(xrel, vrel, L);
		rvec_inc(result[cellID], L);
	}
}

/**
 * Compares angular momentum for each cell calculated before and after collisions.
 */
static void check_angular_momentum_EXP(t_srd *srd)
{
	int ri, rj, rk;

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int cc = cxy + srd->localCells[ZZ][rk];
			if(!srd->isOccupied[cc])
				continue;

			int ci = srd->localCells[XX][ri];
			int cj = srd->localCells[YY][rj];
			int ck = srd->localCells[ZZ][rk];

			if(!rvec_eq(srd->initialAngularMomentum[cc], srd->finalAngularMomentum[cc], 1e-4))
				fprintf(srd->localLogFP, "<Step %"GMX_PRId64"> Angular momentum in cell %d (%d %d %d): (%f %f %f) %f --> %f (%f %f %f)\n",
					srd->step, cc, ci, cj ,ck,
					srd->initialAngularMomentum[cc][XX],
					srd->initialAngularMomentum[cc][YY],
					srd->initialAngularMomentum[cc][ZZ],
					norm(srd->initialAngularMomentum[cc]),
					norm(srd->finalAngularMomentum[cc]),
					srd->finalAngularMomentum[cc][XX],
					srd->finalAngularMomentum[cc][YY],
					srd->finalAngularMomentum[cc][ZZ]);
		}
	}
}
#endif

#ifdef SRD_COMM_TEST
static void comm_test_srd_thermostat(t_srd *srd, int nglobal)
{
	int ai, ci;

	for (ci = 0; ci < srd->numCells; ci++)
	{
		srd->testCellVRel2[ci] = 0;
		srd->testCellRescalingFactor[ci] = 1.0;
	}

	for (ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
			continue;

		int cellID = srd->testCellID[ai];

		rvec relVel;
		rvec_sub(srd->testV[ai], srd->testCellV[cellID], relVel);
		srd->testCellVRel2[cellID] += norm2(relVel);
	}

	for (ci = 0; ci < srd->numCells; ci++)
	{
		double uniformRN[4];
		gmx_rng_cycle_2uniform(srd->step, srd->testRngCount++, srd->rngSeed, RND_SEED_SRD, &uniformRN[0]);
		gmx_rng_cycle_2uniform(srd->step, srd->testRngCount++, srd->rngSeed, RND_SEED_SRD, &uniformRN[2]);

		real vsqrSum = srd->testCellVRel2[ci];
		int occupancy = srd->testCellN[ci];

		if (occupancy > 0)
		{
			real power = 3 * (occupancy - 1);
			real thermostatScalingFactor = 1 + (uniformRN[0] * srd->thermostatStrength);

			if (uniformRN[1] < 0.5)
				thermostatScalingFactor = 1 / thermostatScalingFactor;

			real entropicPrefactor = pow(thermostatScalingFactor, power);

			real scalingFactorFactor = thermostatScalingFactor * thermostatScalingFactor - 1;
			real kB = 0.00831446214; // amu * (nm/ps)^2 / K
			real expPower = (srd->mass * scalingFactorFactor * vsqrSum) / (2 * kB * srd->thermostatTarget);

			real acceptanceProbability = entropicPrefactor * exp(-expPower);
			real p = uniformRN[2];

			srd->testCellRescalingFactor[ci] = (p < acceptanceProbability) ? thermostatScalingFactor : 1.0;
		}
	}
}

static void comm_test_srd(t_srd *srd, int nglobal)
{
	int ai, ci;

	for (ci = 0; ci < srd->numCells; ci++)
	{
		rvec gaussRN;
		gmx_rng_cycle_3gaussian_table(srd->step, srd->testRngCount++, srd->rngSeed, RND_SEED_SRD, gaussRN);
		unitv(gaussRN, srd->testCellRotationAxis[ci]);
	}

	if (srd->enableThermostat)
		comm_test_srd_thermostat(srd, nglobal);

	for (ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
			continue;

		int cellID = srd->testCellID[ai];

		rvec vrel;
		rvec_sub(srd->testV[ai], srd->testCellV[cellID], vrel);

		unitaxis_angle_rotate(vrel, srd->testCellRotationAxis[cellID], srd->sinAngle, srd->cosAngle);

		if (srd->enableThermostat)
			svmul(srd->testCellRescalingFactor[cellID], vrel, vrel);

		rvec_add(vrel, srd->testCellV[cellID], srd->testVp[ai]);
	}
}

static void comm_test_andersen(t_srd *srd, int nglobal)
{
	int ai, ci;

	for (ci = 0; ci < srd->numCells; ci++)
	{
		clear_rvec(srd->testCellRandomSum[ci]);

		if(srd->empcRule == empcATA)
		{
			clear_rvec(srd->testCellDeltaL[ci]);
			clear_rvec(srd->testCellDeltaOmega[ci]);
			clear_dmat(srd->testCellInertiaTensor[ci]);
		}
	}

	for (ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
			continue;

		int cellID = srd->testCellID[ai];

		rvec vrand, gaussRN;
		gmx_rng_cycle_3gaussian_table(srd->step, ai, srd->rngSeed, RND_SEED_SRD, gaussRN);
		svmul(srd->mpcatVariance, gaussRN, vrand);

		rvec_inc(srd->testCellRandomSum[cellID], vrand);
		copy_rvec(vrand, srd->testParticleVRand[ai]);
	}

	for (ci = 0; ci < srd->numCells; ci++)
	{
		int n = srd->testCellN[ci];
		if (n > 1)
		{
			real n_inv = 1.0 / n;
			svmul(n_inv, srd->testCellRandomSum[ci], srd->testCellRandomSum[ci]);
		}
	}

	if(srd->empcRule == empcATA)
	{
		rvec xrel, vrel, dv, dL;

		for (ai = 0; ai < nglobal; ai++)
		{
			if (!srd->isSRD_gl[ai])
				continue;

			int cellID = srd->testCellID[ai];

			rvec_sub(srd->testXRel[ai], srd->testCellX[cellID], xrel);

			srd->testCellInertiaTensor[cellID][XX][XX] += xrel[YY]*xrel[YY] + xrel[ZZ]*xrel[ZZ];
			srd->testCellInertiaTensor[cellID][YY][YY] += xrel[XX]*xrel[XX] + xrel[ZZ]*xrel[ZZ];
			srd->testCellInertiaTensor[cellID][ZZ][ZZ] += xrel[XX]*xrel[XX] + xrel[YY]*xrel[YY];
			srd->testCellInertiaTensor[cellID][XX][YY] -= xrel[XX]*xrel[YY];
			srd->testCellInertiaTensor[cellID][YY][XX] -= xrel[XX]*xrel[YY];
			srd->testCellInertiaTensor[cellID][XX][ZZ] -= xrel[XX]*xrel[ZZ];
			srd->testCellInertiaTensor[cellID][ZZ][XX] -= xrel[XX]*xrel[ZZ];
			srd->testCellInertiaTensor[cellID][YY][ZZ] -= xrel[YY]*xrel[ZZ];
			srd->testCellInertiaTensor[cellID][ZZ][YY] -= xrel[YY]*xrel[ZZ];

			// vrel = v - cell_mean(v)
			rvec_sub(srd->testV[ai], srd->testCellV[cellID], vrel);

			// dv = (vrand - cell_mean(vrand)) - vrel
			rvec_sub(srd->testParticleVRand[ai], srd->testCellRandomSum[cellID], dv);
			rvec_dec(dv, vrel);

			cprod(xrel, dv, dL);
			rvec_inc(srd->testCellDeltaL[cellID], dL);
		}

		for (ci = 0; ci < srd->numCells; ci++)
		{
			if (srd->testCellN[ci] < 2)
				continue;

			dmatrix invInertiaTensor;
			gmx_bool success = dm_inv(srd->testCellInertiaTensor[ci], invInertiaTensor);

			if(success)
				dmvmul(invInertiaTensor, srd->testCellDeltaL[ci], srd->testCellDeltaOmega[ci]);
			else
				gmx_fatal(FARGS, "Failed to invert moment of inertia tensor in do_test_andersen_collisions()");
		}
	}

	for (ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
			continue;

		int cellID = srd->testCellID[ai];

		rvec_add(srd->testCellV[cellID], srd->testParticleVRand[ai], srd->testVp[ai]);

		// remove random drift
		rvec_dec(srd->testVp[ai], srd->testCellRandomSum[cellID]);

		// remove added angular momentum
		if(srd->empcRule == empcATA)
		{
			rvec xrel, angularMomentumCorrection;
			rvec_sub(srd->testXRel[ai], srd->testCellX[cellID], xrel);

			cprod(srd->testCellDeltaOmega[cellID], xrel, angularMomentumCorrection);
			rvec_dec(srd->testVp[ai], angularMomentumCorrection);
		}
	}
}

static void comm_test_collisions(t_srd *srd, int nglobal)
{
	int ai, ci;

	for (ci = 0; ci < srd->numCells; ci++)
	{
		srd->testCellN[ci] = 0;
		srd->testCellM[ci] = 0;
		clear_rvec(srd->testCellV[ci]);
		clear_rvec(srd->testCellX[ci]);
	}

	for (ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
			continue;

		int cellID = srd->testCellID[ai];
		rvec_inc(srd->testCellV[cellID], srd->testV[ai]);
		rvec_inc(srd->testCellX[cellID], srd->testXRel[ai]);
		srd->testCellN[cellID]++;
	}

	for (ci = 0; ci < srd->numCells; ci++)
	{
		int n = srd->testCellN[ci];
		if (n > 1)
		{
			real n_inv = 1.0 / n;
			svmul(n_inv, srd->testCellV[ci], srd->testCellV[ci]);
			svmul(n_inv, srd->testCellX[ci], srd->testCellX[ci]);
		}
	}

	switch(srd->empcRule)
	{
	case empcSRD:
		comm_test_srd(srd, nglobal);
		break;
	case empcAT:
	case empcATA:
		comm_test_andersen(srd, nglobal);
		break;
	}
}

static void comm_test_init(t_srd *srd, int nglobal)
{
	snew(srd->testX, nglobal);
	snew(srd->testV, nglobal);
	snew(srd->testVp, nglobal);
	snew(srd->testCellID, nglobal);
	snew(srd->testXRel, nglobal);
	srd->testRngCount = 0;

	snew(srd->testCellN, srd->numCells);
	snew(srd->testCellM, srd->numCells);
	snew(srd->testCellX, srd->numCells);
	snew(srd->testCellV, srd->numCells);

	switch(srd->empcRule)
	{
	case empcSRD:
		snew(srd->testCellRotationAxis, srd->numCells);
		snew(srd->testCellVRel2, srd->numCells);
		snew(srd->testCellRescalingFactor, srd->numCells);
		break;
	case empcAT:
	case empcATA:
		snew(srd->testCellRandomSum, srd->numCells);
		snew(srd->testParticleVRand, nglobal);
		break;
	}

	if(srd->empcRule == empcATA)
	{
		snew(srd->testCellDeltaL, srd->numCells);
		snew(srd->testCellDeltaOmega, srd->numCells);
		snew(srd->testCellInertiaTensor, srd->numCells);
	}
}

static void comm_test_assign_atoms_to_cells(t_srd *srd, int nglobal)
{
	int d, ai;
	ivec cv;

	for(ai = 0; ai < nglobal; ai++)
	{
		if (!srd->isSRD_gl[ai])
		{
			srd->testCellID[ai] = -1;
			continue;
		}

		// Work in the frame of collision cell grid, which extends from 0 to boxSize
		// in each dimension. In this frame, atoms need to be shifted and may fall
		// outside the periodic box.
		for (d = 0; d < DIM; d++)
		{
			real boxSize = srd->boxSize[d];
			real shifted = srd->testX[ai][d] - srd->gridShift[d];
			while (shifted < 0.0)
				shifted += boxSize;
			while (shifted >= boxSize)
				shifted -= boxSize;

			// protect against corner cases:
			// * shifted = -0.0
			// * shifted / cellSize > numCells, ex: 9.999999 / 10.000000 => 1.0
			cv[d] = (int)floor(shifted / srd->cellSize[d]);
			if (cv[d] == srd->cellCount[d])
				cv[d] = 0;

			if (srd->bCommunicateX)
				srd->testXRel[ai][d] = shifted - (cv[d] * srd->cellSize[d]);
		}

		srd->testCellID[ai] = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * cv[XX]) + cv[YY])) + cv[ZZ];
	}
}

static void comm_test_bcast(t_srd *srd, int nglobal, t_commrec *cr)
{
	gmx_bcast_sim(nglobal * sizeof(rvec), srd->testX, cr);
	gmx_bcast_sim(nglobal * sizeof(rvec), srd->testVp, cr);
	gmx_bcast_sim(srd->numCells * sizeof(int), srd->testCellN, cr);
	gmx_bcast_sim(srd->numCells * sizeof(real), srd->testCellM, cr);
	gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellX, cr);
	gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellV, cr);

	switch(srd->empcRule)
	{
	case empcSRD:
		gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellRotationAxis, cr);
		gmx_bcast_sim(srd->numCells * sizeof(real), srd->testCellVRel2, cr);
		gmx_bcast_sim(srd->numCells * sizeof(real), srd->testCellRescalingFactor, cr);
		break;
	case empcAT:
	case empcATA:
		gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellRandomSum, cr);
		gmx_bcast_sim(nglobal * sizeof(rvec), srd->testParticleVRand, cr);
		break;
	}

	if(srd->empcRule == empcATA)
	{
		gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellDeltaL, cr);
		gmx_bcast_sim(srd->numCells * sizeof(rvec), srd->testCellDeltaOmega, cr);
		gmx_bcast_sim(srd->numCells * sizeof(dmatrix), srd->testCellInertiaTensor, cr);
	}
}

static void comm_test_update(t_srd *srd, t_state *state_local, t_state *state_global, t_commrec *cr)
{
	int ai;
	srd->testRngCount = 0;

	if (DOMAINDECOMP(cr))
	{
		dd_collect_vec(cr->dd, state_local, state_local->x, srd->testX);
		dd_collect_vec(cr->dd, state_local, state_local->v, srd->testV);
	}
	else
	{
		for(ai = 0; ai < state_global->natoms; ai++)
		{
			copy_rvec(state_local->x[ai], srd->testX[ai]);
			copy_rvec(state_local->v[ai], srd->testV[ai]);
		}
	}

	if (MASTER(cr))
	{
		comm_test_assign_atoms_to_cells(srd, state_global->natoms);
		comm_test_collisions(srd, state_global->natoms);
	}

	if (DOMAINDECOMP(cr))
		comm_test_bcast(srd, state_global->natoms, cr);
}

static void comm_test_check_results(t_srd *srd, t_state *state_local, t_commrec *cr)
{
	int ai, ci, cj, ck;
	gmx_bool foundBadCell = FALSE;

	if (DOMAINDECOMP(cr))
	{
		int sumLocalCell[srd->numCells];
		#ifdef GMX_MPI
		MPI_Reduce(srd->isLocalCell, sumLocalCell, srd->numCells, MPI_INT, MPI_SUM, MASTERRANK(cr), cr->dd->mpi_comm_all);
		#endif

		if (MASTER(cr))
		{
			for (ci = 0; ci < srd->numCells; ci++)
			{
				if (sumLocalCell[ci] != 1)
					printf("<Step %"GMX_PRId64"> ERROR: Cell %d is claimed by %d nodes!\n", srd->step, ci, sumLocalCell[ci]);
			}
		}
	}

	for (ci = 0; ci < srd->cellCount[XX]; ci++)
	for (cj = 0; cj < srd->cellCount[YY]; cj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * ci) + cj));
		for (ck = 0; ck < srd->cellCount[ZZ]; ck++)
		{
			int cc = cxy + ck;
			if (!srd->isLocalCell[cc])
				continue;

			if (srd->testCellN[cc] != srd->cellOccupancy[cc])
			{
				foundBadCell = TRUE;
				#ifdef SRD_COMM_DEBUG
				fprintf(srd->localLogFP, "ERROR: Inconsistent cell occupancy [node %2d][cell %2d %2d %2d (%d)] %d vs %d\n",
						cr->nodeid, ci, cj, ck, cc,
						srd->testCellN[cc], srd->cellOccupancy[cc]);
				#endif
				printf("<Step %"GMX_PRId64"> ERROR: Inconsistent cell occupancy [node %2d][cell %2d %2d %2d (%d)] %d vs %d\n",
						srd->step, cr->nodeid, ci, cj, ck, cc,
						srd->testCellN[cc], srd->cellOccupancy[cc]);
			}
			else if (!rvec_eq(srd->testCellV[cc], srd->cellMeanV[cc], 1e-6))
			{
				printf("ERROR: Inconsistent cell velocity [node %2d][cell %2d %2d %2d] (%f %f %f) vs (%f %f %f)\n",
						cr->nodeid, ci, cj, ck,
						srd->testCellV[cc][XX], srd->testCellV[cc][YY], srd->testCellV[cc][ZZ],
						srd->cellMeanV[cc][XX], srd->cellMeanV[cc][YY], srd->cellMeanV[cc][ZZ]);
			}
		}
	}

	// no bad cell data, check particle velocity data
	if(!foundBadCell)
	{
		for (ai = 0; ai < srd->homenr; ai++)
		{
			int globalIndex = (srd->gatindex == NULL) ? ai : srd->gatindex[ai];
			if (!srd->isSRD_gl[globalIndex])
				continue;

			if (!rvec_eq(srd->testVp[globalIndex], state_local->v[ai], 1e-6))
			{
				printf("ERROR: Inconsistent particle velocity [node %d][atom %d] (%f %f %f) vs (%f %f %f)\n",
						cr->nodeid, globalIndex,
						srd->testVp[globalIndex][XX], srd->testVp[globalIndex][YY], srd->testVp[globalIndex][ZZ],
						state_local->v[ai][XX], state_local->v[ai][YY], state_local->v[ai][ZZ]);
			}
		}
	}
}

static void comm_test_check_results_EXP(t_srd *srd, t_state *state_local)
{
	int ai, ri, rj, rk;
	gmx_bool foundBadCell = FALSE;

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int cc = cxy + srd->localCells[ZZ][rk];
			if(!srd->isOccupied[cc])
				continue;

			int ci = srd->localCells[XX][ri];
			int cj = srd->localCells[YY][rj];
			int ck = srd->localCells[ZZ][rk];

			if (srd->testCellN[cc] != srd->cellOccupancy[cc])
			{
				foundBadCell = TRUE;
				fprintf(srd->localLogFP,
					"<Step %"GMX_PRId64"> ERROR: Inconsistent cell occupancy [cell %d (%d %d %d)] %d vs %d (correct)\n",
					srd->step, cc, ci, cj, ck, srd->cellOccupancy[cc], srd->testCellN[cc]);
			}
			else if (!rvec_eq(srd->testCellV[cc], srd->cellMeanV[cc], 1e-6))
			{
				foundBadCell = TRUE;
				fprintf(srd->localLogFP,
					"<Step %"GMX_PRId64"> ERROR: Inconsistent cell velocity [cell %d (%d %d %d)] (%f %f %f) vs (%f %f %f) (correct)\n",
					srd->step, cc, ci, cj, ck,
					srd->cellMeanV[cc][XX], srd->cellMeanV[cc][YY], srd->cellMeanV[cc][ZZ],
					srd->testCellV[cc][XX], srd->testCellV[cc][YY], srd->testCellV[cc][ZZ]);
			}
		}
	}

	// no bad cell data, check particle velocity data
	if(!foundBadCell)
	{
		for (ai = 0; ai < srd->homenr; ai++)
		{
			int globalIndex = (srd->gatindex == NULL) ? ai : srd->gatindex[ai];
			if (!srd->isSRD_gl[globalIndex])
				continue;

			if (!rvec_eq(srd->testVp[globalIndex], state_local->v[ai], 1e-6))
			{
				fprintf(srd->localLogFP,
					"<Step %"GMX_PRId64"> ERROR: Inconsistent particle velocity [atom %d] (%f %f %f) vs (%f %f %f) (correct)\n",
					srd->step, globalIndex,
					state_local->v[ai][XX], state_local->v[ai][YY], state_local->v[ai][ZZ],
					srd->testVp[globalIndex][XX], srd->testVp[globalIndex][YY], srd->testVp[globalIndex][ZZ]);
			}
		}
	}
}
#endif

/**
 * Allocates and initializes data structures used for SRD collisions and communication.
 */
void init_srd(
		FILE				*fplog,
		t_srd				*srd,
		t_state				*state_local,
		t_state				*state_global,
		gmx_mtop_t			*mtop,
		t_inputrec			*ir,
		t_commrec			*cr)
{
	if (ir->empcType == empcOFF)
		return;

	srd->commVer = ir->userint1; //XXX TEMP

	if(TRICLINIC(state_local->box))
		gmx_fatal(FARGS, "MPC requires a rectangular periodic cell!\n");

	int i, j;
	for (i = 0; i < 3; i++)
	for (j = 0; j < i; j++)
		if (ir->deform[i][j] != 0)
		{
			gmx_fatal(FARGS, "MPC cannot be used with deform!\n");
		}

	please_cite(fplog, "Zgorski2016");

	int ai, ci, d;

	srd->bMaster = MASTER(cr);
	srd->nodeID = cr->nodeid;
	srd->step = 0;
	srd->delta_t = ir->delta_t;

	// ======================================================================
	// read input parameters from inputrec
	// ======================================================================

	srd->empcRule = ir->empcType;

	srd->bCommunicateX = (srd->empcRule == empcATA);

	srd->collisionFrequency = ir->mpcCollisionFrequency;
	srd->collisionAngle = ir->srdAngle;

	srd->sinAngle = sin(M_PI * ir->srdAngle / 180.0);
	srd->cosAngle = cos(M_PI * ir->srdAngle / 180.0);

	srd->rngSeed = ir->mpcSeed;
	srd->rngCount = 0;
	srd->rng = gmx_rng_init(ir->mpcSeed);

	srd->enableThermostat = ir->tcoupleSRD;
	srd->thermostatStrength = ir->srdThermostatStrength;
	srd->thermostatTarget = ir->srdThermostatTarget;

	srd->outputFrequency = ir->mpcOutputFrequency;
	srd->samplingFrequency = ir->mpcOutputSampling;
	srd->writeCellVectors = ir->mpcWriteCellVectors;
	srd->writeCellOccupancy = ir->mpcWriteCellOccupancy;

	srd->binaryOutputFile = NULL;

	// print applicable warnings to the log file
	if (MASTER(cr))
	{
		if (srd->commVer == 1)
			printf("<MPC> Using experimental communication scheme.\n");
		#ifdef SRD_COMM_TEST
		printf("<MPC> WARNING: Compiled with communication test enabled, expect degraded performance.\n");
		#endif
		#ifdef SRD_ANGULAR_MOMENTUM_TEST
		printf("<MPC> WARNING: Compiled with angular momentum test enabled, expect degraded performance.\n");
		if (srd->empcRule != empcATA)
			gmx_fatal(FARGS, "Angular momentum tests are only compatile with the at+a collision rule!\n");
		#endif
		#ifdef SRD_ROUTING_DEBUG
		printf("<MPC> WARNING: Compiled with routing debug enabled.\n");
		#endif
		#ifdef SRD_COMM_DEBUG
		printf("<MPC> WARNING: Compiled with communication debug enabled.\n");
		#endif
	}

	// ======================================================================
	// calculate number and initial size of collision cells
	// ======================================================================

	for (d = 0; d < DIM; d++)
	{
		srd->cellCount[d] = round(state_local->box[d][d] / ir->mpcCellSize);
		srd->cellCount[d] = max(1, srd->cellCount[d]);
		snew(srd->homeCells[d], srd->cellCount[d]);

		srd->cellSize[d] = state_local->box[d][d] / srd->cellCount[d];

		if(srd->cellSize[d] < 1e-4)
			gmx_fatal(FARGS, "MPC cells are too small: less than 1e-4 is not allowed! Found %f\n", srd->cellSize[d]);
	}

	srd->numCells = srd->cellCount[XX]*srd->cellCount[YY]*srd->cellCount[ZZ];

	// ======================================================================
	// allocate arrays
	// ======================================================================

	int initialAtomBufferSize = state_global->natoms / cr->nnodes;

	// could be replaced with a flag array to reduce required memory
	snew(srd->isSRD_gl, state_global->natoms);

	srd->cellID = (srd_int_buf*)malloc(sizeof(srd_int_buf));
	init_srd_int_buffer(srd, srd->cellID, "cellID", ceil(1.5 * initialAtomBufferSize), state_global->natoms);

	#ifdef SRD_ROUTING_DEBUG
	snew(srd->cellHomeRanks, srd->numCells);
	#endif

	snew(srd->isLocalCell, srd->numCells);
	snew(srd->cellRouting, srd->numCells);

	snew(srd->cellMeanV, srd->numCells);
	snew(srd->cellOccupancy, srd->numCells);
	snew(srd->cellMassTotal, srd->numCells);

	srd->relXBuf = (srd_rvec_buf*)malloc(sizeof(srd_rvec_buf));
	init_srd_rvec_buffer(srd, srd->relXBuf, "relX", ceil(1.5 * initialAtomBufferSize), state_global->natoms);

	if(srd->bCommunicateX)
		snew(srd->cellMeanX, srd->numCells);

	switch(srd->empcRule)
	{
	case empcSRD:
		snew(srd->cellRotationAxis, srd->numCells);
		snew(srd->cellRescalingFactor, srd->numCells);
		snew(srd->cellVRel2, srd->numCells);
		break;
	case empcAT:
	case empcATA:
		// handle common stuff here
		snew(srd->cellRandomSum, srd->numCells);
		srd->nativeVrandBuf = (srd_rvec_buf*)malloc(sizeof(srd_rvec_buf));
		srd->adoptedVrandBuf = (srd_rvec_buf*)malloc(sizeof(srd_rvec_buf));
		init_srd_rvec_buffer(srd, srd->nativeVrandBuf, "nativeVrand", ceil(1.5 * initialAtomBufferSize), state_global->natoms);
		init_srd_rvec_buffer(srd, srd->adoptedVrandBuf, "adoptedVrand", initialAtomBufferSize, state_global->natoms);
		break;
	}

	// handle data uses for ATA only
	if(srd->empcRule == empcATA)
	{
		snew(srd->cellInertiaTensor, srd->numCells);
		snew(srd->cellDeltaL, srd->numCells);
		snew(srd->cellDeltaOmega, srd->numCells);
	}

	#ifdef SRD_ANGULAR_MOMENTUM_TEST
	snew(srd->initialAngularMomentum, srd->numCells);
	snew(srd->finalAngularMomentum, srd->numCells);
	#endif

#ifdef SRD_COMM_TEST
	comm_test_init(srd, state_global->natoms);
#endif

	if(srd->outputFrequency > 0)
	{
		if (MASTER(cr))
			initialize_output_file(srd);

		if(srd->writeCellOccupancy)
		{
			snew(srd->outOccBuf, srd->numCells);
			snew(srd->outOccSum, srd->numCells);
		}

		if(srd->writeCellVectors)
		{
			snew(srd->outVelBuf, srd->numCells);
			snew(srd->outVelSum, srd->numCells);
		}

		clear_dvec(srd->outSizeSum);
		srd->outFrameCount = 0;
	}

	// ======================================================================
	// initialize communication
	// ======================================================================

	srd->sendBuffer = (srd_comm_buf*)malloc(sizeof(srd_comm_buf));
	srd->recvBuffer = (srd_comm_buf*)malloc(sizeof(srd_comm_buf));
	srd->dispatchBuffer = (srd_comm_buf*)malloc(sizeof(srd_comm_buf));
	srd->adoptionBuffer = (srd_comm_buf*)malloc(sizeof(srd_comm_buf));

	init_srd_comm_buffer(srd, srd->sendBuffer, "send", initialAtomBufferSize, state_global->natoms);
	init_srd_comm_buffer(srd, srd->recvBuffer, "recv", initialAtomBufferSize, state_global->natoms);
	init_srd_comm_buffer(srd, srd->adoptionBuffer, "adoption", initialAtomBufferSize, state_global->natoms);
	init_srd_comm_buffer(srd, srd->dispatchBuffer, "dispatch", ceil(1.5 * initialAtomBufferSize), state_global->natoms); // slightly larger than others
	snew(srd->dispatched, srd->dispatchBuffer->capacity);

	for(ci = 0; ci < srd->numCells; ci++)
	{
		srd->isLocalCell[ci] = !DOMAINDECOMP(cr);
		srd->cellRouting[ci] = -1;
	}

	if (!DOMAINDECOMP(cr))
	{
		for(d = 0; d < DIM; d++)
		{
			srd->homeCellCount[d] = srd->cellCount[d];
			for (ci = 0; ci < srd->cellCount[d]; ci++)
				srd->homeCells[d][ci] = ci;
		}
	}
	else
	{
		int ni, nj, nk;
		for (ni = 0; ni < 3; ni++)
		for (nj = 0; nj < 3; nj++)
		for (nk = 0; nk < 3; nk++)
		{
			clear_ivec(srd->prevCellRange[ni][nj][nk][0]);
			clear_ivec(srd->prevCellRange[ni][nj][nk][1]);
		}
	}

	#ifdef SRD_COMM_DEBUG
	snew(srd->atomState, state_global->natoms);
	#endif

	#ifdef SRD_LOCAL_LOGS
	char filename[16];
	sprintf(filename, "debug_%02d.log", cr->nodeid);
	srd->localLogFP = fopen(filename, "w");
	#endif

	// find atoms that participate in collisions
	int numSRD = 0;
	int srdAtomIndex = -1;

	for (ai = 0; ai < state_global->natoms; ai++)
	{
		char *atomname;
		int resnr;
		char *resname;
		gmx_mtop_atominfo_global(mtop, ai, &atomname, &resnr, &resname);

		if (strcmp("SRD", atomname) == 0)
		{
			if(srdAtomIndex < 0)
				srdAtomIndex = ai;
		}

		if (ggrpnr(&mtop->groups, egcMPC, ai) == 0)
		{
			srd->isSRD_gl[ai] = TRUE;
			numSRD++;
		}
	}

	if(numSRD < 1)
		gmx_fatal(FARGS, "No MPC particles were found!\n");

	// look up SRD atom properties
	t_atom *srdAtom;
	gmx_mtop_atomlookup_t atomlookup = gmx_mtop_atomlookup_init(mtop);
	gmx_mtop_atomnr_to_atom(atomlookup, srdAtomIndex, &srdAtom);
	gmx_mtop_atomlookup_destroy(atomlookup);
	srd->type = srdAtom->type;
	srd->mass = srdAtom->m;

	if (MASTER(cr))
	{
		printf("<MPC> Found %d MPC particles.\n", numSRD);
		printf("<MPC> Mass of MPC particles is %f\n", srd->mass);
		printf("<MPC> MPC cell grid is %d x %d x %d (%f %f %f)\n",
				srd->cellCount[XX], srd->cellCount[YY], srd->cellCount[ZZ],
				srd->cellSize[XX], srd->cellSize[YY], srd->cellSize[ZZ]);
	}

	real kB = 0.00831446214; // [amu * (nm/ps)^2 / K]
	srd->mpcatVariance = sqrt(kB * srd->thermostatTarget / srd->mass); // [nm/ps]

	// ======================================================================
	// prepare domdec
	// ======================================================================

	if (DOMAINDECOMP(cr))
		dd_srd_init(srd, cr);

	// ======================================================================

	if(srd->commVer == 1)
		init_srd_EXP(srd, cr);
}

/**
 * Sets the value of srd->isLocalCell and srd->cellRouting by iterating
 * over the srd->cellRange provided by get_neighbors_cell_range
 */
static gmx_inline void assign_collision_cells(t_srd *srd)
{
	int ni, nj, nk; // neighbor indicies 0-2
	int ci, cj, ck; // cell ranges from neighbors
	int si, sj, sk; // cell ranges after periodic shifts
	int d;

	ivec cmin, cmax;

	// clear values from previous step
	for(ni = 0; ni < 3; ni++)
	for(nj = 0; nj < 3; nj++)
	for(nk = 0; nk < 3; nk++)
	{
		if(srd->neighbors[ni][nj][nk] == SRD_NO_NEIGHBOR)
			continue;

		gmx_bool local = (ni == 1) && (nj == 1) && (nk == 1);

		copy_ivec(srd->prevCellRange[ni][nj][nk][0], cmin);
		copy_ivec(srd->prevCellRange[ni][nj][nk][1], cmax);

		for (ci = cmin[XX]; ci < cmax[XX]; ci++)
		{
			si = ci;
			if (si >= srd->cellCount[XX])
				si -= srd->cellCount[XX];

			for (cj = cmin[YY]; cj < cmax[YY]; cj++)
			{
				sj = cj;
				if (sj >= srd->cellCount[YY])
					sj -= srd->cellCount[YY];

				int cc = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * si) + sj)) + cmin[ZZ];

				for (ck = cmin[ZZ]; ck < cmax[ZZ]; ck++)
				{
					sk = ck;
					if (sk >= srd->cellCount[ZZ])
					{
						sk -= srd->cellCount[ZZ];
						cc -= srd->cellCount[ZZ];
					}

					if (local)
						srd->isLocalCell[cc] = FALSE;
					else
						srd->cellRouting[cc] = -1;
					cc++;
				}
			}
		}
	}

	// calculate cell ranges for neighbors
	for(ni = 0; ni < 3; ni++)
	for(nj = 0; nj < 3; nj++)
	for(nk = 0; nk < 3; nk++)
	{
		if(srd->neighbors[ni][nj][nk] == SRD_NO_NEIGHBOR)
			continue;

		for(d = 0; d < DIM; d++)
		{
			real xmin = srd->boundaries[ni][nj][nk][0][d];
			real xmax = srd->boundaries[ni][nj][nk][1][d];
			srd->cellRange[ni][nj][nk][0][d] = round((xmin - srd->gridShift[d]) / srd->cellSize[d]);
			srd->cellRange[ni][nj][nk][1][d] = round((xmax - srd->gridShift[d]) / srd->cellSize[d]);
		}

		#ifdef SRD_ROUTING_DEBUG
			fprintf(srd->localLogFP, "<Step %"GMX_PRId64"> neighboring cells:\n", srd->step);
			for(i = 0; i < 3; i++)
				for(j = 0; j < 3; j++)
					for(k = 0; k < 3; k++)
					{
						fprintf(srd->localLogFP,"[node %d] [%d %d %d] (%d to %d) (%d to %d) (%d to %d)\n",
								srd->neighbors[i][j][k], i, j, k,
								srd->cellRange[i][j][k][0][XX], srd->cellRange[i][j][k][1][XX],
								srd->cellRange[i][j][k][0][YY],	srd->cellRange[i][j][k][1][YY],
								srd->cellRange[i][j][k][0][ZZ], srd->cellRange[i][j][k][1][ZZ]);
					}
		#endif
	}

	// create home cell lists
	copy_ivec(srd->cellRange[1][1][1][0], cmin);
	copy_ivec(srd->cellRange[1][1][1][1], cmax);
	ivec_sub(cmax, cmin, srd->homeCellCount);
	for(d = 0; d < DIM; d++)
	{
		int i = 0;
		for (ci = cmin[d]; ci < cmax[d]; ci++)
		{
			si = ci;
			if (si >= srd->cellCount[d])
				si -= srd->cellCount[d];
			srd->homeCells[d][i++] = si;
		}
	}

	// iterate over neighbor cell ranges to create cellRouting map
	for(ni = 0; ni < 3; ni++)
	for(nj = 0; nj < 3; nj++)
	for(nk = 0; nk < 3; nk++)
	{
		if(srd->neighbors[ni][nj][nk] == SRD_NO_NEIGHBOR)
			continue;

		gmx_bool local = (ni == 1) && (nj == 1) && (nk == 1);

		copy_ivec(srd->cellRange[ni][nj][nk][0], cmin);
		copy_ivec(srd->cellRange[ni][nj][nk][1], cmax);

		for (ci = cmin[XX]; ci < cmax[XX]; ci++)
		{
			si = ci;
			if (si >= srd->cellCount[XX])
				si -= srd->cellCount[XX];

			for (cj = cmin[YY]; cj < cmax[YY]; cj++)
			{
				sj = cj;
				if (sj >= srd->cellCount[YY])
					sj -= srd->cellCount[YY];

				int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * si) + sj));

				for (ck = cmin[ZZ]; ck < cmax[ZZ]; ck++)
				{
					sk = ck;
					if (sk >= srd->cellCount[ZZ])
						sk -= srd->cellCount[ZZ];

					int cc = cxy + sk;

					if (local)
						srd->isLocalCell[cc] = TRUE;
					else
						srd->cellRouting[cc] = srd->nodeRouting[srd->neighbors[ni][nj][nk]];
				}
			}
		}

		copy_ivec(cmin, srd->prevCellRange[ni][nj][nk][0]);
		copy_ivec(cmax, srd->prevCellRange[ni][nj][nk][1]);
	}
}

/**
 * Implementation of the thermostat described in Heyes 1983 (DOI: 10.1016/0301-0104(83)85235-5)
 * and updated in Hecht et al 2005 (DOI: 10.1103/PhysRevE.72.011408).
 * Operates on all colliding particles, assuming they all share the same mass.
 */
static void do_srd_thermostat(t_srd *srd, rvec *v)
{
	int ai;
	int ri, rj, rk;

	// ======================================================================
	// sum relative velocity squared for local SRD particles
	// ======================================================================

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec relVel;
		rvec_sub(v[ai], srd->cellMeanV[cellID], relVel);
		srd->cellVRel2[cellID] += norm2(relVel);
	}

	// ======================================================================
	// sum relative velocity squared for adopted SRD particles
	// ======================================================================

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		rvec relVel;
		rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], relVel);
		srd->cellVRel2[cellID] += norm2(relVel);
	}

	// ======================================================================
	// calculate thermostat rescaling factor
	// ======================================================================

	gmx_int64_t rngStart = srd->rngCount;

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];
			int rngCount = rngStart + 2*ci;

			double uniformRN[4];
			gmx_rng_cycle_2uniform(srd->step, rngCount, srd->rngSeed, RND_SEED_SRD, &uniformRN[0]);
			gmx_rng_cycle_2uniform(srd->step, rngCount + 1, srd->rngSeed, RND_SEED_SRD, &uniformRN[2]);

			#ifdef SRD_COMM_TEST
			// report the error, but do not propagate it
			if(!float_float_eq(srd->testCellVRel2[ci], srd->cellVRel2[ci], 1e-4)) // errors are compounded, more lenient "equality" check
			{
				printf("[%d] NOTICE: Inconsistent relative velocity squared [cell %2d] %f vs %f (correct)\n",
						srd->nodeID, ci, srd->cellVRel2[ci], srd->testCellVRel2[ci]);
				srd->cellVRel2[ci] = srd->testCellVRel2[ci];
			}
			#endif

			real vsqrSum = srd->cellVRel2[ci];
			int occupancy = srd->cellOccupancy[ci];

			if (occupancy > 0)
			{
				real power = 3 * (occupancy - 1);
				real thermostatScalingFactor = 1 + (uniformRN[0] * srd->thermostatStrength);

				if (uniformRN[1] < 0.5)
					thermostatScalingFactor = 1 / thermostatScalingFactor;

				real entropicPrefactor = pow(thermostatScalingFactor, power);

				real scalingFactorFactor = thermostatScalingFactor * thermostatScalingFactor - 1;
				real kB = 0.00831446214; // amu * (nm/ps)^2 / K
				real expPower = (srd->mass * scalingFactorFactor * vsqrSum) / (2 * kB * srd->thermostatTarget);

				real acceptanceProbability = entropicPrefactor * exp(-expPower);
				real p = uniformRN[2];

				srd->cellRescalingFactor[ci] = (p < acceptanceProbability) ? thermostatScalingFactor : 1.0;
			}

			#ifdef SRD_COMM_TEST
			if( !float_float_eq(srd->testCellRescalingFactor[ci], srd->cellRescalingFactor[ci], 1e-6))
			{
				// report the error, but do not propagate it
				printf("[%d] NOTICE: Inconsistent thermostat rescaling factor [cell %2d] %f vs %f (correct)\n",
						srd->nodeID, ci, srd->cellRescalingFactor[ci], srd->testCellRescalingFactor[ci]);
				srd->cellRescalingFactor[ci] = srd->testCellRescalingFactor[ci];
			}
			#endif
		}
	}

	srd->rngCount += 2 * srd->numCells;
}

static void prep_srd_collisions(t_srd *srd)
{
	int ri, rj, rk;
	gmx_int64_t rngStart = srd->rngCount;

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];

			rvec gaussRN;
			gmx_rng_cycle_3gaussian_table(srd->step, srd->rngCount + ci, srd->rngSeed, RND_SEED_SRD, gaussRN);
			unitv(gaussRN, srd->cellRotationAxis[ci]);

			if(srd->enableThermostat)
			{
				srd->cellVRel2[ci] = 0;
				srd->cellRescalingFactor[ci] = 1.0;
			}
		}
	}

	srd->rngCount = rngStart + srd->numCells;
}

static void do_srd_collisions(t_srd *srd, rvec *v)
{
	int ai, ci;

	// ======================================================================
	// calculate thermostat rescaling factor
	// ======================================================================

	if (srd->enableThermostat )
		do_srd_thermostat(srd, v);

	// ======================================================================
	// native particle collisions
	// ======================================================================

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec vrel;
		rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

		unitaxis_angle_rotate(vrel, srd->cellRotationAxis[cellID], srd->sinAngle, srd->cosAngle);

		if (srd->enableThermostat)
			svmul(srd->cellRescalingFactor[cellID], vrel, vrel);

		rvec_add(vrel, srd->cellMeanV[cellID], v[ai]);
	}

	// ======================================================================
	// adopted particle collisions
	// ======================================================================

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		rvec vrel;
		rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], vrel);

		unitaxis_angle_rotate(vrel, srd->cellRotationAxis[cellID], srd->sinAngle, srd->cosAngle);

		if (srd->enableThermostat)
			svmul(srd->cellRescalingFactor[cellID], vrel, vrel);

		rvec_add(vrel, srd->cellMeanV[cellID], srd->adoptionBuffer->V[ai]);
	}
}

static void prep_andersen_collisions(t_srd *srd)
{
	int ri, rj, rk;

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];

			clear_rvec(srd->cellRandomSum[ci]);

			if(srd->empcRule == empcATA)
			{
				clear_dmat(srd->cellInertiaTensor[ci]);
				clear_rvec(srd->cellDeltaL[ci]);
				clear_rvec(srd->cellDeltaOmega[ci]);
			}
		}
	}
}

static void do_andersen_collisions(t_srd *srd, rvec *v)
{
	int ri, rj, rk;
	int ai, ci;

	// ======================================================================
	// get random components for native particle collisions
	// ======================================================================

	realloc_srd_rvec_buffer(srd, srd->nativeVrandBuf, srd->homenr);

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		int globalAtomIndex = (srd->gatindex == NULL) ? ai : srd->gatindex[ai];

		rvec vrand, gaussRN;
		gmx_rng_cycle_3gaussian_table(srd->step, globalAtomIndex, srd->rngSeed, RND_SEED_SRD, gaussRN);
		svmul(srd->mpcatVariance, gaussRN, vrand);

		rvec_inc(srd->cellRandomSum[cellID], vrand);
		copy_rvec(vrand, srd->nativeVrandBuf->buf[ai]);
	}

	// ======================================================================
	// get random components for adopted particle collisions
	// ======================================================================

	realloc_srd_rvec_buffer(srd, srd->adoptedVrandBuf, srd->adoptedCount);

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		int globalAtomIndex = srd->adoptionBuffer->M[ai][MD_GNDX];

		rvec vrand, gaussRN;
		gmx_rng_cycle_3gaussian_table(srd->step, globalAtomIndex, srd->rngSeed, RND_SEED_SRD, gaussRN);
		svmul(srd->mpcatVariance, gaussRN, vrand);

		rvec_inc(srd->cellRandomSum[cellID], vrand);
		copy_rvec(vrand, srd->adoptedVrandBuf->buf[ai]);
	}

	// ======================================================================
	// sum random components for each cell
	// ======================================================================

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];

			int n = srd->cellOccupancy[ci];
			if (n > 1)
			{
				real n_inv = 1.0 / n;
				svmul(n_inv, srd->cellRandomSum[ci], srd->cellRandomSum[ci]);
			}
		}
	}

	// ======================================================================
	// sum moment of intertia and change in angular momentum for each cell
	// ======================================================================

	if(srd->empcRule == empcATA)
	{
		rvec xrel, vrel, dv, dL;

		for (ai = 0; ai < srd->homenr; ai++)
		{
			int cellID = srd->cellID->buf[ai];
			if (cellID < 0 || !srd->isLocalCell[cellID])
				continue;

			rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);

			srd->cellInertiaTensor[cellID][XX][XX] += xrel[YY]*xrel[YY] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][YY] += xrel[XX]*xrel[XX] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][ZZ] += xrel[XX]*xrel[XX] + xrel[YY]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][YY] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][YY][XX] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][ZZ] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][XX] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][ZZ] -= xrel[YY]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][YY] -= xrel[YY]*xrel[ZZ];

			// vrel = v - cell_mean(v)
			rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

			// dv = (vrand - cell_mean(vrand)) - vrel
			rvec_sub(srd->nativeVrandBuf->buf[ai], srd->cellRandomSum[cellID], dv);
			rvec_dec(dv, vrel);

			cprod(xrel, dv, dL);
			rvec_inc(srd->cellDeltaL[cellID], dL);
		}

		for (ai = 0; ai < srd->adoptedCount; ai++)
		{
			int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
			if (!srd->isLocalCell[cellID])
				gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

			rvec_sub(srd->adoptionBuffer->X[ai], srd->cellMeanX[cellID], xrel);

			srd->cellInertiaTensor[cellID][XX][XX] += xrel[YY]*xrel[YY] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][YY] += xrel[XX]*xrel[XX] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][ZZ] += xrel[XX]*xrel[XX] + xrel[YY]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][YY] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][YY][XX] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][ZZ] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][XX] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][ZZ] -= xrel[YY]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][YY] -= xrel[YY]*xrel[ZZ];

			// vrel = v - cell_mean(v)
			rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], vrel);

			// dv = (vrand - cell_mean(vrand)) - vrel
			rvec_sub(srd->adoptedVrandBuf->buf[ai], srd->cellRandomSum[cellID], dv);
			rvec_dec(dv, vrel);

			cprod(xrel, dv, dL);
			rvec_inc(srd->cellDeltaL[cellID], dL);
		}

		for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
		for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
		{
			int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
			for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
			{
				int ci = cxy + srd->homeCells[ZZ][rk];

				if (srd->cellOccupancy[ci] < 2)
					continue;

				#ifdef SRD_COMM_TEST
				if(!rvec_eq(srd->cellDeltaL[ci], srd->testCellDeltaL[ci], 1e-4))
				{
					printf("[%d] NOTICE: Inconsistent cellDeltaL! [cell %2d] (%f %f %f) vs (%f %f %f) (correct)\n",
							srd->nodeID, ci,
							srd->cellDeltaL[ci][XX],
							srd->cellDeltaL[ci][YY],
							srd->cellDeltaL[ci][ZZ],
							srd->testCellDeltaL[ci][XX],
							srd->testCellDeltaL[ci][YY],
							srd->testCellDeltaL[ci][ZZ]);
				}
				#endif

				dmatrix invInertiaTensor;
				gmx_bool success = dm_inv(srd->cellInertiaTensor[ci], invInertiaTensor);

				if(success)
					dmvmul(invInertiaTensor, srd->cellDeltaL[ci], srd->cellDeltaOmega[ci]);
				else
				{
					printf("<Cell %d> ERROR: Cannot invert moment of inertia tensor!", ci);
					printf("<Cell %d> %f %f %f\n", ci, srd->cellInertiaTensor[ci][XX][XX], srd->cellInertiaTensor[ci][XX][YY], srd->cellInertiaTensor[ci][XX][ZZ]);
					printf("<Cell %d> %f %f %f\n", ci, srd->cellInertiaTensor[ci][YY][XX], srd->cellInertiaTensor[ci][YY][YY], srd->cellInertiaTensor[ci][YY][ZZ]);
					printf("<Cell %d> %f %f %f\n", ci, srd->cellInertiaTensor[ci][ZZ][XX], srd->cellInertiaTensor[ci][ZZ][YY], srd->cellInertiaTensor[ci][ZZ][ZZ]);
					printf("<Cell %d> determinant = %f\n", ci, ddet(srd->cellInertiaTensor[ci]));
					printf("<Cell %d> cell occupancy = %d\n", ci, srd->cellOccupancy[ci]);
					rvec xrel, vrel;

					for (ai = 0; ai < srd->homenr; ai++)
					{
						int cellID = srd->cellID->buf[ai];
						if (cellID < 0 || !srd->isLocalCell[cellID] || cellID != ci)
							continue;

						rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);
						rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

						printf("<Cell %d> %f %f %f -- %f %f %f\n", ci, xrel[XX], xrel[YY], xrel[ZZ], vrel[XX], vrel[YY], vrel[ZZ]);
					}

					for (ai = 0; ai < srd->adoptedCount; ai++)
					{
						int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
						if (!srd->isLocalCell[cellID])
							gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

						if(cellID != ci)
							continue;

						rvec_sub(srd->adoptionBuffer->X[ai], srd->cellMeanX[cellID], xrel);
						rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], vrel);

						printf("<Cell %d> %f %f %f -- %f %f %f\n", ci, xrel[XX], xrel[YY], xrel[ZZ], vrel[XX], vrel[YY], vrel[ZZ]);
					}
				}

				#ifdef SRD_COMM_TEST
				if(!rvec_eq(srd->cellDeltaOmega[ci], srd->testCellDeltaOmega[ci], 1e-4))
				{
					printf("[%d] NOTICE: Inconsistent cellDeltaOmega! [cell %2d] (%f %f %f) vs (%f %f %f) (correct)\n",
							srd->nodeID, ci,
							srd->cellDeltaOmega[ci][XX],
							srd->cellDeltaOmega[ci][YY],
							srd->cellDeltaOmega[ci][ZZ],
							srd->testCellDeltaOmega[ci][XX],
							srd->testCellDeltaOmega[ci][YY],
							srd->testCellDeltaOmega[ci][ZZ]);
				}
				#endif
			}
		}
	}

	// ======================================================================
	// get final velocity for native particles
	// ======================================================================

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec_add(srd->cellMeanV[cellID], srd->nativeVrandBuf->buf[ai], v[ai]);

		// remove random drift
		rvec_dec(v[ai], srd->cellRandomSum[cellID]);

		// remove added angular momentum
		if(srd->empcRule == empcATA)
		{
			rvec xrel, angularMomentumCorrection;
			rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);

			cprod(srd->cellDeltaOmega[cellID], xrel, angularMomentumCorrection);
			rvec_dec(v[ai], angularMomentumCorrection);
		}
	}

	// ======================================================================
	// get final velocity for adopted particles
	// ======================================================================

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		rvec_add(srd->cellMeanV[cellID], srd->adoptedVrandBuf->buf[ai], srd->adoptionBuffer->V[ai]);

		// remove random drift
		rvec_dec(srd->adoptionBuffer->V[ai], srd->cellRandomSum[cellID]);

		// remove added angular momentum
		if(srd->empcRule == empcATA)
		{
			rvec xrel, angularMomentumCorrection;
			rvec_sub(srd->adoptionBuffer->X[ai], srd->cellMeanX[cellID], xrel);

			cprod(srd->cellDeltaOmega[cellID], xrel, angularMomentumCorrection);
			rvec_dec(srd->adoptionBuffer->V[ai], angularMomentumCorrection);
		}
	}
}

static void do_local_collisions(t_srd *srd, t_state *state_local)
{
	int ai, ci;
	int ri, rj, rk;

	// ======================================================================
	// prepare cells for new collision step
	// ======================================================================

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];

			clear_rvec(srd->cellMeanV[ci]);
			srd->cellOccupancy[ci] = 0;
			srd->cellMassTotal[ci] = 0;

			if(srd->bCommunicateX)
				clear_rvec(srd->cellMeanX[ci]);

			#ifdef SRD_ANGULAR_MOMENTUM_TEST
			clear_rvec(srd->initialAngularMomentum[ci]);
			clear_rvec(srd->finalAngularMomentum[ci]);
			#endif
		}
	}

	switch(srd->empcRule)
	{
	case empcSRD:
		prep_srd_collisions(srd);
		break;
	case empcAT:
	case empcATA:
		prep_andersen_collisions(srd);
		break;
	}

	// ======================================================================
	// sum native totals
	// ======================================================================

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0 || !srd->isLocalCell[cellID])
			continue;

		rvec_inc(srd->cellMeanV[cellID], state_local->v[ai]);

		if(srd->bCommunicateX)
			rvec_inc(srd->cellMeanX[cellID], srd->relXBuf->buf[ai]);

		srd->cellOccupancy[cellID]++;
	}

	// ======================================================================
	// sum adopted totals
	// ======================================================================

	for (ai = 0; ai < srd->adoptedCount; ai++)
	{
		int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
		if (!srd->isLocalCell[cellID])
			gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

		rvec_inc(srd->cellMeanV[cellID], srd->adoptionBuffer->V[ai]);

		if(srd->bCommunicateX)
			rvec_inc(srd->cellMeanX[cellID], srd->adoptionBuffer->X[ai]);

		srd->cellOccupancy[cellID]++;
	}

	// ======================================================================
	// find cell averages
	// ======================================================================

	for (ri = 0; ri < srd->homeCellCount[XX]; ri++)
	for (rj = 0; rj < srd->homeCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->homeCells[XX][ri]) + srd->homeCells[YY][rj]));
		for (rk = 0; rk < srd->homeCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->homeCells[ZZ][rk];

			int n = srd->cellOccupancy[ci];
			if (n > 1)
			{
				real n_inv = 1.0 / n;
				svmul(n_inv, srd->cellMeanV[ci], srd->cellMeanV[ci]);
				if (srd->bCommunicateX)
					svmul(n_inv, srd->cellMeanX[ci], srd->cellMeanX[ci]);
			}
		}
	}

	// ======================================================================
	// do collisions
	// ======================================================================

	#ifdef SRD_ANGULAR_MOMENTUM_TEST
	calculate_angular_momentum(srd, state_local, srd->initialAngularMomentum);
	#endif

	switch(srd->empcRule)
	{
	case empcSRD:
		do_srd_collisions(srd, state_local->v);
		break;
	case empcAT:
	case empcATA:
		do_andersen_collisions(srd, state_local->v);
		break;
	}

	#ifdef SRD_ANGULAR_MOMENTUM_TEST
	calculate_angular_momentum(srd, state_local, srd->finalAngularMomentum);
	check_angular_momentum(srd);
	#endif
}

void update_srd(
		t_srd				*srd,
		gmx_int64_t			 step,
		t_state				*state_local,
		gmx_unused t_state	*state_global,
		t_mdatoms			*mdatoms,
		t_commrec			*cr,
		gmx_wallcycle_t		 wcycle)
{
	int d, ai, ci, cj, ck, cc;

	if(srd->commVer == 1)
	{
		update_srd_EXP(srd, step, state_local, state_global, mdatoms, cr, wcycle);
		return;
	}

	if (!do_per_step(step, srd->collisionFrequency))
		return;

	// ======================================================================
	// initialization
	// ======================================================================

	wallcycle_start(wcycle, ewcMPC_INIT);

	srd->step = step;
	srd->rngCount = 0;
	srd->homenr = mdatoms->homenr;
	srd->gatindex = (DOMAINDECOMP(cr)) ? cr->dd->gatindex : NULL; // accommodate singe-core

	// update collision cell size to compensate for changes in box size.
	// the number of cells are fixed during initialization.
	for(d = 0; d < DIM; d++)
	{
		srd->boxSize[d] = state_local->box[d][d];
		srd->cellSize[d] = srd->boxSize[d] / srd->cellCount[d];
		srd->gridShift[d] = (gmx_rng_uniform_real(srd->rng) - 0.5) * srd->cellSize[d];
	}

	assign_atoms_to_cells(srd, state_local);

	wallcycle_stop(wcycle, ewcMPC_INIT);

	// ======================================================================
	// find local collision cells and those of domdec neighbors
	// ======================================================================

	if (DOMAINDECOMP(cr))
	{
		#ifdef SRD_ROUTING_DEBUG
		if (MASTER(cr))
			printAllCells(step, cr->dd);

		printCell(step, cr);

		if (MASTER(cr))
			get_reference_cell_homes(srd, cr->dd, state_local->box);

		if (DOMAINDECOMP(cr))
			gmx_bcast_sim(srd->numCells * sizeof(int), srd->cellHomeRanks, cr);
		#endif

		wallcycle_start(wcycle, ewcMPC_CELLS);
		dd_srd_update(srd, cr);
		assign_collision_cells(srd);
		wallcycle_stop(wcycle, ewcMPC_CELLS);

		#ifdef SRD_ROUTING_DEBUG
		cc = 0;
		for(ci = 0; ci < srd->cellCount[XX]; ci++)
		{
			for(cj = 0; cj < srd->cellCount[YY]; cj++)
			{
				for(ck = 0; ck < srd->cellCount[ZZ]; ck++)
				{
					gmx_bool isLocal = srd->isLocalCell[cc];
					gmx_bool isLocal_check = srd->cellHomeRanks[cc] == cr->nodeid;
					if(isLocal != isLocal_check)
					{
						fprintf(srd->localLogFP, "<Step %"GMX_PRId64"> Inconsistent cell home %d (%d %d %d): %d vs %d (expected)\n",
							step, cc, ci, cj ,ck, isLocal, isLocal_check);
					//	srd->isLocalCell[cc] = isLocal_check;
					}
					cc++;
				}
			}
		}
		#endif
	}

	#ifdef SRD_COMM_TEST
	wallcycle_start(wcycle, ewcMPC_COMM_TEST);
	comm_test_update(srd, state_local, state_global, cr);
	wallcycle_stop(wcycle, ewcMPC_COMM_TEST);
	#endif

	// ======================================================================
	// collision step
	// ======================================================================

	if (DOMAINDECOMP(cr))
	{
		wallcycle_start(wcycle, ewcMPC_COMM1);
		dd_before_srd_collisions(srd, state_local, cr);
		wallcycle_stop(wcycle, ewcMPC_COMM1);
	}

	wallcycle_start(wcycle, ewcMPC_COLLISION);
	do_local_collisions(srd, state_local);
	wallcycle_stop(wcycle, ewcMPC_COLLISION);

	if (DOMAINDECOMP(cr))
	{
		wallcycle_start(wcycle, ewcMPC_COMM2);
		dd_after_srd_collisions(srd, state_local, cr);
		wallcycle_stop(wcycle, ewcMPC_COMM2);
	}

#ifdef SRD_COMM_TEST
	comm_test_check_results(srd, state_local, cr);
#endif

	// ======================================================================
	// output
	// ======================================================================

	if (srd->outputFrequency && step > 0)
	{
		wallcycle_start(wcycle, ewcMPC_OUTPUT);

		if (do_per_step(step, srd->samplingFrequency))
		{
			// get copy of all cell occupancies and velocities
			if (DOMAINDECOMP(cr))
			{
				// clear local junk data out before reducing
				for(ci = 0; ci < srd->numCells; ci++)
				{
					if(!srd->isLocalCell[ci])
					{
						clear_rvec(srd->cellMeanV[ci]);
						srd->cellOccupancy[ci] = 0;
					}
				}
				#ifdef GMX_MPI
				MPI_Reduce(srd->cellMeanV, srd->outVelBuf, DIM * srd->numCells, GMX_MPI_REAL, MPI_SUM, MASTERRANK(cr), cr->dd->mpi_comm_all);
				MPI_Reduce(srd->cellOccupancy, srd->outOccBuf, srd->numCells, MPI_INT, MPI_SUM, MASTERRANK(cr), cr->dd->mpi_comm_all);
				#endif
			} else {
				for(ci = 0; ci < srd->numCells; ci++)
				{
					copy_rvec(srd->cellMeanV[ci], srd->outVelBuf[ci]);
					srd->outOccBuf[ci] = srd->cellOccupancy[ci];
				}
			}

			// spatial average: buffer -> sum
			if (MASTER(cr))
				do_output_sample(srd);
		}

		if (MASTER(cr) && do_per_step(step, srd->outputFrequency))
			write_output_frame(srd);

		wallcycle_stop(wcycle, ewcMPC_OUTPUT);
	}
}

void cleanup_srd(t_srd *srd, t_commrec *cr)
{
	if (MASTER(cr) && srd->binaryOutputFile != NULL)
		finalize_output_file(srd);

#ifdef SRD_LOCAL_LOGS
	fclose(srd->localLogFP);
#endif
}

void init_srd_EXP(t_srd *srd, t_commrec *cr)
{
	int d, ci;

	for(d = 0; d < DIM; d++)
	{
		snew(srd->localCells[d], srd->cellCount[d]);
		snew(srd->sharedCellsB[d], srd->cellCount[d]);
		snew(srd->sharedCellsF[d], srd->cellCount[d]);
		srd->localCellCount[d] = 0;

		if (!DOMAINDECOMP(cr))
		{
			srd->localCellCount[d] = srd->cellCount[d];
			srd->sharedCellsB[d] = 0;
			srd->sharedCellsF[d] = 0;
			for (ci = 0; ci < srd->cellCount[d]; ci++)
				srd->localCells[d][ci] = ci;
		}
	}

	int initialBufferSize = ceil((float)srd->numCells / cr->nnodes);

	srd->sbuf_int = (srd_int_buf*)malloc(sizeof(srd_int_buf));
	init_srd_int_buffer(srd, srd->sbuf_int, "sbuf_int", 2 * initialBufferSize, 2 * srd->numCells);

	srd->rbuf_int = (srd_int_buf*)malloc(sizeof(srd_int_buf));
	init_srd_int_buffer(srd, srd->rbuf_int, "rbuf_int", 2 * initialBufferSize, 2 * srd->numCells);

	srd->sbuf_real = (srd_real_buf*)malloc(sizeof(srd_real_buf));
	init_srd_real_buffer(srd, srd->sbuf_real, "sbuf_real", 9 * initialBufferSize, 9 * srd->numCells);

	srd->rbuf_real = (srd_real_buf*)malloc(sizeof(srd_real_buf));
	init_srd_real_buffer(srd, srd->rbuf_real, "rbuf_real", 9 * initialBufferSize, 9 * srd->numCells);

	srd->sbuf_double = (srd_double_buf*)malloc(sizeof(srd_double_buf));
	init_srd_double_buffer(srd, srd->sbuf_double, "sbuf_double", 9 * initialBufferSize, 9 * srd->numCells);

	srd->rbuf_double = (srd_double_buf*)malloc(sizeof(srd_double_buf));
	init_srd_double_buffer(srd, srd->rbuf_double, "rbuf_double", 9 * initialBufferSize, 9 * srd->numCells);

	srd->bSumX    = (srd->empcRule == empcATA);
	srd->bSumV2   = (srd->empcRule == empcSRD && srd->enableThermostat);
	srd->bSumRand = (srd->empcRule == empcAT || srd->empcRule == empcATA);

	snew(srd->isOccupied, srd->numCells);

	// set phase 0 communication flags

	srd->flagsPhase0 = SRD_COMM_FLAGS_SUM_N | SRD_COMM_FLAGS_SUM_V;

	if (srd->bSumX)
		srd->flagsPhase0 |= SRD_COMM_FLAGS_SUM_X;

	if (srd->bSumRand)
		srd->flagsPhase0 |= SRD_COMM_FLAGS_SUM_RAND;

	// set phase 1 communication flags

	srd->flagsPhase1 = 0;

	if (srd->bSumV2)
		srd->flagsPhase1 |= SRD_COMM_FLAGS_SUM_V2;

	if (srd->empcRule == empcATA)
		srd->flagsPhase1 |= (SRD_COMM_FLAGS_SUM_DL | SRD_COMM_FLAGS_SUM_MIT);
}

// must be called before new cells are calculated for current update
static gmx_inline void clear_local_cells(t_srd *srd, t_commrec *cr)
{
	int ri, rj, rk;

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->localCells[ZZ][rk];

			srd->isLocalCell[ci] = !DOMAINDECOMP(cr);
			srd->isOccupied[ci] = FALSE;

			srd->cellOccupancy[ci] = 0;
			srd->cellMassTotal[ci] = 0.0;

			clear_rvec(srd->cellMeanV[ci]);

			if(srd->bSumX)
				clear_rvec(srd->cellMeanX[ci]);

			if(srd->bSumRand)
				clear_rvec(srd->cellRandomSum[ci]);

			if(srd->bSumV2)
				srd->cellVRel2[ci] = 0.0;

			if (srd->empcRule == empcATA)
			{
				clear_rvec(srd->cellDeltaL[ci]);
				clear_rvec(srd->cellDeltaOmega[ci]);
				clear_dmat(srd->cellInertiaTensor[ci]);
			}

			#ifdef SRD_ANGULAR_MOMENTUM_TEST
			clear_rvec(srd->initialAngularMomentum[ci]);
			clear_rvec(srd->finalAngularMomentum[ci]);
			#endif
		}
	}
}

/**
 * 	Sum relative velocity squared for local SRD particles
 */
static gmx_inline void do_srd_thermostat_1(t_srd *srd, rvec *v)
{
	int ai;

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0)
			continue;

		rvec relVel;
		rvec_sub(v[ai], srd->cellMeanV[cellID], relVel);
		srd->cellVRel2[cellID] += norm2(relVel);
	}
}

/**
 * Implementation of the thermostat described in Heyes 1983 (DOI: 10.1016/0301-0104(83)85235-5)
 * and updated in Hecht et al 2005 (DOI: 10.1103/PhysRevE.72.011408).
 * Operates on all colliding particles, assuming they all share the same mass.
 */
static gmx_inline void do_srd_thermostat_2(t_srd *srd)
{
	int ai;
	int ri, rj, rk;

	// calculate thermostat rescaling factor

	gmx_int64_t rngStart = srd->rngCount;

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->localCells[ZZ][rk];
			if(!srd->isOccupied[ci])
				continue;

			int rngCount = rngStart + 2*ci;

			double uniformRN[4];
			gmx_rng_cycle_2uniform(srd->step, rngCount, srd->rngSeed, RND_SEED_SRD, &uniformRN[0]);
			gmx_rng_cycle_2uniform(srd->step, rngCount + 1, srd->rngSeed, RND_SEED_SRD, &uniformRN[2]);

			#ifdef SRD_COMM_TEST
			// report the error, but do not propagate it
			if(!float_float_eq(srd->testCellVRel2[ci], srd->cellVRel2[ci], 1e-4)) // errors are compounded, more lenient "equality" check
			{
				printf("[%d] NOTICE: Inconsistent relative velocity squared [cell %2d] %f vs %f (correct)\n",
						srd->nodeID, ci, srd->cellVRel2[ci], srd->testCellVRel2[ci]);
				srd->cellVRel2[ci] = srd->testCellVRel2[ci];
			}
			#endif

			real vsqrSum = srd->cellVRel2[ci];
			int occupancy = srd->cellOccupancy[ci];

			if (occupancy > 0)
			{
				real power = 3 * (occupancy - 1);
				real thermostatScalingFactor = 1 + (uniformRN[0] * srd->thermostatStrength);

				if (uniformRN[1] < 0.5)
					thermostatScalingFactor = 1 / thermostatScalingFactor;

				real entropicPrefactor = pow(thermostatScalingFactor, power);

				real scalingFactorFactor = thermostatScalingFactor * thermostatScalingFactor - 1;
				real kB = 0.00831446214; // amu * (nm/ps)^2 / K
				real expPower = (srd->mass * scalingFactorFactor * vsqrSum) / (2 * kB * srd->thermostatTarget);

				real acceptanceProbability = entropicPrefactor * exp(-expPower);
				real p = uniformRN[2];

				srd->cellRescalingFactor[ci] = (p < acceptanceProbability) ? thermostatScalingFactor : 1.0;
			}

			#ifdef SRD_COMM_TEST
			if( !float_float_eq(srd->testCellRescalingFactor[ci], srd->cellRescalingFactor[ci], 1e-6))
			{
				// report the error, but do not propagate it
				printf("[%d] NOTICE: Inconsistent thermostat rescaling factor [cell %2d] %f vs %f\n",
						srd->nodeID, ci, srd->testCellRescalingFactor[ci], srd->cellRescalingFactor[ci]);
				srd->cellRescalingFactor[ci] = srd->testCellRescalingFactor[ci];
			}
			#endif
		}
	}

	srd->rngCount += 2 * srd->numCells;
}

/*
 * Chooses cell rotation axes for SRD cells
 */
static gmx_inline void do_srd_collisions_0(t_srd *srd)
{
	int ri, rj, rk;
	gmx_int64_t rngStart = srd->rngCount;

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->localCells[ZZ][rk];

			rvec gaussRN;
			gmx_rng_cycle_3gaussian_table(srd->step, srd->rngCount + ci, srd->rngSeed, RND_SEED_SRD, gaussRN);
			unitv(gaussRN, srd->cellRotationAxis[ci]);

			if(srd->enableThermostat)
			{
				srd->cellVRel2[ci] = 0;
				srd->cellRescalingFactor[ci] = 1.0;
			}
		}
	}

	srd->rngCount = rngStart + srd->numCells;
}


static gmx_inline void do_srd_collisions_1(t_srd *srd, rvec *v)
{
	if (srd->enableThermostat )
		do_srd_thermostat_1(srd, v);
}

static gmx_inline void do_srd_collisions_2(t_srd *srd, rvec *v)
{
	int ai, ci;

	if (srd->enableThermostat )
		do_srd_thermostat_2(srd);

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0)
			continue;

		rvec vrel;
		rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

		unitaxis_angle_rotate(vrel, srd->cellRotationAxis[cellID], srd->sinAngle, srd->cosAngle);

		if (srd->enableThermostat)
			svmul(srd->cellRescalingFactor[cellID], vrel, vrel);

		rvec_add(vrel, srd->cellMeanV[cellID], v[ai]);
	}
}

/**
 * Choose random velocity components for MPC particles
 */
static gmx_inline void do_andersen_collisions_0(t_srd *srd)
{
	int ai;

	realloc_srd_rvec_buffer(srd, srd->nativeVrandBuf, srd->homenr);

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0)
			continue;

		int globalAtomIndex = (srd->gatindex == NULL) ? ai : srd->gatindex[ai]; // support single core

		rvec vrand, gaussRN;
		gmx_rng_cycle_3gaussian_table(srd->step, globalAtomIndex, srd->rngSeed, RND_SEED_SRD, gaussRN);
		svmul(srd->mpcatVariance, gaussRN, vrand);

		rvec_inc(srd->cellRandomSum[cellID], vrand);
		copy_rvec(vrand, srd->nativeVrandBuf->buf[ai]);
	}
}

/**
 * If angular momentum conservation is on, get partial cell sums for moment of
 * intertia and change in angular momentum.
 */
void do_andersen_collisions_1(t_srd *srd, rvec *v)
{
	int ri, rj, rk;
	int ai, ci;

	if(srd->empcRule == empcATA)
	{
		rvec xrel, vrel, dv, dL;

		for (ai = 0; ai < srd->homenr; ai++)
		{
			int cellID = srd->cellID->buf[ai];
			if (cellID < 0)
				continue;

			rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);

			srd->cellInertiaTensor[cellID][XX][XX] += xrel[YY]*xrel[YY] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][YY] += xrel[XX]*xrel[XX] + xrel[ZZ]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][ZZ] += xrel[XX]*xrel[XX] + xrel[YY]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][YY] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][YY][XX] -= xrel[XX]*xrel[YY];
			srd->cellInertiaTensor[cellID][XX][ZZ] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][XX] -= xrel[XX]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][YY][ZZ] -= xrel[YY]*xrel[ZZ];
			srd->cellInertiaTensor[cellID][ZZ][YY] -= xrel[YY]*xrel[ZZ];

			// vrel = v - cell_mean(v)
			rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

			// dv = (vrand - cell_mean(vrand)) - vrel
			rvec_sub(srd->nativeVrandBuf->buf[ai], srd->cellRandomSum[cellID], dv);
			rvec_dec(dv, vrel);

			cprod(xrel, dv, dL);
			rvec_inc(srd->cellDeltaL[cellID], dL);
		}
	}
}

/**
 * Set the resultant velocities
 */
void do_andersen_collisions_2(t_srd *srd, rvec *v)
{
	int ri, rj, rk;
	int ai, ci;

	if(srd->empcRule == empcATA)
	{
		for (ri = 0; ri < srd->localCellCount[XX]; ri++)
		for (rj = 0; rj < srd->localCellCount[YY]; rj++)
		{
			int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
			for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
			{
				int ci = cxy + srd->localCells[ZZ][rk];

				if (srd->cellOccupancy[ci] < 2)
					continue;

				dmatrix invInertiaTensor;
				gmx_bool success = dm_inv(srd->cellInertiaTensor[ci], invInertiaTensor);

				if(success)
					dmvmul(invInertiaTensor, srd->cellDeltaL[ci], srd->cellDeltaOmega[ci]);
				else
				{
					printf("ERROR: Cannot invert moment of inertia tensor!");
					printf("%f %f %f\n", srd->cellInertiaTensor[ci][XX][XX], srd->cellInertiaTensor[ci][XX][YY], srd->cellInertiaTensor[ci][XX][ZZ]);
					printf("%f %f %f\n", srd->cellInertiaTensor[ci][YY][XX], srd->cellInertiaTensor[ci][YY][YY], srd->cellInertiaTensor[ci][YY][ZZ]);
					printf("%f %f %f\n", srd->cellInertiaTensor[ci][ZZ][XX], srd->cellInertiaTensor[ci][ZZ][YY], srd->cellInertiaTensor[ci][ZZ][ZZ]);
					printf("determinant = %f\n", ddet(srd->cellInertiaTensor[ci]));
					printf("cell occupancy = %d\n", srd->cellOccupancy[ci]);
					rvec xrel, vrel;

					for (ai = 0; ai < srd->homenr; ai++)
					{
						int cellID = srd->cellID->buf[ai];
						if (cellID < 0 || !srd->isLocalCell[cellID] || cellID != ci)
							continue;

						rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);
						rvec_sub(v[ai], srd->cellMeanV[cellID], vrel);

						printf("%f %f %f -- %f %f %f\n", xrel[XX], xrel[YY], xrel[ZZ], vrel[XX], vrel[YY], vrel[ZZ]);
					}

					for (ai = 0; ai < srd->adoptedCount; ai++)
					{
						int cellID = srd->adoptionBuffer->M[ai][MD_CELL];
						if (!srd->isLocalCell[cellID])
							gmx_fatal(FARGS, "SRD particle adopted by incorrect node!\n");

						if(cellID != ci)
							continue;

						rvec_sub(srd->adoptionBuffer->X[ai], srd->cellMeanX[cellID], xrel);
						rvec_sub(srd->adoptionBuffer->V[ai], srd->cellMeanV[cellID], vrel);

						printf("%f %f %f -- %f %f %f\n", xrel[XX], xrel[YY], xrel[ZZ], vrel[XX], vrel[YY], vrel[ZZ]);
					}
				}
			}
		}
	}

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0)
			continue;

		rvec_add(srd->cellMeanV[cellID], srd->nativeVrandBuf->buf[ai], v[ai]);

		// remove random drift
		rvec_dec(v[ai], srd->cellRandomSum[cellID]);

		// remove added angular momentum
		if(srd->empcRule == empcATA)
		{
			rvec xrel, angularMomentumCorrection;
			rvec_sub(srd->relXBuf->buf[ai], srd->cellMeanX[cellID], xrel);

			cprod(srd->cellDeltaOmega[cellID], xrel, angularMomentumCorrection);
			rvec_dec(v[ai], angularMomentumCorrection);
		}
	}
}

void update_srd_EXP(
		t_srd				*srd,
		gmx_int64_t			 step,
		t_state				*state_local,
		gmx_unused t_state	*state_global,
		t_mdatoms			*mdatoms,
		t_commrec			*cr,
		gmx_wallcycle_t		 wcycle)
{
	int ai, d;
	int ri, rj, rk;

	if (!do_per_step(step, srd->collisionFrequency))
		return;

	srd->step = step;
	srd->rngCount = 0;
	srd->homenr = mdatoms->homenr;
	srd->gatindex = (DOMAINDECOMP(cr)) ? cr->dd->gatindex : NULL; // accommodate singe-core

	// ======================================================================
	// initialization
	// ======================================================================

	wallcycle_start(wcycle, ewcMPC_INIT);

	clear_local_cells(srd, cr);

	// update collision cell size to compensate for changes in box size.
	// the number of cells are fixed during initialization.
	for(d = 0; d < DIM; d++)
	{
		srd->boxSize[d] = state_local->box[d][d];
		srd->cellSize[d] = srd->boxSize[d] / srd->cellCount[d];
		srd->gridShift[d] = (gmx_rng_uniform_real(srd->rng) - 0.5) * srd->cellSize[d];
	}

	assign_atoms_to_cells(srd, state_local);

	// ======================================================================
	// get local cell range
	// ======================================================================

	// sets srd->localCells based on dd zone
	if (DOMAINDECOMP(cr))
		dd_srd_find_local_collision_cells(srd, cr);

	#ifdef SRD_COMM_TEST
	wallcycle_start(wcycle, ewcMPC_COMM_TEST);
	comm_test_update(srd, state_local, state_global, cr);
	wallcycle_stop(wcycle, ewcMPC_COMM_TEST);
	#endif

	// ======================================================================
	// accumulate partial sums and ensure all particles are in a local cell
	// ======================================================================

	for (ai = 0; ai < srd->homenr; ai++)
	{
		int cellID = srd->cellID->buf[ai];
		if (cellID < 0)
			continue;

		if (!srd->isLocalCell[cellID])
			ERROR_LOST_PARTICLE(srd, cr, state_local, ai, cellID);

		srd->isOccupied[cellID] = TRUE;
		srd->cellOccupancy[cellID]++;
		rvec_inc(srd->cellMeanV[cellID], state_local->v[ai]);

		if(srd->bSumX)
			rvec_inc(srd->cellMeanX[cellID], srd->relXBuf->buf[ai]);
	}

	wallcycle_stop(wcycle, ewcMPC_INIT);

	// ======================================================================
	// COLLISIONS PHASE 0
	// only things that don't require cell sums
	// ======================================================================

	wallcycle_start(wcycle, ewcMPC_COLLISION);

	switch(srd->empcRule)
	{
	case empcSRD:
		do_srd_collisions_0(srd);
		break;
	case empcAT:
	case empcATA:
		do_andersen_collisions_0(srd);
		break;
	}

	// ======================================================================
	// phase 0 communication
	// exchange partial sums for N, X, V, and randV
	// ======================================================================

	if (DOMAINDECOMP(cr))
	{
		wallcycle_stop(wcycle, ewcMPC_COLLISION);
		wallcycle_start(wcycle, ewcMPC_COMM1);

		dd_srd_comm(srd, cr->dd, srd->flagsPhase0);

		wallcycle_stop(wcycle, ewcMPC_COMM1);
		wallcycle_start_nocount(wcycle, ewcMPC_COLLISION);
	}

	// ======================================================================
	// get cell averages from partial sums
	// ======================================================================

	for (ri = 0; ri < srd->localCellCount[XX]; ri++)
	for (rj = 0; rj < srd->localCellCount[YY]; rj++)
	{
		int cxy = (srd->cellCount[ZZ] * ((srd->cellCount[YY] * srd->localCells[XX][ri]) + srd->localCells[YY][rj]));
		for (rk = 0; rk < srd->localCellCount[ZZ]; rk++)
		{
			int ci = cxy + srd->localCells[ZZ][rk];
			if(!srd->isOccupied[ci])
				continue;

			int n = srd->cellOccupancy[ci];
			if (n > 1)
			{
				real n_inv = 1.0 / n;
				svmul(n_inv, srd->cellMeanV[ci], srd->cellMeanV[ci]);
				if (srd->bSumX)
					svmul(n_inv, srd->cellMeanX[ci], srd->cellMeanX[ci]);
				if (srd->bSumRand)
					svmul(n_inv, srd->cellRandomSum[ci], srd->cellRandomSum[ci]);
			}
		}
	}

	// ======================================================================
	// COLLISIONS PHASE 1
	// ======================================================================

	#ifdef SRD_ANGULAR_MOMENTUM_TEST
	calculate_angular_momentum_EXP(srd, state_local, srd->initialAngularMomentum);
	if (DOMAINDECOMP(cr))
		dd_srd_comm(srd, cr->dd, SRD_COMM_FLAGS_SUM_LI);
	#endif

	switch(srd->empcRule)
	{
	case empcSRD:
		do_srd_collisions_1(srd, state_local->v);
		break;
	case empcAT:
	case empcATA:
		do_andersen_collisions_1(srd, state_local->v);
		break;
	}

	// ======================================================================
	// phase 1 communication
	// exchange partial sums for Vrel2, dL, and MI
	// ======================================================================

	if (DOMAINDECOMP(cr))
	{
		wallcycle_stop(wcycle, ewcMPC_COLLISION);
		wallcycle_start(wcycle, ewcMPC_COMM2);

		dd_srd_comm(srd, cr->dd, srd->flagsPhase1);

		wallcycle_stop(wcycle, ewcMPC_COMM2);
		wallcycle_start_nocount(wcycle, ewcMPC_COLLISION);
	}

	// ======================================================================
	// COLLISIONS PHASE 2
	// finish collision step
	// ======================================================================

	switch(srd->empcRule)
	{
	case empcSRD:
		do_srd_collisions_2(srd, state_local->v);
		break;
	case empcAT:
	case empcATA:
		do_andersen_collisions_2(srd, state_local->v);
		break;
	}

	wallcycle_stop(wcycle, ewcMPC_COLLISION);

	#ifdef SRD_ANGULAR_MOMENTUM_TEST
	calculate_angular_momentum_EXP(srd, state_local, srd->finalAngularMomentum);
	if (DOMAINDECOMP(cr))
		dd_srd_comm(srd, cr->dd, SRD_COMM_FLAGS_SUM_LF);
	check_angular_momentum_EXP(srd);
	#endif

	#ifdef SRD_COMM_TEST
		comm_test_check_results_EXP(srd, state_local);
	#endif
}
