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

#ifndef SRD_H_
#define SRD_H_

//#define SRD_LOCAL_LOGS			ENABLED // enables the creation of log files for each rank
//#define SRD_COMM_TEST				ENABLED // checks result of SRD collisions using optimized vs global communication
//#define SRD_ROUTING_DEBUG			ENABLED // checks the routing using global communication -- only works with static load balancing!
//#define SRD_COMM_DEBUG			ENABLED // prints debugging information for optimized communication scheme
//#define SRD_ANGULAR_MOMENTUM_TEST	ENABLED // calculates angular momentum for each cell before and after the collision step
//#define SRD_PRINT_BUFFER_INFO		ENABLED // prints to the output file when buffers are enlarged

#include "gromacs/random/random.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/fileio/gmxfio.h"

#include "gromacs/utility/gmxmpi.h"
#include "gromacs/legacyheaders/types/commrec.h"
#include "vec.h"

#if defined(SRD_COMM_TEST) || defined(SRD_COMM_DEBUG) || defined(SRD_ROUTING_DEBUG) || defined(SRD_ANGULAR_MOMENTUM_TEST)
#define SRD_LOCAL_LOGS ENABLED
#endif

#define RND_SEED_SRD 13

#define SRD_NO_NEIGHBOR -1

// flags used by the experimental communication scheme
#define SRD_COMM_FLAGS_SUM_N      (1 << 0)
#define SRD_COMM_FLAGS_SUM_X      (1 << 1)
#define SRD_COMM_FLAGS_SUM_V      (1 << 2)
#define SRD_COMM_FLAGS_SUM_V2     (1 << 3)
#define SRD_COMM_FLAGS_SUM_RAND   (1 << 4)
#define SRD_COMM_FLAGS_SUM_DL     (1 << 5)
#define SRD_COMM_FLAGS_SUM_MIT    (1 << 6)
#define SRD_COMM_FLAGS_SUM_LI     (1 << 7)
#define SRD_COMM_FLAGS_SUM_LF     (1 << 8)

#define SRD_COMM_FLAGS_REAL    (SRD_COMM_FLAGS_SUM_X \
		| SRD_COMM_FLAGS_SUM_V | SRD_COMM_FLAGS_SUM_V2 | SRD_COMM_FLAGS_SUM_RAND \
		| SRD_COMM_FLAGS_SUM_DL | SRD_COMM_FLAGS_SUM_LI | SRD_COMM_FLAGS_SUM_LF)

#define SRD_COMM_FLAGS_DOUBLE  (SRD_COMM_FLAGS_SUM_MIT)

// moment of inertia tensor sometimes needs the extra precision
typedef double dmatrix[DIM][DIM];

static gmx_inline void copy_dmat(dmatrix a, dmatrix b)
{
	copy_dvec(a[XX], b[XX]);
	copy_dvec(a[YY], b[YY]);
	copy_dvec(a[ZZ], b[ZZ]);
}

static gmx_inline void dmat_inc(dmatrix a, dmatrix b)
{
	dvec_inc(a[XX], b[XX]);
	dvec_inc(a[YY], b[YY]);
	dvec_inc(a[ZZ], b[ZZ]);
}

// metadata stored in buffer
typedef int	srd_mdata[4];

#define MD_NODE  0  // home node (cr->nodeid) -- used to return atoms after collisions
#define MD_GNDX  1  // global atom index -- used to uniquely identify atoms during collision step
#define MD_CELL  2  // srd cell index -- used to send atoms before collisions
#define MD_LNDX  3  // local atom index

static gmx_inline void copy_srd_mdata(const srd_mdata a, srd_mdata b)
{
    b[MD_NODE] = a[MD_NODE];
    b[MD_GNDX] = a[MD_GNDX];
    b[MD_CELL] = a[MD_CELL];
    b[MD_LNDX] = a[MD_LNDX];
}

typedef struct
{
	rvec *X; // particle positions
	rvec *V; // particle velocities
	srd_mdata *M; // metadata for communication
	int capacity;
	int maxCapacity;
	char *name;
} srd_comm_buf;

typedef struct
{
	rvec *buf;
	int capacity;
	int maxCapacity;
	char *name;
} srd_rvec_buf;

typedef struct
{
	int *buf;
	int capacity;
	int maxCapacity;
	char *name;
} srd_int_buf;

typedef struct
{
	real *buf;
	int capacity;
	int maxCapacity;
	char *name;
} srd_real_buf;

typedef struct
{
	double *buf;
	int capacity;
	int maxCapacity;
	char *name;
} srd_double_buf;

typedef struct
{
	int empcRule;
	int commVer;

	ivec cellCount;
	int numCells;

	rvec boxSize;
	rvec cellSize;
	rvec gridShift;

	real collisionAngle;
	real sinAngle;
	real cosAngle;

	int collisionFrequency;

	gmx_bool bCommunicateX;

	// copy these to avoid passing around their data structures
	gmx_bool bMaster;	/* copied from commrec during init */
	int nodeID;			/* copied from commrec during init */
	int homenr;			/* copied from mdatoms every update */
	int *gatindex;		/* copied from cr->dd when dd is enabled, NULL otherwise */
	real delta_t;
	gmx_int64_t step;

	// srd particle info
	unsigned short type;
	real mass;

	// atom data
	gmx_bool     *isSRD_gl;
	srd_int_buf  *cellID;
	srd_rvec_buf *relXBuf;

	// general cell data
	int  *cellOccupancy;
	real *cellMassTotal; // unused
	rvec *cellMeanV;
	rvec *cellMeanX;

	// ======================================================================
	// collisions
	// ======================================================================

	// mpc random numbers
	gmx_int64_t rngSeed;
	gmx_int64_t rngCount;
	gmx_rng_t rng;

	// srd collisions
	rvec *cellRotationAxis;
	real *cellVRel2;
	real *cellRescalingFactor;

	// srd cell thermostat
	gmx_bool enableThermostat;
	real thermostatStrength;
	real thermostatTarget;

	// andersen collisions
	real mpcatVariance;
	rvec *cellRandomSum;
	srd_rvec_buf *nativeVrandBuf;
	srd_rvec_buf *adoptedVrandBuf;

	// andersen collisions with angular momentum conservation
	dmatrix *cellInertiaTensor;
	rvec *cellDeltaL;
	rvec *cellDeltaOmega;

	// angular momentum test
	rvec *initialAngularMomentum;
	rvec *finalAngularMomentum;

	// ======================================================================
	// cell data output
	// ======================================================================

	int outputFrequency;
	int samplingFrequency;
	gmx_bool writeCellVectors;
	gmx_bool writeCellOccupancy;
	t_fileio *binaryOutputFile;
	gmx_off_t outFrameCountPos;

	rvec *outVelBuf;
	int *outOccBuf;

	dvec *outVelSum;
	double *outOccSum;
	dvec outSizeSum;
	int outFrameCount;

	// ======================================================================
	// general communication
	// ======================================================================

	int neighbors[3][3][3];
	rvec boundaries[3][3][3][2];
	ivec cellRange[3][3][3][2];     // ni, nj, nk, min/max
	ivec prevCellRange[3][3][3][2]; // ni, nj, nk, min/max

	int *nodeRouting; // nodeID -> required pulseID
	int *cellRouting; // cellID -> required pulseID

	gmx_bool *isLocalCell;

	// ======================================================================
	// default communication scheme
	// ======================================================================

	// for faster cell iteration, keep track of home cells
	ivec homeCellCount;
	int *homeCells[3];

	srd_comm_buf *sendBuffer;
	srd_comm_buf *recvBuffer;
	srd_comm_buf *dispatchBuffer;
	srd_comm_buf *adoptionBuffer;

	gmx_bool *dispatched;
	int ndispatch;
	int adoptedCount;

	// ======================================================================
	// experimental communication scheme
	// ======================================================================

	gmx_bool *isOccupied;	// does this cell contain at least 1 home atom?

	// cell ranges used for neighboring cell communication protocol
	// sharedF and sharedB refer to "Forward" and "Backward" in the sense of
	// the domain decomposition. the local cell range encompasses all shared
	// cells in addition to those cells entirely "native" to this zone.

	int localCellCount[3];
	int *localCells[3];

	int sharedCellCountB[3];
	int *sharedCellsB[3];

	int sharedCellCountF[3];
	int *sharedCellsF[3];

	gmx_bool bSumX;
	gmx_bool bSumV2;
	gmx_bool bSumRand;
	int flagsPhase0, flagsPhase1;

	srd_int_buf    *sbuf_int,    *rbuf_int;
	srd_real_buf   *sbuf_real,   *rbuf_real;
	srd_double_buf *sbuf_double, *rbuf_double;

	int isend, irecv;
	int rsend, rrecv;
	int dsend, drecv;

	// ======================================================================
	// testing
	// ======================================================================

#ifdef SRD_COMM_TEST
	rvec *testX;  // buffer to collect state->x on master
	rvec *testV;  // buffer to collect state->v on master
	rvec *testVp; // buffer to scatter v' from master
	int  *testCellID;
	rvec *testXRel;
	gmx_int64_t testRngCount;
	int  *testCellN;
	real *testCellM;
	rvec *testCellX;
	rvec *testCellV;
	rvec *testCellRotationAxis;
	real *testCellVRel2;
	real *testCellRescalingFactor;
	rvec *testCellRandomSum;
	rvec *testParticleVRand;
	rvec *testCellDeltaL;
	rvec *testCellDeltaOmega;
	dmatrix *testCellInertiaTensor;
#endif

#ifdef SRD_COMM_DEBUG
	int totalSent;
	int totalReturned;
	int *atomState;
#define STATE_SENT 1
#define STATE_RETURNED 2
#endif

#ifdef SRD_ROUTING_DEBUG
	int *cellHomeRanks;
#endif

#ifdef SRD_LOCAL_LOGS
	FILE *localLogFP;
#endif
} t_srd;

void init_srd(
		FILE				*fplog,
		t_srd				*srd,
		t_state				*state_local,
		t_state				*state_global,
		gmx_mtop_t			*mtop,
		t_inputrec			*ir,
		t_commrec			*cr);

void update_srd(
		t_srd				*srd,
		gmx_int64_t			 step,
		t_state				*state_local,
		gmx_unused t_state	*state_global,
		t_mdatoms			*mdatoms,
		t_commrec			*cr,
		gmx_wallcycle_t		 wcycle);

void cleanup_srd(t_srd *srd, t_commrec *cr);

gmx_bool realloc_srd_comm_buffer(t_srd *srd, srd_comm_buf *buffer, int size);
gmx_bool realloc_srd_rvec_buffer(t_srd *srd, srd_rvec_buf *buffer, int size);
gmx_bool realloc_srd_int_buffer(t_srd *srd, srd_int_buf *buffer, int size);
gmx_bool realloc_srd_real_buffer(t_srd *srd, srd_real_buf *buffer, int size);
gmx_bool realloc_srd_double_buffer(t_srd *srd, srd_double_buf *buffer, int size);

void getCellIndicies(t_srd *srd, int cellID, int *ci, int *cj, int *ck);

#endif
