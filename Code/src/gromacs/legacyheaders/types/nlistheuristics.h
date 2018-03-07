/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2011,2013, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef _nlistheuristics_h
#define _nlistheuristics_h

#ifdef __cplusplus
extern "C" {
#endif

#if 0
}
/* Hack to make automatic indenting work */
#endif

typedef struct {
    gmx_bool        bGStatEveryStep;
    gmx_int64_t     step_ns;
    gmx_int64_t     step_nscheck;
    gmx_int64_t     nns;
    matrix          scale_tot;
    int             nabnsb;
    double          s1;
    double          s2;
    double          ab;
    double          lt_runav;
    double          lt_runav2;
} gmx_nlheur_t;

void reset_nlistheuristics(gmx_nlheur_t *nlh, gmx_int64_t step);

void init_nlistheuristics(gmx_nlheur_t *nlh,
                          gmx_bool bGStatEveryStep, gmx_int64_t step);

void update_nliststatistics(gmx_nlheur_t *nlh, gmx_int64_t step);

void set_nlistheuristics(gmx_nlheur_t *nlh, gmx_bool bReset, gmx_int64_t step);

#ifdef __cplusplus
}
#endif

#endif
