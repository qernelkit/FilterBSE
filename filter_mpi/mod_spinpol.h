#include "fd.h"
#include "mod_pseudopot.h"
#include "mod_filter.h"
#include "filter.h"
#include "mod_ortho.h"
#include "mod_diag.h"
#include "mod_sigma.h"
#include "mod_output.h"
#include "mod_optional_output.h"

/*****************************************************************************/
/*
 * Collinear spin-polarized driver (non-periodic path).
 *
 * A spin-polarized run builds two independent scalar Hamiltonians, one per spin
 * channel, that differ only in their local pseudopotential (pot<X>_up.par vs
 * pot<X>_dn.par). Because the semiempirical potentials are fixed (no
 * self-consistency), the two channels are completely independent: run_spinpol
 * simply performs the full filter -> gather -> ortho -> diag -> sigma -> output
 * pipeline once for spin up and once for spin down, tagging each channel's output
 * files with "_up" / "_dn" (eval_up.dat / eval_dn.dat, etc.).
 *
 * The shared non-local / spin-orbit projectors are built once by mod_pseudopot
 * before this is called; only the local potential is rebuilt per channel here.
 */
void run_spinpol(
    zomplex*      psi,
    zomplex*      phi,
    double*       pot_local,
    pot_st*       pot,
    xyz_st*       R,
    atom_info*    atom,
    grid_st*      grid,
    zomplex*      LS,
    nlc_st*       nlc,
    long*         nl,
    zomplex*      an,
    double*       zn,
    double*       ene_targets,
    double*       ksqr,
    lattice_st*   lattice,
    vector*       G_vecs,
    vector*       k_vecs,
    double*       eig_vals,
    double*       sigma_E,
    index_st*     ist,
    par_st*       par,
    flag_st*      flag,
    parallel_st*  parallel);

/*****************************************************************************/
