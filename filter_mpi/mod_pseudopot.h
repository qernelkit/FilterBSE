#include "fd.h"
#include "init.h"
#include "hamiltonian.h"

void mod_pseudopot(
    double*       pot_local,
    pot_st*       pot,
    xyz_st*       R,
    atom_info*    atom,
    grid_st*      grid,
    zomplex*      LS,
    nlc_st*       nlc,
    long*         nl,
    double*       SO_projectors,
    index_st*     ist,
    par_st*       par,
    flag_st*      flag,
    parallel_st*  parallel);

// Allocate the local-pot read buffers, build the local potential on the grid for
// the current configuration (par->spin_channel selects the spin-up/down pot file
// in a spin-polarized run), write its cube file, and free the read buffers.
void build_local_pot_channel(
    double*       pot_local,
    pot_st*       pot,
    xyz_st*       R,
    atom_info*    atom,
    grid_st*      grid,
    index_st*     ist,
    par_st*       par,
    flag_st*      flag,
    parallel_st*  parallel);



