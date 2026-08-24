#include "mod_pseudopot.h"

void mod_pseudopot(
  double*       pot_local,
  pot_st*       pot,
  xyz_st*       R,
  atom_info*    atom,
  grid_st*      grid,
  zomplex*      LS,
  nlc_st*       nlc,
  long*         nl,
  double*       SO_projs,
  index_st*     ist,
  par_st*       par,
  flag_st*      flag,
  parallel_st*  parallel){

  
  /************************************************************/
  /*******************  DECLARE VARIABLES   *******************/
  /************************************************************/

  const int mpir = parallel->mpi_rank;

  if (mpir == 0) printf("\nInitializing potentials...\n");

  /************************************************************/
  /*******************    LOCAL POTENTIAL   *******************/
  /************************************************************/

  // In a collinear spin-polarized run the local potential differs between the two
  // spin channels, so it is built (per channel) by run_spinpol via
  // build_local_pot_channel; here we build only the shared non-local/spin-orbit
  // projectors below. In a normal run we build the single local potential now.
  if (0 == flag->spinPolarized)
  {
    build_local_pot_channel(pot_local, pot, R, atom, grid, ist, par, flag, parallel);
  }

  /************************************************************/
  /*****************    SPIN-ORBIT/NL POT    ******************/
  /************************************************************/

  // Spin-orbit and non-local are independent. The spin-orbit radial projectors
  // (and the L.S coupling matrix) are only built when SO is on, but the grid
  // projector table (init_NL_projectors) is needed whenever EITHER potential is
  // active: it fills both the SO grid projectors (.proj, scaled by SO_par, used
  // only by the spin-orbit operator) and the scalar non-local grid projectors
  // (.NL_proj, used only by the non-local operator). When SO is off, SO_projs is
  // zero so the unused .proj entries vanish harmlessly.
  if(flag->SO==1) {
    if (mpir == 0) printf("\nSpin-orbit pseudopotential:\n");

    init_SO_projectors(SO_projs, grid, R, atom, ist, par, flag, parallel);

    def_LS(LS, ist, par);

    if (mpir == 0) printf("\tSO projectors generated.\n");
  }

  if ( (flag->SO == 1) || (flag->NL == 1) ){
    if (mpir == 0) printf("\nNon-local pseudopotential:\n"); fflush(0);

    init_NL_projectors(nlc, nl, SO_projs, grid, R, atom, ist, par, flag, parallel);

    if (mpir == 0) printf("\tNL projectors generated.\n");
  }
  
  /************************************************************/
  /*******************    FREE POT MEMORY   *******************/
  /************************************************************/

  // The local-potential arrays (pot->r, pot->pseudo, ...) are allocated and freed
  // inside build_local_pot_channel, so there is nothing to free here for them.

  // SO/NL pots
  // free memory allocated to SO_projectors
  if ( (flag->SO == 1) || (flag->NL == 1) ){
    free(SO_projs); SO_projs = NULL;
  }

  return;
}

/*****************************************************************************/

void build_local_pot_channel(
  double*       pot_local,
  pot_st*       pot,
  xyz_st*       R,
  atom_info*    atom,
  grid_st*      grid,
  index_st*     ist,
  par_st*       par,
  flag_st*      flag,
  parallel_st*  parallel){

  /*******************************************************************
   * Allocate the local-pseudopotential read buffers, build the local *
   * potential on the grid for the currently selected configuration    *
   * (par->spin_channel picks the spin-up/spin-down pot file in a       *
   * spin-polarized run), write the cube file, then free the read       *
   * buffers. Factored out of mod_pseudopot so run_spinpol can rebuild   *
   * the local potential for each spin channel while the shared          *
   * non-local/spin-orbit projectors are built only once.               *
   ********************************************************************/

  const int atyp_tot = ist->ngeoms * ist->n_atom_types;
  const int potfl_tot = atyp_tot * ist->max_pot_file_len;
  const int mpir = parallel->mpi_rank;

  if (mpir == 0) printf("\nLocal pseudopotential:\n");

  // Alloc mem for reading the atomic potentials
  ALLOCATE(&(pot->file_lens), atyp_tot, "pot->file_lens");
  ALLOCATE(&(pot->dr), atyp_tot, "pot->dr");
  ALLOCATE(&(pot->r), potfl_tot, "pot->r");
  ALLOCATE(&(pot->pseudo), potfl_tot, "pot->pseudo");
  if (1.0 != par->scale_surface_Cs){
    // allocate mem for separate LR potentials
    // if the surface atoms will be charge balanced
    ALLOCATE(&(pot->r_LR), potfl_tot, "pot->r_LR");
    ALLOCATE(&(pot->pseudo_LR), potfl_tot, "pot->pseudo_LR");
  }

  // Build local potential on the grid
  build_local_pot(pot_local, pot, R, atom, grid, ist, par, flag, parallel);

  // Tag the cube file per spin channel (file_tag is "" for a normal run).
  char cube_name[64];
  snprintf(cube_name, sizeof(cube_name), "local-pot%s.cube", par->file_tag);
  write_cube_file(pot_local, grid, cube_name);

  // Free the local-potential read buffers
  free(pot->r); pot->r = NULL;
  free(pot->pseudo); pot->pseudo = NULL;
  free(pot->dr); pot->dr = NULL;
  free(pot->file_lens); pot->file_lens = NULL;
  if (1.0 != par->scale_surface_Cs){
    free(pot->r_LR); pot->r_LR = NULL;
    free(pot->pseudo_LR); pot->pseudo_LR = NULL;
  }

  return;
}
