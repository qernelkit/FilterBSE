/************************************************************/
#pragma once
/************************************************************/
/*******************    REQ'D LIBRARIES   *******************/
/************************************************************/

#include <stdio.h>
#include <string.h>
#include "unistd.h"
#include <math.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <time.h>
#include <assert.h>
#include <fftw3.h>
#include <omp.h>
#include <mkl.h>
#include "vector.h"
#include <limits.h>
#include <complex.h>
#include <mpi.h>

/************************************************************/

/************************************************************/
/*******************    CUSTOM STRUCTS    *******************/
/************************************************************/

typedef struct flag
{
  // Configuration flags
  int centerConf;
  int setTargets;
  int setSeed;
  int interpolatePot;
  int useStrain;

  // Spin and interaction settings
  int SO;
  int NL;
  int LR;
  int LSD;
  int useSpinors;
  int isComplex;

  // Input control
  int inputPsiFilt;
  int readProj;
  int readCoeffs;

  // Output control
  int printPsiFilt;
  int printPsiDiag;
  int printPsiOrtho;
  int printNorm;
  int printCubes;
  int printGaussCubes;

  // Computation settings
  int calcPotOverlap;
  int getAllStates;
  int timeHamiltonian;
  int calcSpinAngStat;

  // Restart and checkpointing
  int retryFilter;
  int alreadyTried;
  int restartFromOrtho;
  int restartFromSigma;
  int saveCheckpoints;
  int restartFromCheckpoint;
  int saveOutput;

  // Special modes
  int calcFilterOnly;
  int approxEnergyRange;
  int noTimeRev;

  // Performance and parallelization
  int useFastHam;
  int useMPIOMP;
  int MPIOrtho;
  int MPIDiag;
  int parallelSigma;

  // Structural settings
  int periodic;
  int useGaussianBasis;
  int readKPath;

  // Collinear spin polarization (total magnetization): two independent scalar
  // Hamiltonians (spin up / spin down) built from two local pseudopotentials and
  // filter-ortho-diagonalized separately. Mutually exclusive with spin-orbit /
  // spinors (SO couples the spin channels; collinear polarization keeps them
  // separate). Non-local is allowed and shared between channels.
  int spinPolarized;
} flag_st;

/************************************************************/
/************************************************************/

typedef struct index
{
  // State filtering and orthogonalization
  long m_states_per_filter;
  long n_filter_cycles;
  long mn_states_tot;
  long mn_states_per_k;     // periodic: filtered states per k before ortho (= n_filter_cycles * m_states_per_filter)
  long n_filters_per_rank;
  long n_states_per_rank;
  long n_states_for_ortho;
  long n_states_for_sigma;
  long long psi_rank_size;
  long init_psi_start;
  long init_psi_end;

  // HOMO-LUMO and energy state indices
  long homo_idx;
  long lumo_idx;
  long total_homo;
  long total_lumo;

  // Grid and atomic system sizes
  long ngrid;
  long nspinngrid;
  long ncheby;
  long natoms;
  long n_atom_types;
  long n_max_atom_types;

  // Atomic type and potential information
  long *atom_types;
  long max_pot_file_len;
  long n_NL_gridpts;
  long n_NL_atoms;
  long nproj;

  // Computational parameters
  int nspin;
  int ncubes;
  int n_gcubes_s;
  int n_gcubes_e;
  int ngeoms;
  int complex_idx;

  // Crystal and material properties
  int crystal_structure_int;
  int outmost_material_int;

  // Angular momentum states
  int n_s_ang_mom;
  int n_l_ang_mom;
  int n_j_ang_mom;

  // Band structure and wavevector information
  int n_bands;
  int n_G_vecs;
  int n_G_zeros;
  int n_k_pts;
  int nk1;
  int nk2;
  int nk3;

  // Redundant parameters (possibly for convenience)
  long nx;
  long ny;
  long nz;
  long nthreads;
} index_st;

/************************************************************/
/************************************************************/

typedef struct par
{
  // Potential energy ranges
  double Vmin;
  double Vmax;
  double VBmin;
  double VBmax;
  double CBmin;
  double CBmax;

  // Surface scaling factor
  double scale_surface_Cs;

  // Target states for valence and conduction bands
  long n_targets_VB;
  long n_targets_CB;

  // Random seed
  long rand_seed;

  // Energy and time parameters
  double KE_max;
  double fermi_E;
  double dt;
  double dE;
  double dE_1;

  // Nonlocal potential cutoff and energy cutoffs
  double R_NLcut2;
  double sigma_E_cut;

  // >>> NL projector spectral-bound estimate (remove with the matching blocks in
  // init.c:init_NL_projectors and energy.c:get_energy_range[_k]) >>>
  // Extreme eigenvalues (Ha) of the non-local (KB) operator, used to pad the
  // approximate energy range physically. Default 0 => no NL pad added.
  double NL_lambda_min;
  double NL_lambda_max;
  // <<< end NL projector spectral-bound estimate <<<

  // Time reversal and checkpointing
  int t_rev_factor;
  int checkpoint_id;

  // Crystal and material properties
  char crystal_structure[15];
  char outmost_material[15];

  // FFT settings
  char fft_wisdom_dir[200];
  char fftw_wisdom[250];

  // Numerical precision settings
  double psi_zero_cut;

  // Parallelization
  int ham_threads;

  // Orbital and basis settings
  int n_orbitals;
  int n_orbitals_per_atom;
  int n_gauss_per_orbital;

  // Gaussian integral cutoff
  double R_gint_cut2;

  // Potential cutoff radius
  double pot_cut_rad2;

  // Periodic parameters
  double box_z;
  int diag_k_idx;

  // Collinear spin-polarization bookkeeping (flag->spinPolarized). spin_channel
  // selects which local pseudopotential read_pot loads (0 = up -> pot<X>_up.par,
  // 1 = down -> pot<X>_dn.par). file_tag is appended to the eval/psi/output file
  // names so each spin channel writes its own results; it is "" for a normal
  // (non-spin-polarized) run, leaving the default file names unchanged.
  int spin_channel;
  char file_tag[8];

  // Per-k Hamiltonian spectrum bounds (n_k_pts elements each, indexed by global
  // k index). The kinetic energy is k-dependent, so the spectrum range -- and the
  // Newton interpolation coefficients built from it -- differ for each k-point.
  double *Emin;
  double *Emax;

  // Band printing range
  int nb_min;
  int nb_max;

  // Redundant parameter (possibly for convenience)
  double dv;
} par_st;

/************************************************************/
/************************************************************/

typedef struct grid
{
  // Grid values
  double *x;
  double *y;
  double *z;

  // Grid spacing
  double dx;
  double dy;
  double dz;
  double dr;
  double dv;
  double dkx;
  double dky;
  double dkz;

  // Grid boundaries
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  double zmin;
  double zmax;

  // Grid sizes
  long nx;
  long ny;
  long nz;

  // Grid inverse sizes
  double nx_1;
  double ny_1;
  double nz_1;
  double ngrid_1;

  // Grid indices
  long *g_idx;
  long *g_zeros;

  // Redundant total grid points
  long ngrid;
} grid_st;

/************************************************************/
/************************************************************/

typedef struct atom_info
{
  long idx;
  char atyp[4];
  int Zval;
  double SO_par;
  double geom_par;
  double LR_par;
  double NL_par[2];
} atom_info;

/************************************************************/
/************************************************************/

typedef struct pot_st
{
  double *r;
  double *r_LR;
  double *pseudo;
  double *pseudo_LR;
  double *dr;
  long *file_lens;
  double *a4_params;
  double *a5_params;
} pot_st;

/************************************************************/
/************************************************************/

typedef struct zomplex
{
  double re;
  double im;
} zomplex;

/************************************************************/
/************************************************************/

typedef struct lattice_st
{
  double a;
  double b;
  double c;
  double alpha;
  double beta;
  double gamma;
  double V_lat;
  vector a1;
  vector a2;
  vector a3;
  vector b1;
  vector b2;
  vector b3;
} lattice_st;

/************************************************************/
/************************************************************/

typedef struct st5
{
  double x;
  double y;
  double z;
} xyz_st;

/************************************************************/
/************************************************************/

typedef struct st10
{
  double SO[2];
  double NL1[2];
  double NL2[2];
} coeff_st;

/************************************************************/
/************************************************************/

typedef struct st11
{
  long jxyz;
  zomplex y1[3];
  double proj[5];
  double NL_proj[5];
  int NL_proj_sign[5];
  double r;
  double r2_1;
  double r2;
  double Vr;
  // Minimum-image displacement (atom -> grid point), Bohr. Used by the periodic
  // (k != 0) NL/SO projectors to twist them by the Bloch phase e^{-i k.delta}.
  double dx;
  double dy;
  double dz;
} nlc_st;

/************************************************************/
/************************************************************/

typedef struct parallel
{
  long nthreads;
  long nranks;
  int mpi_size;
  int mpi_rank;
  int mpi_root;
  int nestedOMP;
  int n_outer_threads;
  int n_inner_threads;
  long *jns;
  long *jms;

  /* --- k-point grouping (periodic path only) --- */
  /* MPI_COMM_WORLD is split into n_k_groups sub-communicators. Each group owns a
     contiguous block of n_k_local global k-points and gathers its filtered states to
     its master (k_rank == 0) for orthogonalization/diagonalization. */
  MPI_Comm k_comm;          // sub-communicator for this rank's k-group (MPI_COMM_NULL if not periodic)
  int k_color;              // group id == world_rank / k_size
  int k_rank;               // rank within k_comm; 0 == group master
  int k_size;               // size of k_comm (ranks per k-group, == group_size)
  int n_k_groups;           // number of k-groups
  int n_k_local;            // number of global k-points this group owns
  int k_global_start;       // first global k index owned by this group (contiguous)
  int *k_local_to_global;   // length n_k_local; maps local k index -> global k index
} parallel_st;

/************************************************************/
/************************************************************/

typedef struct gauss_st
{
  double Rx;
  double Ry;
  double Rz;
  double *coeff;
  double *exp;
} gauss_st;

/************************************************************/
/************************************************************/

typedef struct MO_st
{
  double *coeff;
} MO_st;

/************************************************************/
/************************************************************/

typedef fftw_plan fftw_plan_loc;

/************************************************************/

/************************************************************/
/*******************     DEFINE MACROS    *******************/
/************************************************************/

// double macros
#define sqr(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))
#define forth(x) ((x) * (x) * (x) * (x))
#define ISWAP(a, b)    \
  {                    \
    double temp = (a); \
    (a) = (b);         \
    (b) = -temp;       \
  }
#define ISWAP2(a, b)   \
  {                    \
    double temp = (a); \
    (a) = 2.0 * (b);   \
    (b) = -2.0 * temp; \
  }

// complex number macros
#define cplus(a, b, c)          \
  {                             \
    zomplex tmp;                \
    (tmp).re = (a).re + (b).re; \
    (tmp).im = (a).im + (b).im; \
    c = tmp;                    \
  }
#define cminus(a, b, c)         \
  {                             \
    zomplex tmp;                \
    (tmp).re = (a).re - (b).re; \
    (tmp).im = (a).im - (b).im; \
    c = tmp;                    \
  }
#define cmul(a, b, c)                             \
  {                                               \
    zomplex tmp;                                  \
    (tmp).re = (a).re * (b).re - (a).im * (b).im; \
    (tmp).im = (a).re * (b).im + (a).im * (b).re; \
    c = tmp;                                      \
  }
#define cmuls(a, b, c)                            \
  {                                               \
    zomplex tmp;                                  \
    (tmp).re = (a).re * (b).re + (a).im * (b).im; \
    (tmp).im = (a).re * (b).im - (a).im * (b).re; \
    c = tmp;                                      \
  }
#define cdev(a, b, c)                                         \
  {                                                           \
    zomplex tmp;                                              \
    double mechane;                                           \
    mechane = (1.0 / ((b).re * (b).re + (b).im * (b).im));    \
    (tmp).re = ((a).re * (b).re + (a).im * (b).im) * mechane; \
    (tmp).im = ((a).im * (b).re - (a).re * (b).im) * mechane; \
    c = tmp;                                                  \
  }
#define cexp(a, c)                 \
  {                                \
    double texp = exp((a).re);     \
    zomplex tmp;                   \
    (tmp).re = texp * cos((a).im); \
    (tmp).im = texp * sin((a).im); \
    c = tmp;                       \
  }
#define cexpminx(a, c)            \
  {                               \
    double texp = exp(-(a).re);   \
    (c).re = texp * cos((a).im);  \
    (c).im = -texp * sin((a).im); \
  }

// constant macros
#define AUTOEV 27.211385
#define PIE 3.14159265358979323846
#define TWOPI 6.28318530717958647692
#define SVDEPS 1.0e-10
#define EPSR 1.0e-10
#define EPSE 1.0e-10
#define EPSR02 1.0e-2
#define EPSR0 1.0e-4
#define EPSCHI 1.0e-8
#define DENE 1.0e-10
#define EPSDX 1.0e-20
#define ANGTOBOHR 1.889726125
#define PROJ_LEN 1024
#define N_MAX_ATOM_TYPES 20
#define N_MAX_G_VECS 10000

// physical parameters
#define PbIBondMax ((3.3) * (ANGTOBOHR))
#define orthoBondAngle 160.6344
#define minCubicBondAngle 175.0
#define HBAR 1.0
#define MASS_E 1.0

// memory allocation macro for code readability
#define ALLOCATE(dblptr, length, message) allocate_memory((void **)(dblptr), (length), sizeof(typeof(**(dblptr))), message)

/*****************************************************************************/
// Function declarations

void allocate_memory(void **ptr, size_t length, size_t type_size, char *message);
// init.c
void init_grid_params(grid_st *grid, xyz_st *R, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void build_grid_ksqr(double *ksqr, xyz_st *R, grid_st *grid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void set_ene_targets(double *ene_targets, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void init_psi(zomplex *psi, long *rand_seed, grid_st *grid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
double calc_dot_dimension(xyz_st *R, long n, char *dir);
double ret_ideal_bond_len(long natyp_1, long natyp_2, int crystal_structure_int);

// gauss.c
void gauss_driver(
    gauss_st *gauss, double *pot_local, double *eig_vals, xyz_st *R, atom_info *atom, grid_st *grid,
    index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void init_gauss_params(gauss_st *gauss, xyz_st *R, atom_info *atom, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void build_gauss_hamiltonian(double *H_mat, double *T_mat, double *V_mat, index_st *ist, par_st *par);
void proj_gauss_on_grid(MO_st *MO, gauss_st *gauss, grid_st *grid, int start, int end, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void calc_X_canonical(double *S_mat, double *X, double *U, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void transform_H(double *H_mat, double *X, double *eig_vals, MO_st *MO, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);

// read.c
void read_input(flag_st *flag, grid_st *grid, index_st *ist, par_st *par, parallel_st *parallel);
void read_conf(xyz_st *R, atom_info *atm, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void read_pot(pot_st *pot, xyz_st *R, atom_info *atom, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void read_pot_file(FILE *pf, pot_st *pot, long j, long n, char *req);
void interpolate_pot(xyz_st *R, atom_info *atom, index_st *ist, par_st *par, parallel_st *parallel);
void read_periodic_input(lattice_st *lattice, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void calc_geom_par(xyz_st *R, atom_info *atm, index_st *ist, parallel_st *parallel);
double calc_bond_angle(long index1, long index2, long index3, xyz_st *R, parallel_st *parallel);
long assign_atom_number(char atyp[4]);
int is_ligand(char atyp[4]);
void assign_atom_type(char *atype, long j);
long get_number_of_atom_types(atom_info *atm, index_st *ist, long *list);
int assign_crystal_structure(char *crystal_structure);
int assign_outmost_material(char *outmost_material);
double get_ideal_bond_len(long natyp_1, long natyp_2, int crystalStructureInt);

// strain.c
void read_nearest_neighbors(vector *atom_neighbors, double *tetrahedron_vol_ref, long natoms, int crystal_structure, int outmost_material);
void calc_strain_scale(double *strain_scale, vector *atom_neighbors, double *tetrahedron_vol_ref, atom_info *atom, double *a4_params, double *a5_params, long natoms);
double calc_regular_tetrahedron_volume(double bond_length1, double bond_length2, double bond_length3, double bond_length4);

// interpolate.c
double interpolate(double r, double dr, double *vr, double *vr_LR, double *pot, double *pot_LR, long pot_file_len, long n, long j, int scale_LR, double scale_LR_par, double strain_factor, int is_LR);

// rand.c
double ran_nrc(long *rand_seed);
void Randomize(void);

// filter.c
void scale_eigs_for_cheby(zomplex *phi, zomplex *psi, double *pot_local, vector *G_vecs, vector k, grid_st *grid, nlc_st *nlc, long *nl, double zm1,
                          fftw_plan_loc planfw, fftw_plan_loc planbw, fftw_complex *fftwpsi, index_st *ist, par_st *par, flag_st *flag);

// integral.c
void overlap_gauss(double *S_mat, gauss_st *gauss, atom_info *atom, index_st *ist, par_st *par, flag_st *flag);
void kinetic_gauss(double *T_mat, gauss_st *gauss, atom_info *atom, index_st *ist, par_st *par, flag_st *flag);
void potential_gauss(double *V_mat, double *pot_local, gauss_st *gauss, grid_st *grid, atom_info *atom, index_st *ist, par_st *par, flag_st *flag);

// norm.c
double calc_norm(zomplex *, double, long, long);
double normalize(zomplex *, long ngrid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void normalize_all(double *psitot, long n_states, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);

// Hmat.c
MKL_Complex16 dotp(zomplex *psi, double *phi, long m, long ngrid, double dv);
double dotpreal(zomplex *psi, double *phi, long m, long ngrid, double dv);

// nerror.c
void nerror(char *);

// projectors.c
void gen_SO_projectors(double dx, double rcut, long nproj, double *projectors, double *vr);
void gen_nlc_projectors(double dx, double rcut, long nproj, double *projectors, int *sgnProj, double *vr, atom_info *atm, long jatom);

// pseudopotential.c
void calc_pot_overlap(double *psitot, double *pot_local, nlc_st *nlc, long *nl, double *eval, par_st *par, index_st *ist, flag_st *flag);

// save.c
void print_input_state(FILE *pf, flag_st *flag, grid_st *grid, par_st *par, index_st *ist, parallel_st *parallel);
void save_job_state(char *file_name, int checkpoint_id, double *psitot, double *pot_local, double *ksqr, zomplex *an, double *zn, double *ene_targets, long *nl,
                    nlc_st *nlc, grid_st *grid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void restart_from_save(char *file_name, int checkpoint_id, double *psitot, double *pot_local, double *ksqr, zomplex *an, double *zn, double *ene_targets, long *nl,
                       nlc_st *nlc, grid_st *grid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);
void save_output(char *file_name, double *psitot, double *eig_vals, double *sigma_E, xyz_st *R, grid_st *grid, index_st *ist, par_st *par, flag_st *flag, parallel_st *parallel);

// write.c
long cube_atom_number(char *atomSymbol);
double density_phase_sign(double re, double im);
void write_cube_file(double *rho, grid_st *grid, char *fileName);
void write_cube_super(double *rho, grid_st *grid, int N1, int N2, int N3, char *fileName);
void write_separation(FILE *pf, char *top_bttm);
void write_state_dat(zomplex *psi, long n_elems, char *fileName);
void write_vector_dat(vector *vec, int n_elems, char *fileName);
void write_eval_dat(double *eig_vals, double *sigma_E, long n_elems, char *fileName);
void write_psi_dat(double *psitot, long n_elems, char *fileName);

/*****************************************************************************/
