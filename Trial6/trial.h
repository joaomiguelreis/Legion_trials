#include <cstdio>

#include <iostream>
#include <vector>
#include <random>
// #define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "pcb++.h"
#include "cub++.h"
#include "mstoch++.h"
#include "kldec.h"


#include "legion.h"
#include "ddm.h"

using namespace std;
using namespace Legion;


struct GlobalProblem{
  PartitionDomain *partition;
  VectorXd Solution;
  VectorXd BCVal;
  LogicalRegion all_subproblems_lr;
};

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  UPDATE_BCS_TASK_ID,
  SUBPROBLEM_SOLVER_TASK_ID,
};

enum FieldIDs {
  // INDEX_SUBDOMAIN,
  // MAIN_SUBDOMAIN,
  NEXT_LEFT_BC,
  NEXT_RIGHT_BC,
  COLOR,
};


struct SubProblem{
  // Subdomain main_subdomain;
  double next_Ul, next_Ur;
  int color;
};

// typedef FieldAccessor<READ_WRITE, unsigned, 1> > AccessorRWunsigned;
// typedef FieldAccessor<READ_WRITE, Subdomain, 1> > AccessorRWsub;


// struct DomainDecomp {
//   LogicalRegion all_subproblems;
//   PhysicalRegion all_subproblems_pr;
//   PartitionDomain *partition;
//   int num_subproblems;
// };

// struct DomainDecompPiece {

//   LogicalRegion subproblem_lr;
//   LogicalRegion main_subdomain_lr, neigh_subdomains_lr;
//   PartitionDomain *partition_piece;
//   Subdomain main_subdomain, left_subdoman, right_subdomain;

// };