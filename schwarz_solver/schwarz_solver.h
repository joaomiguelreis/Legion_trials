#include <cstdio>
#include <iostream>
#include <vector>
#include <random>

// #define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include "../StochTk++/includes/pcb++.h"
// #include "../StochTk++/includes/cub++.h"
// #include "../StochTk++/includes/mstoch++.h"
// #include "kldec.h"

#include "../legion/runtime/legion.h"


using namespace std;
using namespace Eigen;
using namespace Legion;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::Triplet<double> T;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  SCHWARZ_SOLVER_TASK_ID,
  INIT_FIELDS_TASK_ID,
  INIT_SPATIAL_TASK_ID,
  LOC_SOLVER_TASK_ID,
  UPDATE_BC_TASK_ID,
  SUBDOMAIN_COLOR_TASK_ID,
};

enum EdgeFieldsID{
	EDGE_LEFT_ID,
	EDGE_RIGHT_ID,
	MESH_LEFT_ID,
	MESH_RIGHT_ID,
};

enum InnerFieldsID{
	INNER_LEFT_ID,
	INNER_RIGHT_ID,
	DONE_ID,
};

enum SpatialFieldsID
{
	FIELD_ID,
	SOURCE_ID,
	MESH_ID,
	SUBDOMAIN_COLOR_ID,
};


struct Indices{
	int num_pieces;
	double u0, u1;
	int elements_per_subdomain;
	int num_overlapping_elem;
	int num_total_elements;
	int subdomain_index;
	double tolerance;
};

struct LocalSpatial{
	int num_elements_subdomain;
	int num_pieces;
	vector<double> KF_vector;
	vector<double> SF_vector;
	vector<double> MeshG_vector;
};



