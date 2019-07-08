

#include <cstdio>

#include "legion.h"

#include <iostream>
#include <vector>
#include <random>
// #define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include "pcb++.h"
// #include "cub++.h"
// #include "mstoch++.h"
// #include "kldec.h"

using namespace std;
using namespace Eigen;
using namespace Legion;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

// #include "ddm.h"

// using namespace std;
// using namespace Eigen;

// // All of the important user-level objects live 
// // in the Legion namespace.
// using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  SCHWARZ_SOLVER_TASK_ID,
  DO_ONE_SCHWARZ_TASK_ID,
  INIT_FIELDS_TASK_ID,
  LOC_SOLVER_TASK_ID,
  UPDATE_BV_TASK_ID,
};



enum SolutionsFieldIDs {
  ULOC_FID,
};



enum BVFieldIDs {
  NEXT_LEFT_BC_FID,
  NEXT_RIGHT_BC_FID,
  CHOL_FID,
  FIELD_FID,
};


struct LocalProblems{

	int index;
	VectorXd X;
	unsigned eglo_s;
	VectorXd source;
	VectorXd field;
	VectorXd UBC;
	// SpMat OPer;
	// SimplicialCholesky<SpMat> cholesky;	
	unsigned ned;
	VectorXi IGL;
	VectorXi IGR;
	int num_pieces;
	int subproblems_per_piece;
	double u0, u1;
};

void allocate_solutions_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_BV_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
VectorXd SetSource(const VectorXd &S_Global, const unsigned ned, const unsigned eglo_s, const VectorXd X);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{

	unsigned Ne = 10;
	unsigned Nd = 12;
	unsigned No = 3; 
	unsigned Net = Nd*Ne+No;  // number of total elements (added "no" elements to last subdomain)


	/* Spatial settings */
	double dx = 1 / float(Net); 
	VectorXd MeshG = VectorXd::Zero(Net+1);

	printf("Do spatial mesh\n");
	for (int e = 0; e < Net+1; e++) MeshG(e) = e*dx;
	VectorXd SF = VectorXd::Ones(Net);
	VectorXd KF = VectorXd::Ones(Net);




	int num_pieces = Nd;
	int subproblems_per_piece = (int) Nd/num_pieces;

	/*

	DEFINE TWO LOGICAL REGIONS: FOR ALL SOLUTIONS AND FOR ALL OPERATORS
	EACH INDEX CORRESPONDS TO A SUBDOMAIN

	*/


	printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");

	// Make field spaces
	printf("Define field spaces for solutions\n");
	FieldSpace solutions_fs = runtime->create_field_space(ctx);
	runtime->attach_name(solutions_fs, "all_solutions_field_space");

	// Make field spaces
	printf("Define field spaces boundary values\n");
	FieldSpace BV_fs = runtime->create_field_space(ctx);
	runtime->attach_name(BV_fs, "all_BV_field_space");

	// Allocate fields
	printf("Allocate field spaces\n");
	allocate_solutions_fields(ctx, runtime, solutions_fs);
	allocate_BV_fields(ctx, runtime, BV_fs);


	//Finally create logical region all_subproblems
	printf("Create logical region\n");
	LogicalRegion solutions_lr = runtime->create_logical_region(ctx, is, solutions_fs);
	runtime->attach_name(solutions_lr, "solutions_lr");
	LogicalRegion BV_lr = runtime->create_logical_region(ctx, is, BV_fs);
	runtime->attach_name(BV_lr, "all_BV_lr");


	/*

	THE LOGICAL REGIONS ARE PARTITIONED INTO THE NUMBER OF SUBDOMAINS
	THE NUMBER OF PIECES = NUMBER OF SUBDOMAINS

	*/

	// CREATE PARTITIONS
	printf("START CREATING PARTITIONS\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);

	LogicalPartition solutions_lp = runtime->get_logical_partition(ctx, solutions_lr, ip);
	LogicalPartition BV_lp = runtime->get_logical_partition(ctx, BV_lr, ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");

	/*

	THIS ARGUMENT MAP CONTAINS THE ARGUMENTS TO SOLVE EACHH LOCAL PROBLEM

	*/

	printf("CREATING ARGUMENT MAP\n");
	ArgumentMap init_args;
	for (int in=0; in<num_pieces; in++){
		LocalProblems local_problems;
		Point<1> point(in);
		local_problems.ned = Ne+No;
		local_problems.eglo_s = in * Ne;
		local_problems.X = MeshG.segment(in*Ne,Ne+No+1);
		local_problems.source = SetSource(SF,Ne+No,in*Ne,local_problems.X);
		local_problems.field = KF.segment(in*Ne,Ne+No);

		//UBC
		local_problems.UBC = VectorXd::Zero(2);
		VectorXd X = local_problems.X;
		VectorXd field = local_problems.field;
		local_problems.UBC(0) = field(0)/fabs(X(1)-X(0));
		local_problems.UBC(1) = field(Ne+No-1)/fabs(X(Ne+No)-X(Ne+No-1));

		init_args.set_point(point, TaskArgument(&local_problems, sizeof(LocalProblems)));
	}

	// INITIATE FIELDS. HERE I INITIATE THE CHOLESKY DECOMPOSITION OF A SPARSE MATRIX AMONG
	// OTHER THINGS
	printf("INTIATE LAUNCH FOR OTHER INITIATE FIELDS\n");
	IndexTaskLauncher init_launcher(INIT_FIELDS_TASK_ID, is, TaskArgument(NULL,0), init_args);

	RegionRequirement sol_reqWO(solutions_lp, 0, WRITE_ONLY, EXCLUSIVE, solutions_lr);
	init_launcher.add_region_requirement(sol_reqWO);
	init_launcher.region_requirements[0].add_field(0,ULOC_FID);

	RegionRequirement oper_reqWO(BV_lp, 0, WRITE_ONLY, EXCLUSIVE, BV_lr);
	init_launcher.add_region_requirement(oper_reqWO);
	init_launcher.region_requirements[1].add_field(0,NEXT_LEFT_BC_FID);
	init_launcher.region_requirements[1].add_field(1,NEXT_RIGHT_BC_FID);
	init_launcher.region_requirements[1].add_field(2,CHOL_FID);

	printf("-  execute\n");
	FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
	fm.wait_all_results();



	// SOLVE LOCAL PROBLEMS
	printf("INTIATE LAUNCH FOR LOCAL SOLVER\n");
	IndexTaskLauncher loc_solver_launch(LOC_SOLVER_TASK_ID, is, TaskArgument(NULL,0), init_args);

	RegionRequirement sol_reqRW(solutions_lp, 0, READ_WRITE, EXCLUSIVE, solutions_lr);
	loc_solver_launch.add_region_requirement(sol_reqRW);
	loc_solver_launch.region_requirements[0].add_field(0,ULOC_FID);

	RegionRequirement oper_reqRO(BV_lp, 0, READ_ONLY, EXCLUSIVE, BV_lr);
	loc_solver_launch.add_region_requirement(oper_reqRO);
	loc_solver_launch.region_requirements[1].add_field(0,NEXT_LEFT_BC_FID);
	loc_solver_launch.region_requirements[1].add_field(1,NEXT_RIGHT_BC_FID);
	loc_solver_launch.region_requirements[1].add_field(2,CHOL_FID);

	printf("-  execute\n");
	runtime->execute_index_space(ctx, loc_solver_launch);





}

void loc_solver_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime){

	LocalProblems loc = *((const LocalProblems*)task->local_args);

	VectorXd Source = loc.source;
	VectorXd UBC = loc.UBC;
	unsigned ned = loc.ned;
	VectorXd field = loc.field;
	VectorXd X = loc.X;

	const FieldAccessor<READ_WRITE, VectorXd, 1> acc_uloc(regions[0], ULOC_FID);
	const FieldAccessor<READ_ONLY, double, 1> acc_left(regions[1], NEXT_LEFT_BC_FID);
	const FieldAccessor<READ_ONLY, double, 1> acc_right(regions[1], NEXT_RIGHT_BC_FID);
	const FieldAccessor<READ_ONLY, SimplicialCholesky<SpMat>, 1> acc_chol(regions[1], CHOL_FID);

	Point<1> point(1);


	double Ul = acc_left[point];
	double Ur = acc_left[point];

	// applies boundary values in the source vector
	VectorXd b = Source; b(0) += UBC(0)*Ul; b(ned-2) += UBC(1)*Ur;

	// just making sure we are not deleting any rows in the FE system
	if(b.rows()!=ned-1) cout << "Mismatch\n";

	// solving M_d x_d = b_d
	VectorXd Usol = VectorXd(ned+1); // initiate vector with solution values
	SimplicialCholesky<SpMat> chol;
	&chol = acc_chol[point];
	// chol.compute(OPer);
	Usol.segment(1,ned-1) = chol.solve(b); /* solves portion of Mx=b for inner points of current
											  subdomain. Boundary values will be added later.
											  This is the heaviest computation and should be done 
											  in parallel. */
	Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges


	acc_uloc[point] = Usol;

	cout << Usol.transpose() << endl;
	cout << " " << endl;


}


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void init_fields_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime){

	LocalProblems loc = *((const LocalProblems*)task->local_args);

	unsigned ned = loc.ned;
	VectorXd field = loc.field;
	VectorXd X = loc.X;


	const FieldAccessor<WRITE_ONLY, VectorXd, 1> acc_uloc(regions[0], ULOC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_left(regions[1], NEXT_LEFT_BC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_right(regions[1], NEXT_RIGHT_BC_FID);
	const FieldAccessor<WRITE_ONLY, VectorXd, 1> acc_field(regions[1], FIELD_FID);
	const FieldAccessor<WRITE_ONLY, SimplicialCholesky<SpMat>, 1> acc_chol(regions[1], CHOL_FID);

	Point<1> point(1);
	// acc_uloc[point] = VectorXd::Zero(loc.ned-1);

	// // update boundary conditions of each problem 
	// if (loc.index==0) acc_left[point] = loc.u0;
	// else acc_left[point] = 0;
	// if (loc.index==loc.num_pieces-1) acc_right[point] = loc.u1;
	// else acc_right[point] = 0;




	// Find Cholesky decomp
	vector<T> cprec; //inititate the vector with values of FEM integral

	//Loop over elements
	for(int e=0; e<ned; e++){				
		double kv = field(e)/fabs(X(e+1)-X(e)); // k(x_e)/dx	

		if(e==0){	// first sudomain	
			cprec.push_back(T(e,e,kv)); 

		}else if (e==ned-1){ // last subdomain
			cprec.push_back(T(e-1,e-1,kv));

		} else { //inner subdomains
			cprec.push_back(T(e-1,e-1, kv));
			cprec.push_back(T(e-1,e  ,-kv));
			cprec.push_back(T(e  ,e-1,-kv));
			cprec.push_back(T(e  ,e  , kv));
			 /*T(e1, e2, value_kappa) is the approx of 
					the FEM integral for basis functions with indices (e1,e2).
					This tells the value is approximated by value_kappa.
					Since we are doing linear interpolation, we just have a flat
					approx of the integral*/
			
		}
	}
	
	SpMat OPer = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
	OPer.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
														 Oper_(e1,e2) = kappa_value */
	SimplicialCholesky<SpMat> chol;
	chol.compute(OPer);   // find cholesky decomposition. This will be used to solve the FE 
						  // linear system 
	

	acc_chol[point] = chol; // this will be the same for each subdomain at every iteration
							// I don't want to be computing this in the local solver
							// I want to pass this to the local solver or use it there somehow



}

VectorXd SetSource(const VectorXd &S_Global, const unsigned ned, const unsigned eglo_s, const VectorXd X){
	// S_Global: source function defined global domain

	VectorXd Source = VectorXd::Zero(ned-1);

	// loop over elements subdomain
	for(int e=0; e<ned; e++){
		double f = S_Global(eglo_s+e)*.5*fabs(X(e+1)-X(e));	//Source term
		if(e==0){
			Source(e  ) += f;													
		}else if (e==ned-1){
			Source(e-1) += f;
		} else {
			Source(e-1) += f;							
			Source(e  ) += f;
		}
	} //next element

	return Source;
};

void allocate_solutions_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(VectorXd), ULOC_FID);

};

void allocate_BV_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), NEXT_LEFT_BC_FID);
	allocator.allocate_field(sizeof(double), NEXT_RIGHT_BC_FID);
	allocator.allocate_field(sizeof(SpMat), CHOL_FID);
	allocator.allocate_field(sizeof(VectorXd), FIELD_FID);
};


int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}


	{
	  TaskVariantRegistrar registrar(INIT_FIELDS_TASK_ID, "init_fields_task");
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	  // registrar.set_leaf(true);
	  Runtime::preregister_task_variant<init_fields_task>(registrar, "init_fields_task");
	}

	{
	  TaskVariantRegistrar registrar(LOC_SOLVER_TASK_ID, "loc_solver_task");
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	  // registrar.set_leaf(true);
	  Runtime::preregister_task_variant<loc_solver_task>(registrar, "loc_solver_task");
	}


	return Runtime::start(argc, argv);
}