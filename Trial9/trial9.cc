#include <cstdio>

#include "legion.h"

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

#include "ddm.h"

// using namespace std;
// using namespace Eigen;

// // All of the important user-level objects live 
// // in the Legion namespace.
// using namespace Legion;


struct LocalProblems{

	int index;
	VectorXd X;
	unsigned eglo_s;
	VectorXd source;
	VectorXd field;
	VectorXd UBC;
	// SpMat OPer;
	// SimplicialCholesky<SpMat> *cholesky;
	// vector<T> cprec; //inititate the vector with values of FEM integral
	unsigned ned;
	unsigned no;
	VectorXi IGL;
	VectorXi IGR;
	int num_pieces;
	int subproblems_per_piece;
	double u0, u1;
};


struct GlobalProblem{

	VectorXi IGL;
	VectorXi IGR;
	int num_pieces;
	int subproblems_per_piece;
	double u0, u1;
	VectorXd Uinit;
};


void allocate_solutions_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_BV_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void set_partitions(Context ctx, Runtime *runtime,  IndexSpace parent_is,LogicalRegion solutions_lr, LogicalRegion BV_lr,
						PartitionDomain Part, int num_pieces);
VectorXd SetSource(const VectorXd &S_Global, const unsigned ned, const unsigned eglo_s, const VectorXd X);
VectorXd OperatorFEM(vector<T> &cprec, const VectorXd field, const VectorXd UBC, const unsigned ned, const VectorXd X);
// LogicalPartition load_all_subdomains(Context ctx, Runtime *runtime, GlobalProblem GP, 
// 	                      vector<SubProblem> subproblems, int num_pieces, int subproblems_per_piece);


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{

	unsigned Ne = 10;
	unsigned Nd = 12;
	unsigned No = 3; 
	double mu_kl = 1;
	double var_kl = 0.1;
	double len_kl = 0.1;
	double prec = 1e-9;
	// int M = 10;
	unsigned Net = Nd*Ne+No;  // number of total elements (added "no" elements to last subdomain)


	/* Spatial settings */
	double dx = 1 / float(Net); 
	VectorXd MeshG = VectorXd::Zero(Net+1);

	printf("Do spatial mesh\n");
	for (int e = 0; e < Net+1; e++) MeshG(e) = e*dx;
	VectorXd SF = VectorXd::Ones(Net);


	/* Vector with subdomains */
	printf("Initiate subdomains\n");
	vector<Subdomain> SDOM(Nd);

	VectorXd Usol = VectorXd::Zero(Net+1);
	Usol(Net)=1;


	/* Partition of the domain */
	printf("Initiate partitions\n");
	PartitionDomain Part(Nd,No,MeshG,Net,mu_kl,prec,Usol);
	Part.Set(&SDOM);      //Set the partition (i.e. do the subdomains)


	/* Find KL modes */
	printf("Find KL modes\n");
	VectorXd Lam;
	MatrixXd Modes = SetKLModes(Net, Lam, len_kl, var_kl);
	for(int e=0; e<Net; e++) Modes.col(e) *= Lam(e);


	VectorXd KG = VectorXd::Ones(Net+1);
	VectorXd SG = VectorXd::Ones(Net+1);
	// Part.SetOperators(KG);
	// Part.SetSources(SG);

	// SimplicialCholesky<SpMat> chol;
	// SpMat OPerator = SDOM[0].OPer;
	// chol.compute(OPerator);




	int num_pieces = Nd;
	int subproblems_per_piece = (int) Nd/num_pieces;


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


	// int num_pieces = Part.GetNumPieces();

	// CREATE PARTITIONS
	printf("START CREATING PARTITIONS\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);

	LogicalPartition solutions_lp = runtime->get_logical_partition(ctx, solutions_lr, ip);
	LogicalPartition BV_lp = runtime->get_logical_partition(ctx, BV_lr, ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");

	printf("FILL GLOBAL PROBLEM STRUCT\n");
	GlobalProblem GP;
	GP.IGL = Part.GetIGL();
	GP.IGR = Part.GetIGR();
	GP.num_pieces = num_pieces;
	GP.Uinit = Usol;
	GP.u0 = Usol(0);
	GP.u1 = Usol(Net);

	unsigned ned = Ne+No;



	printf("CREATING ARGUMENT MAP\n");
	ArgumentMap init_args;
	for (int in=0; in<num_pieces; in++){
		LocalProblems local_problems;
		Point<1> point(in);
		local_problems.index = in;

		// unsigned ned = Ne+No;
		local_problems.ned = ned;

		local_problems.no = No;
		local_problems.eglo_s = in * Ne;

		VectorXd X = MeshG.segment(in*Ne,Ne+No+1);
		local_problems.X = X;

		local_problems.source = SetSource(SG,Ne+No,in*Ne,local_problems.X);

		VectorXd field = KG.segment(in*Ne,Ne+No);
		local_problems.field = field;
		//UBC
		VectorXd UBC = VectorXd::Zero(2);;
		UBC(0) = field(0)/fabs(X(1)-X(0));
		UBC(1) = field(Ne+No-1)/fabs(X(Ne+No)-X(Ne+No-1));
		// local_problems.OPer = SDOM[in].OPer;
		local_problems.UBC = UBC;


		local_problems.IGL = Part.GetIGL();
		local_problems.IGR = Part.GetIGR();
		local_problems.num_pieces = num_pieces;
		local_problems.subproblems_per_piece = subproblems_per_piece;
		local_problems.u0 = Usol(0);
		local_problems.u1 = Usol(Net);

		init_args.set_point(point, TaskArgument(&local_problems, sizeof(LocalProblems)));
	}

	RegionRequirement req(BV_lr, WRITE_ONLY, EXCLUSIVE, BV_lr);
	req.add_field(OPER_FID);

	InlineLauncher input_launcher(req);
	PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
	input_region.wait_until_valid();

	const FieldAccessor<WRITE_ONLY, vector<T>, 1> acc_oper(input_region, OPER_FID);
	for (int n=0; n<num_pieces; n++){

		VectorXd field = KG.segment(n*Ne,Ne+No);
		VectorXd X = MeshG.segment(n*Ne,Ne+No+1);

		//Loop over elements
		std::vector<T> cprec;
		for(int e=0; e<ned; e++){				
			double kv = field(e)/fabs(X(e+1)-X(e)); // k(x_e)/dx	

			if(e==0){	// first sudomain	
				cprec.push_back(T(e,e,kv)); 
				// UBC(0) = kv;

			}else if (e==ned-1){ // last subdomain
				cprec.push_back(T(e-1,e-1,kv));
				// UBC(1) = kv;

			} else { //inner subdomains
				cprec.push_back(T(e-1,e-1, kv));
				cprec.push_back(T(e-1,e  ,-kv));
				cprec.push_back(T(e  ,e-1,-kv));
				cprec.push_back(T(e  ,e  , kv));
				/* T(e1, e2, value_kappa) is the approx of 
						the FEM integral for basis functions with indices (e1,e2).
						This tells the value is approximated by value_kappa.
						Since we are doing linear interpolation, we just have a flat
						approx of the integral
				*/
			}
		}
		// SpMat OPer = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
		// OPer.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
															// Oper_(e1,e2) = kappa_value */

		acc_oper[n] = cprec;
	}


	



	printf("INTIATE LAUNCH FOR OTHER INITIATE FIELDS\n");
	IndexTaskLauncher init_launcher(INIT_FIELDS_TASK_ID, is, TaskArgument(NULL,0), init_args);

	RegionRequirement sol_reqWO(solutions_lp, 0, WRITE_ONLY, EXCLUSIVE, solutions_lr);
	init_launcher.add_region_requirement(sol_reqWO);
	init_launcher.region_requirements[0].add_field(0,ULOC_FID);

	RegionRequirement oper_reqWO(BV_lp, 0, WRITE_ONLY, EXCLUSIVE, BV_lr);
	init_launcher.add_region_requirement(oper_reqWO);
	init_launcher.region_requirements[1].add_field(0,NEXT_LEFT_BC_FID);
	init_launcher.region_requirements[1].add_field(1,NEXT_RIGHT_BC_FID);
	init_launcher.region_requirements[1].add_field(2,OPER_FID);
	// init_launcher.region_requirements[1].add_field(2,FIELD_FID);

	printf("-  execute\n");
	FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
	fm.wait_all_results();


	printf("INTIATE LAUNCH FOR LOCAL SOLVER\n");
	IndexTaskLauncher loc_solver_launch(LOC_SOLVER_TASK_ID, is, TaskArgument(NULL,0), init_args);

	RegionRequirement sol_reqRW(solutions_lp, 0, READ_WRITE, EXCLUSIVE, solutions_lr);
	loc_solver_launch.add_region_requirement(sol_reqRW);
	loc_solver_launch.region_requirements[0].add_field(0,ULOC_FID);

	RegionRequirement oper_reqRO(BV_lp, 0, READ_WRITE, EXCLUSIVE, BV_lr);
	loc_solver_launch.add_region_requirement(oper_reqRO);
	loc_solver_launch.region_requirements[1].add_field(0,NEXT_LEFT_BC_FID);
	loc_solver_launch.region_requirements[1].add_field(1,NEXT_RIGHT_BC_FID);
	loc_solver_launch.region_requirements[1].add_field(2,OPER_FID);
	// loc_solver_launch.region_requirements[1].add_field(2,FIELD_FID);

	printf("-  execute\n");
	FutureMap futureUsol = runtime->execute_index_space(ctx, loc_solver_launch);
	// futureUsol.wait_all_results();

	IndexTaskLauncher updateBC_launch(UPDATE_BC_TASK_ID, is, TaskArgument(NULL,0), init_args);
	updateBC_launch.add_region_requirement(sol_reqWO);
	updateBC_launch.region_requirements[0].add_field(0,ULOC_FID);

	updateBC_launch.add_region_requirement(oper_reqWO);
	updateBC_launch.region_requirements[1].add_field(0,NEXT_LEFT_BC_FID);
	updateBC_launch.region_requirements[1].add_field(1,NEXT_RIGHT_BC_FID);
	updateBC_launch.region_requirements[1].add_field(2,OPER_FID);

	// ArgumentMap BC_args;
	// cout << GP.IGR << endl;
	// for (int n=0; n<num_pieces; n++){

	// 	VectorXd local_BCval = VectorXd::Zero(2);
	// 	local_BCval(0) = GP.u0;
	// 	local_BCval(1) = GP.u1;
	// 	if(n>0){

	// 		Point<1> left_fm(GP.IGL(n));
	// 		VectorXd uloc_left = futureUsol.get_result<VectorXd>(n-1);
	// 		local_BCval(0) = uloc_left(No);
	// 	}

	// 	if(n<num_pieces-2){
	// 		Point<1> right_fm(GP.IGR(n));
	// 		VectorXd uloc_right = futureUsol.get_result<VectorXd>(n+1);
	// 		local_BCval(1) = uloc_right(No);
	// 	}
		

	// 	Point<1> point_args(n);
	// 	cout << "here" << endl;
	// 	BC_args.set_point(point_args, TaskArgument(&local_BCval, sizeof(VectorXd)));
	// }

	// IndexTaskLauncher updateBC_launch(UPDATE_BC_TASK_ID, is, TaskArgument(NULL,0), BC_args);

	// RegionRequirement BV_reqWO(BV_lp, 0, WRITE_ONLY, EXCLUSIVE, BV_lr);
	// updateBC_launch.add_region_requirement(BV_reqWO);
	// updateBC_launch.region_requirements[0].add_field(0,NEXT_LEFT_BC_FID);
	// updateBC_launch.region_requirements[0].add_field(1,NEXT_RIGHT_BC_FID);

	// runtime->execute_index_space(ctx, updateBC_launch);






}

void updateBCs_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime){

	LocalProblems loc = *((const LocalProblems*)task->local_args);

	int index = loc.index;
	int num_pieces = loc.num_pieces;
	unsigned ned = loc.ned;
	unsigned no = loc.no;

	const FieldAccessor<READ_ONLY, VectorXd, 1> acc_uloc(regions[0], ULOC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_left(regions[1], NEXT_LEFT_BC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_right(regions[1], NEXT_RIGHT_BC_FID);

	Point<1> current_subdomain(index);
	Point<1> left_subdomain(index-1);
	Point<1> right_subdomain(index+1);


	if (loc.index>0){
		VectorXd uloc = acc_uloc[right_subdomain];
		acc_left[current_subdomain] = uloc(ned-no);

	} 
	if (loc.index<num_pieces-1){
		VectorXd uloc = acc_uloc[left_subdomain];
		acc_right[current_subdomain] = uloc(no);
	}


}


// void updateBCs_task(const Task *task,
// 			          const std::vector<PhysicalRegion> &regions,
// 			          Context ctx, Runtime *runtime){

// 	VectorXd local_BCval = *((const VectorXd*)task->local_args);

// 	const FieldAccessor<READ_WRITE, double, 1> acc_left(regions[0], NEXT_LEFT_BC_FID);
// 	const FieldAccessor<READ_WRITE, double, 1> acc_right(regions[0], NEXT_RIGHT_BC_FID);


// 	Point<1> point(1);
// 	acc_left[point] = local_BCval(0);
// 	acc_right[point] = local_BCval(1);

	

// }

VectorXd loc_solver_task(const Task *task,
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
	const FieldAccessor<READ_ONLY, vector<T>, 1> acc_chol(regions[1], OPER_FID);

	Point<1> point(loc.index);


	double Ul = acc_left[point];
	double Ur = acc_right[point];

	// applies boundary values in the source vector
	VectorXd b = Source; b(0) += UBC(0)*Ul; b(ned-2) += UBC(1)*Ur;

	// just making sure we are not deleting any rows in the FE system
	if(b.rows()!=ned-1) cout << "Mismatch\n";
	cout << "  here " << endl;
	vector<T> cprec = acc_chol[point];
	SpMat OPer = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
	OPer.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
														// Oper_(e1,e2) = kappa_value */

	SimplicialCholesky<SpMat> chol; // cholesky decomp of (sparse) mass matrix
	chol.compute(OPer); /*  find cholesky decomposition. This will be used to solve the FE 
							linear system */
	VectorXd Usol = VectorXd(ned+1); // initiate vector with solution values
	Usol.segment(1,ned-1) = chol.solve(b); /* solves portion of Mx=b for inner points of current
											  /* subdomain. Boundary values will be added later.
											  This is the heaviest computation and should be done 
											  in parallel. */
	//Usol.segment(1,ned-1) = VectorXd::Constant(ned-1,1);
	Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges


	acc_uloc[point] = Usol;

	cout << Usol.transpose() << endl;
	cout << " " << endl;

	return Usol;


}


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void init_fields_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime){

	LocalProblems loc = *((const LocalProblems*)task->local_args);

	// edit Dirichlet BC here
	double u0 = 0;
	double u1 = 1;

	unsigned eglo_s = loc.eglo_s;
	unsigned ned = loc.ned;
	VectorXd field = loc.field;
	VectorXd X = loc.X;


	const FieldAccessor<WRITE_ONLY, VectorXd, 1> acc_uloc(regions[0], ULOC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_left(regions[1], NEXT_LEFT_BC_FID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_right(regions[1], NEXT_RIGHT_BC_FID);
	const FieldAccessor<WRITE_ONLY, vector<T>, 1> acc_chol(regions[1], OPER_FID);


	Point<1> point(loc.index);

	acc_uloc[point] = VectorXd::Zero(loc.ned-1);
	if (loc.index==0) acc_left[point] = u0;
	else acc_left[point] = 0;
	if (loc.index==loc.num_pieces-1) acc_right[point] = u1;
	else acc_right[point] = 0;
	// acc_chol[point] = cprec;

	// SimplicialCholesky<SpMat> chol;
	// acc_chol[point] = OPer;



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
	allocator.allocate_field(sizeof(vector<T>), OPER_FID);
	// allocator.allocate_field(sizeof(VectorXd), FIELD_FID);
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
	  Runtime::preregister_task_variant<VectorXd, loc_solver_task>(registrar, "loc_solver_task");
	}

	{
	  TaskVariantRegistrar registrar(UPDATE_BC_TASK_ID, "updateBCs_task");
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	  // registrar.set_leaf(true);
	  Runtime::preregister_task_variant<VectorXd, loc_solver_task>(registrar, "updateBCs_task");
	}

	

	return Runtime::start(argc, argv);
}