#include <cstdio>

// #include "legion.h"

// #include <iostream>
// #include <vector>
// #include <random>
// // #define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
// #include <Eigen/Dense>
// #include <Eigen/Sparse>
// #include "pcb++.h"
// #include "cub++.h"
// #include "mstoch++.h"
// #include "kldec.h"

#include "trial.h"

using namespace std;
using namespace Eigen;

// All of the important user-level objects live 
// in the Legion namespace.
using namespace Legion;




void allocate_subproblem_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
LogicalPartition load_all_subdomains(Context ctx, Runtime *runtime, GlobalProblem GP, 
	                      vector<SubProblem> subproblems, int num_pieces, int subproblems_per_piece);


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
	// double prec = 1e-9;
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


	/* Partition of the domain */
	printf("Initiate partitions\n");
	PartitionDomain Part(Nd,No,MeshG,Net,mu_kl);
	Part.Set(&SDOM);      //Set the partition (i.e. do the subdomains)


	/* Find KL modes */
	printf("Find KL modes\n");
	VectorXd Lam;
	MatrixXd Modes = SetKLModes(Net, Lam, len_kl, var_kl);
	for(int e=0; e<Net; e++) Modes.col(e) *= Lam(e);

	VectorXd Usol = VectorXd::Zero(Net+1);

	int num_pieces = 4;
	int subproblems_per_piece = Nd/num_pieces;

	GlobalProblem GP;

		GP.partition = &Part;

		GP.Solution = Usol;
		GP.BCVal = Part.GetBCVal(Usol);
		cout << GP.BCVal << endl;

		// Make index spaces for all_suproblems
		printf("Define Index Space\n");
		Rect<1> rect(0,Nd-1);
		IndexSpaceT<1> all_subproblems_is = runtime->create_index_space(ctx, rect);
		runtime->attach_name(all_subproblems_is, "all_subdomains_index_space");

		// Make field spaces
		printf("Define field spaces\n");
		FieldSpace all_subproblems_fs = runtime->create_field_space(ctx);
		runtime->attach_name(all_subproblems_fs, "all_subdomains_field_space");

		// Allocate fields
		printf("Allocate field spaces\n");
		allocate_subproblem_fields(ctx, runtime, all_subproblems_fs);

		

		//Finally create logical region all_subproblems
		printf("Create logical region\n");
		GP.all_subproblems_lr = runtime->create_logical_region(ctx, all_subproblems_is, all_subproblems_fs);
		runtime->attach_name(GP.all_subproblems_lr, "all_subdomains");

		std::vector<SubProblem> subproblems(Nd);
		// load_subproblems(GP, subproblems, Nd);
		for (int ip=0; ip<num_pieces; ip++){

			for(int is=0; is<subproblems_per_piece; is++){

				int n = ip * subproblems_per_piece + is;
				int q = 2*n;
				// subproblems[n].main_subdomain = SDOM[n];
				if (n==0 || n==Nd-1){
					if (n==0){
						subproblems[n].next_Ul = 0;
						subproblems[n].next_Ur = GP.BCVal(0);
					}
					if (n==Nd){
						subproblems[n].next_Ul = GP.BCVal(2*(Nd-1));
						subproblems[n].next_Ur = 0;
					}
				}	
				else{
					subproblems[n].next_Ul = GP.BCVal(q);
					subproblems[n].next_Ur = GP.BCVal(q+1);
				}
				subproblems[n].color = ip;
			} // end piece loop
		}


		printf("CREATE PARTITION\n");
		LogicalPartition all_subproblems_lp = load_all_subdomains(ctx, runtime, GP, subproblems, num_pieces, subproblems_per_piece);
		printf("LOGICAL PARTITION SUCCESSFULY CREATED!\n");

		



};



void allocate_subproblem_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	// allocator.allocate_field(sizeof(Subdomain), MAIN_SUBDOMAIN);
	allocator.allocate_field(sizeof(double), NEXT_LEFT_BC);
	allocator.allocate_field(sizeof(double), NEXT_RIGHT_BC);
	allocator.allocate_field(sizeof(Point<1>), COLOR);
};


LogicalPartition load_all_subdomains(Context ctx, Runtime *runtime, GlobalProblem GP, std::vector<SubProblem> subproblems, 
	                       int num_pieces, int subproblems_per_piece){

	printf("Create Index Space\n");
	Rect<1> color_bounds(0,num_pieces-1);
	IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

	printf("Create region of requirements\n");
	RegionRequirement all_subproblems_req(GP.all_subproblems_lr, READ_WRITE, EXCLUSIVE, GP.all_subproblems_lr);
	// all_subproblems_req.add_field(MAIN_SUBDOMAIN);
	all_subproblems_req.add_field(NEXT_LEFT_BC);
	all_subproblems_req.add_field(NEXT_RIGHT_BC);
	all_subproblems_req.add_field(COLOR);

	printf("Create physical region\n");
	PhysicalRegion all_subdomains_pr = runtime->map_region(ctx, all_subproblems_req);

	printf("Create accessors\n");
	// const FieldAccessor<READ_WRITE, Subdomain, 1> acc_subdomain(all_subdomains_pr, MAIN_SUBDOMAIN);
	const FieldAccessor<READ_WRITE, double, 1> acc_left(all_subdomains_pr, NEXT_LEFT_BC);
	const FieldAccessor<READ_WRITE, double, 1> acc_right(all_subdomains_pr, NEXT_RIGHT_BC);
	const FieldAccessor<READ_WRITE, Point<1>, 1> acc_color(all_subdomains_pr, COLOR);

	printf("Fill pointers\n");
	{
		int ip = 0;
		for (int n = 0; n < num_pieces; n++)
		{
			for (int i = 0; i < subproblems_per_piece; i++)
			{
				const Point<1> ptr(n * subproblems_per_piece + i);
				ip++;

				// acc_subdomain[ptr] = subproblem.main_subdomain;
				acc_left[ptr] = subproblems[ip].next_Ul;
				acc_right[ptr] = subproblems[ip].next_Ur;
				acc_color[ptr] = subproblems[ip].color;

			}
		}
	}

	IndexPartition ip = runtime->create_partition_by_field(ctx, GP.all_subproblems_lr,
                                                               GP.all_subproblems_lr,
                                                               COLOR,
                                                               color_is);
	LogicalPartition all_subdomains_lp = runtime->get_logical_partition(ctx, GP.all_subproblems_lr, ip);
	runtime->attach_name(all_subdomains_lp, "all_subdomains_lp");

	return all_subdomains_lp;


}




int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}

	// {
	//     TaskVariantRegistrar registrar(UPDATE_BCS_TASK_ID, "init_bcs");
	//     registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	//     Runtime::preregister_task_variant<update_bcs_task>(registrar, "init_bcs");
	// }

	return Runtime::start(argc, argv);
}