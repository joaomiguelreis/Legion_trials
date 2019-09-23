1§§#include "schwarz_solver.h"
#include "../kldec.h"


MatrixXd SetKLModes(int const nel, VectorXd &Lam, double len_kl, double var_f);



void allocate_inner_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_edge_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_middle_elem(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_edge_elem(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_matrix_fields(Context ctx, Runtime *runtime, FieldSpace field_space);


/*
---------------------------------------------------
FORWARD DECLARATION OF TASKS
---------------------------------------------------
*/

void init_spatial_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime);


void loc_solver_task(const Task *task,
			         const std::vector<PhysicalRegion> &regions,
			         Context ctx, Runtime *runtime);

void updateBC_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime);

void display_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime);

/*
---------------------------------------------------
FORWARD DECLARATION OF TASKS
---------------------------------------------------
*/


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime){

	std::default_random_engine generator;
	std::normal_distribution<double> G(.0,1.0);


	int Ne = 100; // non overlapping elements per subdomain
	int Nd = 10; // number of subdomains
	int No = 10; // overlapping elements per subdomain
	int num_pieces = Nd;
	int Net = Nd*Ne+No;  // number of total elements (added "no" elements to last subdomain)
	double tol = 1e-12; // tolerance
	int num_elements_subdomain = Ne+No;
	double len_kl = 0.1; //correlation length of log(k)
	double var_kl = 1; //var of log(k)
	double mu_kl = 0; // mean of k
	double u0 = 1; //left dirichlet BC
	double u1 = 0; //right Dirichlet BC

	Indices indices;
	indices.num_pieces = num_pieces;
	indices.elements_per_subdomain = num_elements_subdomain;
	indices.num_overlapping_elem = No;
	indices.num_total_elements = Net;
	indices.tolerance = tol;
	indices.len_kl = len_kl;
	indices.var_kl = var_kl;
	indices.mu_kl = mu_kl;
	indices.u0 = u0;
	indices.u1 = u1;
	

	printf("Define Index Space\n");
	Rect<1> rect_main(0,num_pieces-1);
	IndexSpaceT<1> one_dimensional_is = runtime->create_index_space(ctx, rect_main);
	runtime->attach_name(one_dimensional_is, "main_1d_ndex_space");
	Rect<1> rect_middle_elem(0,Nd*(Ne+No)-1);
	IndexSpaceT<1> middle_elem_is = runtime->create_index_space(ctx, rect_middle_elem);
	runtime->attach_name(middle_elem_is, "index for element mid points");
	Rect<1> rect_edge_elem(0,Nd*(Ne+No+1)-1);
	IndexSpaceT<1> edge_elem_is = runtime->create_index_space(ctx, rect_edge_elem);
	runtime->attach_name(edge_elem_is, "index for element edge points");

	Rect<2> matrix_main = Rect<2>(Point<2>(0,0),Point<2>(num_pieces-1,0));
	IndexSpaceT<2> two_dimensional_is = runtime->create_index_space(ctx, matrix_main);
	runtime->attach_name(two_dimensional_is, "main_2d_index_space");
	Rect<2> matrix_spatial = Rect<2>(Point<2>(0,0),Point<2>(Nd*(Ne+No-1) -1,Ne+No-1 -1));   // we store the inner part of matrices
																							// which excludes the bcs
	IndexSpaceT<2> matrix_is = runtime->create_index_space(ctx, matrix_spatial);
	runtime->attach_name(matrix_is, "2d index for spatial settings");

	// Make field spaces
	FieldSpace edge_fs = runtime->create_field_space(ctx);
	runtime->attach_name(edge_fs, "field space for edge nodes");
	FieldSpace inner_fs = runtime->create_field_space(ctx);
	runtime->attach_name(inner_fs, "field space for inner nodes and bool");
	FieldSpace middle_elem_fs = runtime->create_field_space(ctx);
	runtime->attach_name(middle_elem_fs, "field space for space settings using mid pts elem (field, source)");
	FieldSpace edge_elem_fs = runtime->create_field_space(ctx);
	runtime->attach_name(edge_elem_fs, "field space for space settings using edge pts elem (mesh and solution)");
	FieldSpace matrix_fs = runtime->create_field_space(ctx);
	runtime->attach_name(matrix_fs, "field space for the local fe matrices");

	// Allocate fields
	printf("Allocate field spaces\n");
	allocate_edge_fields(ctx, runtime, edge_fs);
	allocate_inner_fields(ctx, runtime, inner_fs);
	allocate_middle_elem(ctx, runtime, middle_elem_fs);
	allocate_edge_elem(ctx, runtime, edge_elem_fs);
	allocate_matrix_fields(ctx, runtime, matrix_fs);


	//Finally create logical region
	printf("Create logical region\n");
	LogicalRegion edge_lr = runtime->create_logical_region(ctx, one_dimensional_is, edge_fs);
	runtime->attach_name(edge_lr, "region with the edge nodes");
	LogicalRegion inner_lr = runtime->create_logical_region(ctx, one_dimensional_is, inner_fs);
	runtime->attach_name(inner_lr, "region with the inner node and boolean");
	LogicalRegion middle_elem_lr = runtime->create_logical_region(ctx, middle_elem_is, middle_elem_fs);
	runtime->attach_name(middle_elem_lr, "region with the space settings (field, source and mesh)");
	LogicalRegion edge_elem_lr = runtime->create_logical_region(ctx, edge_elem_is, edge_elem_fs);
	runtime->attach_name(edge_elem_lr, "region with the space settings (field, source and mesh)");
	LogicalRegion matrix_lr = runtime->create_logical_region(ctx, matrix_is, matrix_fs);
	runtime->attach_name(matrix_lr, "region with the local fe matrices");

	



	// CREATE PARTITIONS FOR NODES
	printf("Create partitions for nodes\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, one_dimensional_is, one_dimensional_is);
	LogicalPartition edge_lp = runtime->get_logical_partition(ctx, edge_lr, ip);
	LogicalPartition inner_lp = runtime->get_logical_partition(ctx, inner_lr, ip);

	// CREATE PARTITIONS FOR SPATIAL SETTINGS
	printf("Creating partition for spatial settings\n");
	DomainPoint middle_elem_blocking_factor = Point<1>(Ne+No);
	IndexPartition middle_elem_ip = runtime->create_partition_by_blockify(ctx, middle_elem_is, middle_elem_blocking_factor);
	DomainPoint edge_elem_blocking_factor = Point<1>(Ne+No+1);
	IndexPartition edge_elem_ip = runtime->create_partition_by_blockify(ctx, edge_elem_is, edge_elem_blocking_factor);

	LogicalPartition middle_elem_lp = runtime->get_logical_partition(ctx, middle_elem_lr, middle_elem_ip);
	LogicalPartition edge_elem_lp = runtime->get_logical_partition(ctx, edge_elem_lr, edge_elem_ip);

	DomainPoint blocking_factor = Point<2>(Ne+No-1, Ne+No-1 +1);
	IndexPartition matrix_ip = runtime->create_partition_by_blockify(ctx, matrix_is, blocking_factor);
	LogicalPartition matrix_lp = runtime->get_logical_partition(ctx, matrix_lr, matrix_ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");




	ArgumentMap arg_map;
	for (int piece=0; piece<num_pieces; piece++){
		Point<1> point(piece);
		arg_map.set_point(point, TaskArgument(&piece, sizeof(int)));
	}

	InlineLauncher KL_launcher(RegionRequirement(middle_elem_lr, WRITE_DISCARD, EXCLUSIVE, middle_elem_lr));
	KL_launcher.requirement.add_field(0,FIELD_ID);
	PhysicalRegion KL_region = runtime->map_region(ctx, KL_launcher);

	/* Find KL modes */
	VectorXd Lam;
	MatrixXd Modes = SetKLModes(Net, Lam, len_kl, var_kl);
	for(int e=0; e<Net; e++) Modes.col(e) *= Lam(e);

	Rect<2> domain = Rect<2>(Point<2>(0,0),Point<2>(num_pieces-1,0));


	// CONSTRUCT SPATIAL SETTINGS
	IndexTaskLauncher spatial_launcher(INIT_SPATIAL_TASK_ID, domain, TaskArgument(&indices, sizeof(Indices)), arg_map);

	RegionRequirement middleElemRW_req(middle_elem_lp, 0, READ_WRITE, EXCLUSIVE, middle_elem_lr);
	RegionRequirement edgeElemWO_req(edge_elem_lp, 0, WRITE_ONLY, EXCLUSIVE, edge_elem_lr);
	RegionRequirement edgeWO_req(edge_lp, 0, WRITE_ONLY, EXCLUSIVE, edge_lr);
	RegionRequirement innerWO_req(inner_lp, 0, WRITE_ONLY, EXCLUSIVE, inner_lr);
	RegionRequirement matrixWO_req(matrix_lp, 0, WRITE_ONLY, EXCLUSIVE, matrix_lr);

	spatial_launcher.add_region_requirement(middleElemRW_req);
	spatial_launcher.add_region_requirement(edgeElemWO_req);
	spatial_launcher.add_region_requirement(edgeWO_req);
	spatial_launcher.add_region_requirement(innerWO_req);
	spatial_launcher.add_region_requirement(matrixWO_req);

	spatial_launcher.region_requirements[0].add_field(0,FIELD_ID);
	spatial_launcher.region_requirements[0].add_field(1,SOURCE_ID);

	spatial_launcher.region_requirements[1].add_field(0,MESH_ID);
	spatial_launcher.region_requirements[1].add_field(1,SOLUTION_ID);

	spatial_launcher.region_requirements[2].add_field(0,EDGE_LEFT_ID);
	spatial_launcher.region_requirements[2].add_field(1,EDGE_RIGHT_ID);

	spatial_launcher.region_requirements[3].add_field(0,INNER_LEFT_ID);
	spatial_launcher.region_requirements[3].add_field(1,INNER_RIGHT_ID);
	spatial_launcher.region_requirements[3].add_field(2,DONE_ID);
	spatial_launcher.region_requirements[3].add_field(3,WRITE_ID);

	spatial_launcher.region_requirements[4].add_field(0, MASS_MATRIX_ID);


	// Local Solver
	IndexTaskLauncher loc_solver_launcher(LOC_SOLVER_TASK_ID, domain, TaskArgument(&indices,sizeof(Indices)), arg_map);
	RegionRequirement edgeRO_req(edge_lp, 0, READ_ONLY, EXCLUSIVE, edge_lr);
	RegionRequirement matrixRO_req(matrix_lp, 0, READ_ONLY, EXCLUSIVE, matrix_lr);
	RegionRequirement middleElemRO_req(middle_elem_lp, 0, READ_ONLY, EXCLUSIVE, middle_elem_lr);
	RegionRequirement edgeElemRO_req(edge_elem_lp, 0, READ_ONLY, EXCLUSIVE, edge_elem_lr);
	loc_solver_launcher.add_region_requirement(edgeRO_req);
	loc_solver_launcher.add_region_requirement(innerWO_req);
	loc_solver_launcher.add_region_requirement(middleElemRO_req);
	loc_solver_launcher.add_region_requirement(edgeElemRO_req);
	loc_solver_launcher.add_region_requirement(matrixRO_req);
	loc_solver_launcher.region_requirements[0].add_field(0,EDGE_LEFT_ID);
	loc_solver_launcher.region_requirements[0].add_field(1,EDGE_RIGHT_ID);
	loc_solver_launcher.region_requirements[1].add_field(0,INNER_LEFT_ID);
	loc_solver_launcher.region_requirements[1].add_field(1,INNER_RIGHT_ID);
	loc_solver_launcher.region_requirements[1].add_field(2,DONE_ID);
	loc_solver_launcher.region_requirements[1].add_field(3,WRITE_ID);
	loc_solver_launcher.region_requirements[2].add_field(0,FIELD_ID);
	loc_solver_launcher.region_requirements[2].add_field(1,SOURCE_ID);
	loc_solver_launcher.region_requirements[3].add_field(0,MESH_ID);
	loc_solver_launcher.region_requirements[3].add_field(1,SOLUTION_ID);
	loc_solver_launcher.region_requirements[4].add_field(0,MASS_MATRIX_ID);

	InlineLauncher done_launcher(RegionRequirement(inner_lr, READ_WRITE, EXCLUSIVE, inner_lr));
	done_launcher.requirement.add_field(2,DONE_ID);
	PhysicalRegion done_region = runtime->map_region(ctx, done_launcher);

	//UPDATE BC
	IndexTaskLauncher updateBC_launcher(UPDATE_BC_TASK_ID, domain, TaskArgument(&indices,sizeof(Indices)), arg_map);
	RegionRequirement innerRO_req(inner_lp, 0, READ_ONLY, EXCLUSIVE, inner_lr);
	updateBC_launcher.add_region_requirement(edgeWO_req);
	updateBC_launcher.add_region_requirement(innerRO_req);
	updateBC_launcher.region_requirements[0].add_field(0,EDGE_LEFT_ID);
	updateBC_launcher.region_requirements[0].add_field(1,EDGE_RIGHT_ID);
	updateBC_launcher.region_requirements[1].add_field(0,INNER_LEFT_ID);
	updateBC_launcher.region_requirements[1].add_field(1,INNER_RIGHT_ID);
	updateBC_launcher.region_requirements[1].add_field(2,DONE_ID);



	// Start MC loop
	for(int samples=0; samples<1; samples++){

		printf("sample %3d \n", samples);


		const FieldAccessor<WRITE_DISCARD,double,1> acc_field(KL_region, FIELD_ID);

		/* KL Expansion */
		VectorXd KF = VectorXd::Ones(Net)*mu_kl; // full KL-epansion
		for(int e=0; e<Net; e++){
			double va = G(generator);
			KF += Modes.col(e)*va;	
		}
		KF.array() = KF.array().exp();

		int ned=Ne+No;
		int id = 0;
		int elem=0;
		for(int n=0; n<Nd; n++){
			if (n==0){
				for(int e=0; e<Ne+No; e++){
					id = n * ned + e;
					elem = n * Ne + e;
					acc_field[id] = KF(elem);
				}
			}
			else{
				for(int e=0; e<No; e++){
					id = n * ned + e;
					acc_field[id] = acc_field[id-No];

				}
				for(int e=No; e<Ne+No; e++){
					id = n * ned + e;
					elem = n * Ne + e;
					acc_field[id] = KF(elem);

				}
			}
			
		}

		// CONSTRUCT SPATIAL SETTINGS
		runtime->execute_index_space(ctx, spatial_launcher);


		// START ITERATION
		bool all_done = false;
		int iterations = 0;
		while(all_done==false){


			
			runtime->execute_index_space(ctx, loc_solver_launcher);


			const FieldAccessor<READ_ONLY,bool,1> acc_done(done_region, DONE_ID);



			bool success = true;
			for(int n=0; n<num_pieces; n++){
				success = acc_done[Point<1>(n)] && success;
				if(success==false) break;
			}
			if (success==true){
				all_done = true;
			} 

			
			runtime->execute_index_space(ctx, updateBC_launcher);

			iterations ++;
		} 
		// END SCHWARZ ITERATIONS


		// FOR DISPLAY
		TaskLauncher display_task(DISPLAY_TASK_ID, TaskArgument(&indices, sizeof(Indices)));
		RegionRequirement edgeElemRO_req2(edge_elem_lr, READ_WRITE, EXCLUSIVE, edge_elem_lr);
		display_task.add_region_requirement(edgeElemRO_req2);
		display_task.region_requirements[0].add_field(1,SOLUTION_ID);
		runtime->execute_task(ctx, display_task);

		printf("\n Classic Schwar: solved in %3d iterations local gap of %12.0e\n", iterations, tol);
	}

}

void allocate_edge_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), EDGE_LEFT_ID);
	allocator.allocate_field(sizeof(double), EDGE_RIGHT_ID);
	allocator.allocate_field(sizeof(double), MESH_LEFT_ID);
	allocator.allocate_field(sizeof(double), MESH_RIGHT_ID);
};

void allocate_inner_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), INNER_LEFT_ID);
	allocator.allocate_field(sizeof(double), INNER_RIGHT_ID);
	allocator.allocate_field(sizeof(bool), DONE_ID);
	allocator.allocate_field(sizeof(bool), WRITE_ID);
};

void allocate_middle_elem(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), FIELD_ID);
	allocator.allocate_field(sizeof(double), SOURCE_ID);
};

void allocate_edge_elem(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), MESH_ID);
	allocator.allocate_field(sizeof(double), SOLUTION_ID);
};

void allocate_matrix_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), MASS_MATRIX_ID);
};




int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}


	{
		TaskVariantRegistrar registrar(INIT_SPATIAL_TASK_ID, "init_spatial_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<init_spatial_task>(registrar, "init_spatial_task");
	}

	{
		TaskVariantRegistrar registrar(LOC_SOLVER_TASK_ID, "loc_solver_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<loc_solver_task>(registrar, "loc_solver_task");
	}


	{
		TaskVariantRegistrar registrar(UPDATE_BC_TASK_ID, "updateBC_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<updateBC_task>(registrar, "updateBC_task");
	}

	{
		TaskVariantRegistrar registrar(DISPLAY_TASK_ID, "display_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<display_task>(registrar, "display_task");
	}

	

	

	return Runtime::start(argc, argv);
}