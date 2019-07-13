#include "schwarz_solver.h"


void allocate_inner_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_edge_fields(Context ctx, Runtime *runtime, FieldSpace field_space);
void allocate_spatial_fields(Context ctx, Runtime *runtime, FieldSpace field_space);

/*
---------------------------------------------------
FORWARD DECLARATION OF TASKS
---------------------------------------------------
*/

void color_spatial_fields_task(const Task *task,
                    		   const std::vector<PhysicalRegion> &regions,
                    		   Context ctx, Runtime *runtime);

void init_fields_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime);

void init_spatial_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime);

bool loc_solver_task(const Task *task,
			         const std::vector<PhysicalRegion> &regions,
			         Context ctx, Runtime *runtime);

void updateBC_task(const Task *task,
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

	int Ne = 5;
	int Nd = 10;
	int No = 3; 
	int num_pieces = Nd;
	int Net = Nd*Ne+No;  // number of total elements (added "no" elements to last subdomain)
	double tol = 1e-6;
	int num_elements_subdomain = Ne+No;


	Indices indices;
	indices.num_pieces = num_pieces;
	indices.elements_per_subdomain = num_elements_subdomain;
	indices.num_overlapping_elem = No;
	indices.num_total_elements = Net;
	indices.tolerance = tol;
	

	printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");
	Rect<1> rect_spatial(0,Nd*(Ne+No)-1);
	IndexSpaceT<1> spatial_is = runtime->create_index_space(ctx, rect_spatial);
	runtime->attach_name(spatial_is, "index for spatial settings");

	// Make field spaces
	FieldSpace edge_fs = runtime->create_field_space(ctx);
	runtime->attach_name(edge_fs, "field space for edge nodes");
	FieldSpace inner_fs = runtime->create_field_space(ctx);
	runtime->attach_name(inner_fs, "field space for inner nodes and bool");
	FieldSpace spatial_fs = runtime->create_field_space(ctx);
	runtime->attach_name(spatial_fs, "field space for space settings (field, source and mesh)");

	// Allocate fields
	printf("Allocate field spaces\n");
	allocate_edge_fields(ctx, runtime, edge_fs);
	allocate_inner_fields(ctx, runtime, inner_fs);
	allocate_spatial_fields(ctx, runtime, spatial_fs);


	//Finally create logical region all_subproblems
	printf("Create logical region\n");
	LogicalRegion edge_lr = runtime->create_logical_region(ctx, is, edge_fs);
	runtime->attach_name(edge_lr, "region with the edge nodes");
	LogicalRegion inner_lr = runtime->create_logical_region(ctx, is, inner_fs);
	runtime->attach_name(inner_lr, "regin with the inner node and boolean");
	LogicalRegion spatial_lr = runtime->create_logical_region(ctx, spatial_is, spatial_fs);
	runtime->attach_name(spatial_lr, "regin with the space settings (field, source and mesh)");



	// CREATE PARTITIONS FOR NODES
	printf("Create partitions for nodes\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);
	LogicalPartition edge_lp = runtime->get_logical_partition(ctx, edge_lr, ip);
	LogicalPartition inner_lp = runtime->get_logical_partition(ctx, inner_lr, ip);

	LocalSpatial settings;
	settings.num_elements_subdomain =num_elements_subdomain;
	settings.num_pieces = num_pieces;

	// COLOR SPATIAL FIELDS TASK
	TaskLauncher color_launcher(SUBDOMAIN_COLOR_TASK_ID, TaskArgument(&settings,sizeof(LocalSpatial)));
	RegionRequirement spatialWO_color_req(spatial_lr, READ_WRITE, EXCLUSIVE, spatial_lr);
	color_launcher.add_region_requirement(spatialWO_color_req);
	color_launcher.region_requirements[0].add_field(3, SUBDOMAIN_COLOR_ID);
	runtime->execute_task(ctx,color_launcher);


	// CREATE PARTITIONS FOR SPATIAL SETTINGS
	printf("Creating partition for spatial settings\n");
	IndexPartition spatial_ip = runtime->create_equal_partition(ctx, spatial_is, is);
	//IndexPartition spatial_ip = runtime->create_partition_by_field(ctx, spatial_lr, spatial_lr, SUBDOMAIN_COLOR_ID, is);
	LogicalPartition spatial_lp = runtime->get_logical_partition(ctx, spatial_lr, spatial_ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");

	ArgumentMap spatial_map;
	for (int piece=0; piece<num_pieces; piece++){		
		// for (int e=0; e<num_elements_subdomain; e++){
			const Point<1> point(piece);
			spatial_map.set_point(point, TaskArgument(&piece, sizeof(int)));
		// }
		
	}

	// CONSTRUCT SPATIAL SETTINGS (INCLUDING KL EXPANSION)
	IndexTaskLauncher spatial_launcher(INIT_SPATIAL_TASK_ID, is, TaskArgument(&indices, sizeof(Indices)), spatial_map);
	RegionRequirement spatialWO_req(spatial_lp, 0, WRITE_ONLY, EXCLUSIVE, spatial_lr);
	spatial_launcher.add_region_requirement(spatialWO_req);	
	spatial_launcher.region_requirements[0].add_field(0,FIELD_ID);
	spatial_launcher.region_requirements[0].add_field(1,SOURCE_ID);
	spatial_launcher.region_requirements[0].add_field(2,MESH_ID);
	//spatial_launcher.region_requirements[0].add_field(3,SUBDOMAIN_COLOR_ID);
	runtime->execute_index_space(ctx, spatial_launcher);

	

	ArgumentMap arg_map;
	for (int piece=0; piece<num_pieces; piece++){
		Point<1> point(piece);
		arg_map.set_point(point, TaskArgument(&piece, sizeof(int)));
	}

	// INITIATE FIELDS TASK
	IndexTaskLauncher init_launcher(INIT_FIELDS_TASK_ID, is, TaskArgument(&indices,sizeof(Indices)), arg_map);
	RegionRequirement edgeWO_req(edge_lp, 0, WRITE_ONLY, EXCLUSIVE, edge_lr);
	RegionRequirement innerWO_req(inner_lp, 0, WRITE_ONLY, EXCLUSIVE, inner_lr);
	init_launcher.add_region_requirement(edgeWO_req);
	init_launcher.add_region_requirement(innerWO_req);
	init_launcher.region_requirements[0].add_field(0,EDGE_LEFT_ID);
	init_launcher.region_requirements[0].add_field(1,EDGE_RIGHT_ID);
	init_launcher.region_requirements[1].add_field(0,INNER_LEFT_ID);
	init_launcher.region_requirements[1].add_field(1,INNER_RIGHT_ID);
	init_launcher.region_requirements[1].add_field(2,DONE_ID);


	runtime->execute_index_space(ctx, init_launcher);

	bool all_done = false;

	FutureMap all_done_future;
	int iterations = 0;
	while(all_done==false){


		IndexTaskLauncher loc_solver_launcher(LOC_SOLVER_TASK_ID, is, TaskArgument(&indices,sizeof(Indices)), arg_map);
		RegionRequirement edgeWO_req(edge_lp, 0, READ_ONLY, EXCLUSIVE, edge_lr);
		RegionRequirement innerRW_req(inner_lp, 0, WRITE_ONLY, EXCLUSIVE, inner_lr);
		RegionRequirement spatialWO_req2(spatial_lp, 0, READ_ONLY, EXCLUSIVE, spatial_lr);
		loc_solver_launcher.add_region_requirement(edgeWO_req);
		loc_solver_launcher.add_region_requirement(innerRW_req);
		loc_solver_launcher.add_region_requirement(spatialWO_req2);
		loc_solver_launcher.region_requirements[0].add_field(0,EDGE_LEFT_ID);
		loc_solver_launcher.region_requirements[0].add_field(1,EDGE_RIGHT_ID);
		// loc_solver_launcher.region_requirements[0].add_field(2,MESH_LEFT_ID);
		// loc_solver_launcher.region_requirements[0].add_field(3,MESH_RIGHT_ID);
		loc_solver_launcher.region_requirements[1].add_field(0,INNER_LEFT_ID);
		loc_solver_launcher.region_requirements[1].add_field(1,INNER_RIGHT_ID);
		loc_solver_launcher.region_requirements[1].add_field(2,DONE_ID);
		loc_solver_launcher.region_requirements[2].add_field(0,FIELD_ID);
		loc_solver_launcher.region_requirements[2].add_field(1,SOURCE_ID);
		loc_solver_launcher.region_requirements[2].add_field(2,MESH_ID);
		all_done_future = runtime->execute_index_space(ctx, loc_solver_launcher);
		all_done_future.wait_all_results();


		bool success = true;
		for(int n=0; n<num_pieces; n++){
			success = all_done_future.get_result<bool>(n) && success;
		}

		if (success==true) all_done = true;

		ArgumentMap all_done_map(all_done_future);

		IndexTaskLauncher updateBC_launcher(UPDATE_BC_TASK_ID, is, TaskArgument(&indices,sizeof(Indices)), all_done_map);
		RegionRequirement edgeRO_req(edge_lp, 0, WRITE_ONLY, EXCLUSIVE, edge_lr);
		RegionRequirement innerWO_req(inner_lp, 0, READ_ONLY, EXCLUSIVE, inner_lr);
		updateBC_launcher.add_region_requirement(edgeRO_req);
		updateBC_launcher.add_region_requirement(innerWO_req);
		updateBC_launcher.region_requirements[0].add_field(0,EDGE_LEFT_ID);
		updateBC_launcher.region_requirements[0].add_field(1,EDGE_RIGHT_ID);
		updateBC_launcher.region_requirements[1].add_field(0,INNER_LEFT_ID);
		updateBC_launcher.region_requirements[1].add_field(1,INNER_RIGHT_ID);
		updateBC_launcher.region_requirements[1].add_field(2,DONE_ID);
		runtime->execute_index_space(ctx, updateBC_launcher);

		iterations ++;
	}

	printf("Schwarz solved in %3d iterations\n", iterations);


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
};

void allocate_spatial_fields(Context ctx, Runtime *runtime, FieldSpace field_space)
{
	FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
	allocator.allocate_field(sizeof(double), FIELD_ID);
	allocator.allocate_field(sizeof(double), SOURCE_ID);
	allocator.allocate_field(sizeof(double), MESH_ID);
	allocator.allocate_field(sizeof(Point<1>), SUBDOMAIN_COLOR_ID);
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
		Runtime::preregister_task_variant<init_fields_task>(registrar, "init_fields_task");
	}

	{
		TaskVariantRegistrar registrar(INIT_SPATIAL_TASK_ID, "init_spatial_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<init_spatial_task>(registrar, "init_spatial_task");
	}

	{
		TaskVariantRegistrar registrar(LOC_SOLVER_TASK_ID, "loc_solver_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<bool, loc_solver_task>(registrar, "loc_solver_task");
	}

	{
		TaskVariantRegistrar registrar(UPDATE_BC_TASK_ID, "updateBC_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<updateBC_task>(registrar, "updateBC_task");
	}

	{
		TaskVariantRegistrar registrar(SUBDOMAIN_COLOR_TASK_ID, "color_spatial_fields_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<color_spatial_fields_task>(registrar, "color_spatial_fields_task");
	}

	return Runtime::start(argc, argv);
}