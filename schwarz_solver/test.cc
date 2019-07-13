#include <cstdio>
#include <iostream>

#include <Eigen/Dense>

#include "../legion/runtime/legion.h"


using namespace std;
using namespace Eigen;
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  SUBDOMAIN_COLOR_TASK_ID,
};

enum SpatialFieldsID
{
	SUBDOMAIN_COLOR_ID,
};

struct LocalSpatial{
	int num_elements_subdomain;
	int num_pieces;
};

/*
---------------------------------------------------
FORWARD DECLARATION OF TASKS
---------------------------------------------------
*/

void color_spatial_fields_task(const Task *task,
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

	int Ne = 10;
	int Nd = 3;
	int No = 3; 
	int num_pieces = Nd;


	printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");
	Rect<1> rect_spatial(0,Nd*(Ne+No)-1);
	IndexSpaceT<1> is_spatial = runtime->create_index_space(ctx, rect_spatial);
	runtime->attach_name(is_spatial, "index for spatial settings");

	// Make field spaces
	FieldSpace spatial_fs = runtime->create_field_space(ctx);
	runtime->attach_name(spatial_fs, "field space for space settings (field, source and mesh)");

	// Allocate fields
	printf("Allocate field spaces\n");
	FieldAllocator allocator = runtime->create_field_allocator(ctx, spatial_fs);
	allocator.allocate_field(sizeof(Point<1>), SUBDOMAIN_COLOR_ID);


	//Finally create logical region all_subproblems
	printf("Create logical region\n");
	LogicalRegion spatial_lr = runtime->create_logical_region(ctx, is_spatial, spatial_fs);
	runtime->attach_name(spatial_lr, "regin with the space settings (field, source and mesh)");


	LocalSpatial settings;
	settings.num_elements_subdomain = Ne+No;
	settings.num_pieces = num_pieces;

	// COLOR SPATIAL FIELDS TASK
	TaskLauncher color_launcher(SUBDOMAIN_COLOR_TASK_ID, TaskArgument(&settings,sizeof(LocalSpatial)));
	RegionRequirement spatialWO_req(spatial_lr, READ_WRITE, EXCLUSIVE, spatial_lr);
	color_launcher.add_region_requirement(spatialWO_req);
	color_launcher.region_requirements[0].add_field(0, SUBDOMAIN_COLOR_ID);
	runtime->execute_task(ctx,color_launcher);


}




void color_spatial_fields_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime){

	cout << "COLOR TASK" << endl;

	const FieldAccessor<WRITE_ONLY, Point<1>, 1> acc_color(regions[0], SUBDOMAIN_COLOR_ID);

	LocalSpatial settings = *((const LocalSpatial*)task->args);
	int num_elements_subdomain = settings.num_elements_subdomain;
	int num_pieces = settings.num_pieces;

	VectorXd color = VectorXd::Zero(num_elements_subdomain*num_pieces);
	for(int n=0; n<num_pieces; n++){
		for (int e=0; e<num_elements_subdomain; e++){
			int id = n * num_elements_subdomain + e;
			acc_color[id] = n;
			color(id) = n;
		}
	}
	cout << color.transpose() << endl;

}



int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}

	{
		TaskVariantRegistrar registrar(SUBDOMAIN_COLOR_TASK_ID, "color_spatial_fields_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<color_spatial_fields_task>(registrar, "color_spatial_fields_task");
	}

	return Runtime::start(argc, argv);
}