#include <cstdio>
#include <iostream>


#include "legion.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>



using namespace std;
using namespace Eigen;
using namespace Legion;

enum TaskIDs{
	TOP_LEVEL_TASK_ID,
	INIT_TASK_ID,
	OUTPUT_TASK_ID,
};

enum FieldsID{
	FIELD_ID,
};


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{

	int num_pieces = 100;

	printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");

	// Make field spaces
	printf("Define field spaces boundary values\n");
	FieldSpace fs = runtime->create_field_space(ctx);
	runtime->attach_name(fs, "field_space");

	//Finally create logical region all_subproblems
	printf("Create logical region\n");
	LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
	runtime->attach_name(lr, "lr");


	// CREATE PARTITIONS
	printf("START CREATING PARTITIONS\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);
	LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");

	FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
	allocator.allocate_field(sizeof(VectorXd), FIELD_ID);

	ArgumentMap arg_map;

	//
	printf("Initiate task\n");
	IndexTaskLauncher init_launcher(INIT_TASK_ID, is, TaskArgument(NULL,0), arg_map);
	RegionRequirement init_req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
	init_launcher.add_region_requirement(init_req);
	init_launcher.region_requirements[0].add_field(0,FIELD_ID);
	runtime->execute_index_space(ctx,init_launcher);

	init_launcher.region_requirements[0].privilege_fields.clear();
	init_launcher.region_requirements[0].instance_fields.clear();


	//
	printf("Output task\n");
	IndexTaskLauncher output_launcher(OUTPUT_TASK_ID, is, TaskArgument(NULL,0), arg_map);
	RegionRequirement out_req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
	output_launcher.add_region_requirement(out_req);
	output_launcher.region_requirements[0].add_field(0,FIELD_ID);
	runtime->execute_index_space(ctx,output_launcher);

	output_launcher.region_requirements[0].privilege_fields.clear();
	output_launcher.region_requirements[0].instance_fields.clear();

	// 
	runtime->destroy_logical_region(ctx, lr);
	runtime->destroy_field_space(ctx, fs);
	runtime->destroy_index_space(ctx, is);
	cout << "finished" << endl;



}

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime){
	cout << "here" << endl;

	const FieldAccessor<READ_WRITE, VectorXd, 1> acc(regions[0], FIELD_ID);

	int index = task->index_point.point_data[0];
	cout << "index " << endl;
	cout << index << endl;

	VectorXd field = VectorXd::Ones(10);

	cout << "make point" << endl;
	Point<1> point(index);

	acc[point] = field;
	VectorXd my_field = acc[point];
	cout << "my_field" << endl;
	cout << my_field.transpose() << endl;

}

void output_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime){
	cout << "here" << endl;

	const FieldAccessor<READ_ONLY, VectorXd, 1> acc(regions[0], FIELD_ID);

	int index = task->index_point.point_data[0];
	cout << "index " << endl;
	cout << index << endl;

	cout << "make point" << endl;
	Point<1> point(index);

	VectorXd my_field = acc[point];
	cout << "another_field" << endl;
	cout << my_field.transpose() << endl;

}

int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}


	{
	  TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	  // registrar.set_leaf(true);
	  Runtime::preregister_task_variant<init_task>(registrar, "init_task");
	}

	{
	  TaskVariantRegistrar registrar(OUTPUT_TASK_ID, "output_task");
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	  // registrar.set_leaf(true);
	  Runtime::preregister_task_variant<output_task>(registrar, "output_task");
	}

	return Runtime::start(argc, argv);
}