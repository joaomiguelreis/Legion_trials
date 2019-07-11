#include <cstdio>
#include <iostream>

#include "legion.h"

#include <Eigen/Dense>

using namespace std;
using namespace Legion;
using namespace Eigen;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  WRITING_TASK_ID,
  READING_TASK_ID,
  INIT_TASK_ID,
};

enum EdgeFieldsID{
	REG1_ID,
};

enum InnerFieldsID{
	REG2_ID,
};



void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime){

	int num_pieces = 5;

	printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");

	// Make field spaces
	FieldSpace reg1_fs = runtime->create_field_space(ctx);
	runtime->attach_name(reg1_fs, "field space for reg1");

	// Make field spaces
	FieldSpace reg2_fs = runtime->create_field_space(ctx);
	runtime->attach_name(reg2_fs, "field space for reg2");

	// Allocate fields
	printf("Allocate field spaces\n");
	FieldAllocator allocator1 = runtime->create_field_allocator(ctx, reg1_fs);
	allocator1.allocate_field(sizeof(double), REG1_ID);
	FieldAllocator allocator2 = runtime->create_field_allocator(ctx, reg2_fs);
	allocator2.allocate_field(sizeof(double), REG2_ID);


	//Finally create logical region all_subproblems
	printf("Create logical region\n");
	LogicalRegion reg1_lr = runtime->create_logical_region(ctx, is, reg1_fs);
	runtime->attach_name(reg1_lr, "region with the writing");
	LogicalRegion reg2_lr = runtime->create_logical_region(ctx, is, reg2_fs);
	runtime->attach_name(reg2_lr, "regin with the reading");

	// CREATE PARTITIONS
	printf("START CREATING PARTITIONS\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);

	LogicalPartition reg1_lp = runtime->get_logical_partition(ctx, reg1_lr, ip);
	LogicalPartition reg2_lp = runtime->get_logical_partition(ctx, reg2_lr, ip);
	printf("PARTITIONS SUCCESSFULLY CREATED\n");

	ArgumentMap arg_map;

	IndexTaskLauncher init_launcher(INIT_TASK_ID, is, TaskArgument(&num_pieces,sizeof(int)), arg_map);
	RegionRequirement reg1WO_req(reg1_lp, 0, WRITE_ONLY, EXCLUSIVE, reg1_lr);
	RegionRequirement reg2WO_req(reg2_lp, 0, WRITE_ONLY, EXCLUSIVE, reg2_lr);
	init_launcher.add_region_requirement(reg1WO_req);
	init_launcher.add_region_requirement(reg2WO_req);
	init_launcher.region_requirements[0].add_field(0,REG1_ID);
	init_launcher.region_requirements[1].add_field(0,REG2_ID);
	runtime->execute_index_space(ctx, init_launcher);

	IndexTaskLauncher task1_launcher(WRITING_TASK_ID, is, TaskArgument(&num_pieces,sizeof(int)), arg_map);
	//RegionRequirement reg1WO_req(reg1_lp, 0, WRITE_ONLY, EXCLUSIVE, reg1_lr);
	RegionRequirement reg2RO_req(reg2_lp, 0, READ_ONLY, EXCLUSIVE, reg2_lr);
	task1_launcher.add_region_requirement(reg1WO_req);
	task1_launcher.add_region_requirement(reg2RO_req);
	task1_launcher.region_requirements[0].add_field(0,REG1_ID);
	task1_launcher.region_requirements[1].add_field(0,REG2_ID);
	runtime->execute_index_space(ctx, task1_launcher);


	cout << " " << endl;

	//
	IndexTaskLauncher task2_launcher(READING_TASK_ID, is, TaskArgument(NULL,0), arg_map);
	RegionRequirement reg1RO_req(reg1_lp, 0, READ_ONLY, EXCLUSIVE, reg1_lr);
	//RegionRequirement reg2WO_req(reg2_lp, 0, WRITE_ONLY, EXCLUSIVE, reg2_lr);
	task2_launcher.add_region_requirement(reg1RO_req);
	task2_launcher.add_region_requirement(reg2WO_req);
	task2_launcher.region_requirements[0].add_field(0,REG1_ID);
	task2_launcher.region_requirements[1].add_field(0,REG2_ID);

	cout << "finished" << endl;



}

void init_reg(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime){

	cout << "WRITING ON REG 2" << endl;

	const FieldAccessor<WRITE_ONLY, double, 1> acc_reg1(regions[0], REG1_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_reg2(regions[1], REG2_ID);

	DomainPoint point = task->index_point;
	int index = point.point_data[0];

	


	cout << "-    index_point " << point << 
	     " with index " << index << endl;

	//cout << "num_pieces " << num_pieces << endl;



	if (index==0) acc_reg2[point] = 0;
	else acc_reg2[point] = index;
	acc_reg1[point] = 0;

	cout << "-    acc_reg1[point] " << acc_reg1[point] << " "
	     << "-    acc_reg2[point] " << acc_reg2[point] << endl;


}

void reg1_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime){

	cout << "WRITING ON REG 1\n";

	const FieldAccessor<WRITE_ONLY, double, 1> acc_reg1(regions[0], REG1_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_reg2(regions[1], REG2_ID);

	int num_pieces = *((const int*)task->args);


	DomainPoint point = task->index_point;
	int index = point.point_data[0];

	VectorXd U = VectorXd::Ones(5);
	if (index==0) U *= acc_reg2[num_pieces-1];
	else U *= acc_reg2[index-1];

	acc_reg1[index] = U(3);

	cout << "-    index " << index << endl;
	cout << "-    acc_reg1[point] " << acc_reg1[index] << " "
	     << "acc_reg2[point] " << acc_reg2[index] << endl;


}

void reg2_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime){
	
}


int main(int argc, char **argv){

	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}

	{
		TaskVariantRegistrar registrar(WRITING_TASK_ID, "reg1_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<reg1_task>(registrar, "reg1_task");
	}

	{
		TaskVariantRegistrar registrar(READING_TASK_ID, "reg2_task");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<reg2_task>(registrar, "reg2_task");
	}

	{
		TaskVariantRegistrar registrar(INIT_TASK_ID, "init_reg");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<init_reg>(registrar, "init_reg");
	}

	

	return Runtime::start(argc, argv);

}