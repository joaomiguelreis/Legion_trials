#include <cstdio>
#include <iostream>


#include "legion.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;



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

	int num_pieces = 10;

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
	allocator.allocate_field(sizeof(double), FIELD_ID);

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

	// output_launcher.region_requirements[0].privilege_fields.clear();
	// output_launcher.region_requirements[0].instance_fields.clear();

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

	const FieldAccessor<READ_WRITE, double, 1> acc(regions[0], FIELD_ID);

	int index = task->index_point.point_data[0];

	double my_d = 2;

	cout << "make point" << endl;
	Point<1> point(index);

	acc[point] = my_d;
	cout << "my_d" << endl;
	cout << my_d << endl;

	vector<T> cprec; //inititate the vector with values of FEM integral
	//Loop over elements
	unsigned ned = 10;
	for(int e=0; e<ned; e++){				
		double kv = 1/fabs(0.01); // k(x_e)/dx	

		if(e==0){	// first sudomain	
			cprec.push_back(T(e,e,kv)); 

		}else if (e==ned-1){ // last subdomain
			cprec.push_back(T(e-1,e-1,kv));

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

	SpMat OPer = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
	OPer.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
														 Oper_(e1,e2) = kappa_value */


	LogicalRegion lr = regions[0].get_logical_region();

	//
	printf("Output task\n");
	TaskLauncher output_launcher(OUTPUT_TASK_ID, TaskArgument(&OPer,sizeof(OPer)));
	RegionRequirement out_req(lr, READ_ONLY, EXCLUSIVE, lr);
	output_launcher.add_region_requirement(out_req);
	output_launcher.region_requirements[0].add_field(0,FIELD_ID);
	runtime->execute_task(ctx,output_launcher);




}

void output_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime){

	SpMat OPer = *((const SpMat*) task->args);
	const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FIELD_ID);

	int index = task->index_point.point_data[0];
	unsigned ned = 10;
	// cout << "index " << endl;
	// cout << index << endl;

	cout << "make point" << endl;
	Point<1> point(index);

	double d = acc[point];

	VectorXd b = VectorXd::Constant(ned-1,1);
	if(b.rows()!=ned-1) cout << "Mismatch\n";
	SimplicialCholesky<SpMat> chol;
	chol.compute(OPer); /*  find cholesky decomposition. This will be used to solve the FE 
							linear system */
	VectorXd Usol = VectorXd(ned+1);
	Usol.segment(1,ned-1) = chol.solve(b);

	Usol(0) = d; Usol(ned) = d;
	cout << " solution " << endl;
	cout << Usol.transpose() << endl;
	cout << " " << endl;

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