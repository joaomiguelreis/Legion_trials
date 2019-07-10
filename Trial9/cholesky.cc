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

struct Indices{
	int subdomain_index;
	int iteration_index;
	int elements_per_subdomain;
};


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{

	int num_pieces = 10;

	//printf("Define Index Space\n");
	Rect<1> rect(0,num_pieces-1);
	IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
	runtime->attach_name(is, "main_index_space");

	// Make field spaces
	//printf("Define field spaces boundary values\n");
	FieldSpace fs = runtime->create_field_space(ctx);
	runtime->attach_name(fs, "field_space");

	//Finally create logical region all_subproblems
	//printf("Create logical region\n");
	LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
	runtime->attach_name(lr, "lr");


	// CREATE PARTITIONS
	//printf("START CREATING PARTITIONS\n");
	IndexPartition ip = runtime->create_equal_partition(ctx, is, is);
	LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
	//printf("PARTITIONS SUCCESSFULLY CREATED\n");

	FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
	allocator.allocate_field(sizeof(double), FIELD_ID);
	// allocator.allocate_field(sizeof(int), INDEX_ID);

	ArgumentMap arg_map;
	for (int n=0; n<num_pieces; n++){
		Point<1> point(n);
		arg_map.set_point(point, TaskArgument(&n, sizeof(int)));
	}

	//
	printf("GLOBAL TASK\n");
	IndexTaskLauncher init_launcher(INIT_TASK_ID, is, TaskArgument(NULL,0), arg_map);
	RegionRequirement init_req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
	init_launcher.add_region_requirement(init_req);
	init_launcher.region_requirements[0].add_field(0,FIELD_ID);
	// init_launcher.region_requirements[0].add_field(1,INDEX_ID);
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

	int subdomain_index = *((const int*)task->local_args);
	cout << "-  subdomain number " << subdomain_index << endl;
	assert(subdomain_index==task->index_point.point_data[0]);

	/*  INPUTS */
	int Ne = 10;
	unsigned Nd = 3;
	unsigned No = 3; 
	// double mu_kl = 1;
	// double var_kl = 0.1;
	// double len_kl = 0.1;
	// double prec = 1e-9;
	// int M = 10;
	unsigned Net = Nd*Ne+No;  // number of total elements (added "no" elements to last subdomain)


	/* Spatial settings */
	double dx = 1 / float(Net); 
	VectorXd MeshG = VectorXd::Zero(Net+1);

	//printf("Do spatial mesh\n");
	for (int e = 0; e < Net+1; e++) MeshG(e) = e*dx;
	VectorXd SF = VectorXd::Ones(Net);
	VectorXd KF = VectorXd::Ones(Net);
	/*  INPUTS */

	const FieldAccessor<READ_WRITE, double, 1> acc(regions[0], FIELD_ID);
	// const FieldAccessor<READ_WRITE, int, 1> acc_index(regions[0], INDEX_ID);

	int index = task->index_point.point_data[0];

	double my_d = 2;
	Point<1> point(index);

	acc[point] = my_d;
	// acc_index[point] = index
	// cout << "my_d" << endl;
	// cout << my_d << endl;

	LogicalRegion lr = regions[0].get_logical_region();

	Indices indices;
	indices.subdomain_index = subdomain_index;
	indices.elements_per_subdomain = Ne;


	//
	printf("LOCAL TASK\n");
	for (int n=0; n<3; n++){
		printf("ITERATION %3d \n", n);
		indices.iteration_index = n;
		TaskLauncher output_launcher(OUTPUT_TASK_ID, TaskArgument(&indices,sizeof(Indices)));
		RegionRequirement out_req(lr, READ_ONLY, EXCLUSIVE, lr);
		output_launcher.add_region_requirement(out_req);
		output_launcher.region_requirements[0].add_field(0,FIELD_ID);
		// output_launcher.region_requirements[0].add_field(1,INDEX_ID);
		runtime->execute_task(ctx,output_launcher);	
	}

	cout << "Next piece" << endl;
	cout << " " << endl;
	




}

void output_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime){



	const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FIELD_ID);

	Indices indices = *((const Indices*)task->args);

	int ned = indices.elements_per_subdomain;

	//static map< int, SimplicialCholesky<SpMat> > Oper_for_point;
	static map<int,SimplicialCholesky<SpMat>* > chol_ptr_for_point;
	static map<int,int> chol_for_point_is_valid_for_timestep;
	static std::mutex cache_mutex;
	int index_point = task->parent_task->index_point.point_data[0]; // find the index point this instance of T2 was called on
	cout << "-   current index point " << index_point << endl;
	int curr_timestep = indices.subdomain_index;  // retrieve from task arguments
	cout << "-   iteration " << curr_timestep << endl;
	//SpMat Oper;
	SimplicialCholesky<SpMat>* chol_ptr;
	int valid_timestep;
	{
	std::lock_guard<std::mutex> guard(cache_mutex);
	//Oper = Oper_for_point[index_point];
	chol_ptr = chol_ptr_for_point[index_point];
	valid_timestep = chol_for_point_is_valid_for_timestep[index_point];
	}
	cout << "-    valid for subdomain " << valid_timestep << endl;
	cout << "-    current subdomain " << curr_timestep << endl;
	if (chol_ptr_for_point[index_point] == NULL || valid_timestep != curr_timestep) {

		cout << "I AM COMPUTING CHOLESKY BECAUSE THE TWO PREVIOUS INTEGERS ARE DIFFERENT!!!!" << endl;
		//cout << "iteration " << curr_timestep << endl;

		//unsigned ned=10;
		vector<T> cprec; //inititate the vector with values of FEM integral

		//Loop over elements
		for(int e=0; e<ned; e++){				
			double kv = 1/fabs(0.1); // k(x_e)/dx	

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
		SpMat Oper = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
		Oper.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
														 Oper_(e1,e2) = kappa_value */
		SimplicialCholesky<SpMat> chol;
		chol.compute(Oper);
		chol_ptr = &chol;


		//cout << chol << endl;
		std::lock_guard<std::mutex> guard(cache_mutex);
		if (chol_ptr_for_point[index_point] != NULL) {
			//Oper_for_point.erase(index_point); // remove old value from cache
			chol_ptr_for_point.erase(index_point); // remove old value from cache
		}

		//Oper_for_point[index_point] = Oper;
		chol_ptr_for_point[index_point] = chol_ptr;
		chol_for_point_is_valid_for_timestep[index_point] = curr_timestep;
	}
	

	Point<1> point(index_point);

	double d = acc[point];

	VectorXd b = VectorXd::Constant(ned-1,1);
	if(b.rows()!=ned-1) cout << "Mismatch\n";
	
	VectorXd Usol = VectorXd(ned+1);

	//SimplicialCholesky<SpMat> cholesky;
	chol_ptr = chol_ptr_for_point[index_point];
	//cholesky.compute(Oper); /*  find cholesky decomposition. This will be used to solve the FE 
							//linear system */

	Usol.segment(1,ned-1) = (*chol_ptr).solve(b);

	Usol(0) = d; Usol(ned) = d;
	cout << "-    Solution " << endl;
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