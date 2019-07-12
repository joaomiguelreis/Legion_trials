#include "schwarz_solver.h"



void init_fields_task(const Task *task,
			          const std::vector<PhysicalRegion> &regions,
			          Context ctx, Runtime *runtime){

	cout << "init_fields_task" << endl;


	Indices indices = *((const Indices*)task->args);
	int current_subdomain = *((const int*)task->local_args);



	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_left(regions[0], EDGE_LEFT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_right(regions[0], EDGE_RIGHT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_inner_left(regions[1], INNER_LEFT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_inner_right(regions[1], INNER_RIGHT_ID);
	const FieldAccessor<WRITE_ONLY, bool, 1> acc_done(regions[1], DONE_ID);


	int num_pieces = indices.num_pieces;
	double u0 = 0;
	double u1 = 1;
	double ul, ur;
	// cout << u0 << "  " << u1 << endl;
	//cout << "current subdomain at init" << current_subdomain << endl;
	if (current_subdomain==0 || current_subdomain==num_pieces-1 ){
		if (current_subdomain== 0){
			ul = u0; //left BC
			ur = 0;
		}
		if (current_subdomain== num_pieces-1){
			ul = 0; // right BC
			ur = u1;
		}
	}
	else{
		ul = 0;
		ur = 0;

	}

	acc_edge_left[current_subdomain] = ul;
	acc_edge_right[current_subdomain] = ur;



	acc_inner_left[current_subdomain] = 0;
	acc_inner_right[current_subdomain] = 0;
	acc_done[current_subdomain] = false;


}

// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------

void init_spatial_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime){


	Indices indices = *((const Indices*)task->args);
	int current_subdomain = *((const int*)task->local_args);


	const FieldAccessor<WRITE_ONLY, double, 1> acc_field(regions[0], FIELD_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_source(regions[0], SOURCE_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_mesh(regions[0], MESH_ID);

	int num_elements_subdomain = indices.elements_per_subdomain;
	int No = indices.num_overlapping_elem;
	int Net = indices.num_total_elements;
	
	int start = current_subdomain * (num_elements_subdomain-No);
	double dx = 1 / float(Net);
	// int id; 
	for (int e=0; e<num_elements_subdomain+1; e++){
		int id = start + e;
		if (e<num_elements_subdomain+1) acc_field[id] = 1; //values in the middle of elements
		if (e<num_elements_subdomain+1) acc_source[id] = 1; //last value is just something...
		acc_mesh[id] = id*dx;

	}
}


// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------


void color_spatial_fields_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime){

	cout << "color_spatial_fields_task" << endl;

	const FieldAccessor<WRITE_ONLY, Point<1>, 1> acc_color(regions[0], SUBDOMAIN_COLOR_ID);

	LocalSpatial settings = *((const LocalSpatial*)task->args);
	int num_elements_subdomain = settings.num_elements_subdomain;
	int num_pieces = settings.num_pieces;


	for(int n=0; n<num_pieces; n++){
		for (int e=0; e<num_elements_subdomain; e++){
			int id = n * num_elements_subdomain + e;
			const Point<1> node_ptr(id);
			acc_color[node_ptr] = n;
		}
	}

}


// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------

bool loc_solver_task(const Task *task,
			         const std::vector<PhysicalRegion> &regions,
			         Context ctx, Runtime *runtime){

	//cout << "running local solver" << endl;

	/* create accessors to regions */
	const FieldAccessor<READ_ONLY, double, 1> acc_edge_left(regions[0], EDGE_LEFT_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_edge_right(regions[0], EDGE_RIGHT_ID);
	const FieldAccessor<READ_WRITE, double, 1> acc_inner_left(regions[1], INNER_LEFT_ID);
	const FieldAccessor<READ_WRITE, double, 1> acc_inner_right(regions[1], INNER_RIGHT_ID);
	const FieldAccessor<READ_WRITE, bool, 1> acc_done(regions[1], DONE_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_field(regions[2], FIELD_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_source(regions[2], SOURCE_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_mesh(regions[2], MESH_ID);

	// Get indices 
	int current_subdomain = *((const int*)task->local_args);
	Indices indices = *((const Indices*)task->args);

	int ned = indices.elements_per_subdomain;
	int no = indices.num_overlapping_elem;


	bool done = acc_done[current_subdomain];

	if(done==false){

		

		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------
		//                          GET SPATIAL OPERATORS
		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------

		/* Local Mesh, field and source */
		VectorXd X = VectorXd::Zero(ned+1);
		VectorXd field = VectorXd::Zero(ned);
		VectorXd Source = VectorXd::Zero(ned-1);

		int start = current_subdomain * (ned-no);
		for (int e=0; e<ned+1; e++){
			int id = start + e;
			X(e) = acc_mesh[id];
			if (e<ned) field(e) = acc_field[id]; //values in the middle of elements

		}



		// loop over elements subdomain
		for(int e=0; e<ned; e++){
			double f = acc_source[start+e]*.5*fabs(X(e+1)-X(e));	//Source term
			if(e==0){
				Source(e  ) += f;													
			}else if (e==ned-1){
				Source(e-1) += f;
			} else {
				Source(e-1) += f;							
				Source(e  ) += f;
			}
		} //next element


		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------
		//                      GET FE SYSTEM OPERATORS
		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------

		// Compute SpMat or get them from cache
		static map< int, SpMat > Oper_for_point;
		static map<int,SpMat* > Oper_ptr_for_point;
		static map<int,int> Oper_for_point_is_valid_for_timestep;
		static std::mutex cache_mutex;
		//int index_point = task->parent_task->index_point.point_data[0];
		SpMat Oper;
		SpMat* Oper_ptr;
		int valid_timestep;
		{
		std::lock_guard<std::mutex> guard(cache_mutex);
		Oper = Oper_for_point[current_subdomain];
		Oper_ptr = Oper_ptr_for_point[current_subdomain];
		valid_timestep = Oper_for_point_is_valid_for_timestep[current_subdomain];
		}
		if (Oper_ptr_for_point[current_subdomain] == NULL || valid_timestep != current_subdomain) { /* check if SpMat in cache 
																						   corresponds to current subdomain */

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
					
			Oper_ptr = &Oper;

			std::lock_guard<std::mutex> guard(cache_mutex);
			if (Oper_ptr_for_point[current_subdomain] != NULL) {
				Oper_for_point.erase(current_subdomain); // remove old value from cache
				Oper_ptr_for_point.erase(current_subdomain); // remove old value from cache
			}

			Oper_for_point[current_subdomain] = Oper;
			Oper_ptr_for_point[current_subdomain] = Oper_ptr;
			Oper_for_point_is_valid_for_timestep[current_subdomain] = current_subdomain;
		}


		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------
		//               SOLVE FE PROBLEM AND UPDATE INNER NODES 
		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------

		/* Dirichlet boundary values at edges for FE system. */
		double BCvalue_edge = field(0)/fabs(X(1)-X(0));
		double BCvalue_right = field(ned-1)/fabs(X(ned)-X(ned-1));

		/* get edges values from regions */
		double Ul = acc_edge_left[current_subdomain];
		double Ur = acc_edge_right[current_subdomain];

		

		/* solve FE problem*/
		SpMat Operator = Oper_for_point[current_subdomain];
		SimplicialCholesky<SpMat> chol; // cholesky decomp of (sparse) mass matrix
		chol.compute(Operator); /*  find cholesky decomposition. This will be used to solve the FE 
								linear system */

		// applies boundary values in the source vector
		VectorXd b = Source; b(0) += BCvalue_edge * Ul; b(ned-2) += BCvalue_right * Ur;

		// just making sure we are not deleting any rows in the FE system
		if(b.rows()!=ned-1) cout << "Mismatch\n";

		// solving M_d x_d = b_d
		VectorXd Usol = VectorXd(ned+1); // initiate vector with solution values
		Usol.segment(1,ned-1) = chol.solve(b); /* solves portion of Mx=b for inner points of current
												  subdomain. Boundary values will be added later.
												  This is the heaviest computation and should be done 
												  in parallel. */
		Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges



		/* getting local gaps */
		double gap_left = fabs(acc_inner_left[current_subdomain] - Usol(no));
		double gap_right = fabs(acc_inner_right[current_subdomain] - Usol(ned-no));


		acc_inner_left[current_subdomain] = Usol(no);
		acc_inner_right[current_subdomain] = Usol(ned-no);

		double tol = indices.tolerance;
		if ((gap_left < tol) && (gap_right < tol)== 1){
			acc_done[current_subdomain] = true;
			cout << "mesh of subdomain " << current_subdomain << endl;
			cout << X.transpose() << endl;
			cout << " piece of solution at subdomain " << current_subdomain << endl;
			cout << Usol.transpose() << endl;
		}

	} // END OF IF BOOL == FALSE

	return done;

}

// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------


void updateBC_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime){




	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_left(regions[0], EDGE_LEFT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_right(regions[0], EDGE_RIGHT_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_inner_left(regions[1], INNER_LEFT_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_inner_right(regions[1], INNER_RIGHT_ID);
	const FieldAccessor<READ_ONLY, bool, 1> acc_done(regions[1], DONE_ID);


	Indices indices = *((const Indices*)task->args);
	int current_subdomain = task->index_point.point_data[0]; 
	bool done = acc_done[current_subdomain];

	if (current_subdomain==0) cout << " new iteration\n \n";

	if (done==false){

		int num_pieces = indices.num_pieces;

		if (current_subdomain>0) acc_edge_left[current_subdomain] = acc_inner_right[current_subdomain-1];
		if (current_subdomain<num_pieces-1) acc_edge_right[current_subdomain] = acc_inner_left[current_subdomain+1];

	}

}










