#include "schwarz_solver.h"

void init_spatial_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime){

	Indices indices = *((const Indices*)task->args);
	int current_subdomain = task->index_point.point_data[0];

	const FieldAccessor<WRITE_ONLY, double, 1> acc_source(regions[0], SOURCE_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_field(regions[0], FIELD_ID);

	const FieldAccessor<WRITE_ONLY, double, 1> acc_mesh(regions[1], MESH_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_sol(regions[1], SOLUTION_ID);

	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_left(regions[2], EDGE_LEFT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_edge_right(regions[2], EDGE_RIGHT_ID);

	const FieldAccessor<WRITE_ONLY, double, 1> acc_inner_left(regions[3], INNER_LEFT_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_inner_right(regions[3], INNER_RIGHT_ID);
	const FieldAccessor<WRITE_ONLY, bool, 1> acc_done(regions[3], DONE_ID);
	const FieldAccessor<WRITE_ONLY, bool, 1> acc_write(regions[3], WRITE_ID);

	const FieldAccessor<WRITE_ONLY, double, 2> acc_matrix(regions[4], MASS_MATRIX_ID);
	

	int Net = indices.num_total_elements;
	int No = indices.num_overlapping_elem;
	int num_pieces = indices.num_pieces;
	double u0 = indices.u0;
	double u1 = indices.u1;
	int ned = indices.elements_per_subdomain;


	double ul, ur;
	int start = -current_subdomain * (No+1);

	// write local matrices R4
	MatrixXd A = MatrixXd::Zero(ned-1, ned-1);

	/* Local Mesh, field and source */
	VectorXd X = VectorXd::Zero(ned+1);
	VectorXd field = VectorXd::Zero(ned);

	// write source R0
	Rect<1> middle_elem_rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
	int e=0;
	for (PointInRectIterator<1> pir(middle_elem_rect); pir(); pir++){
	    acc_source[*pir] = 1;
	    field(e) = acc_field[*pir];
		e++;
	}

	// write mesh and solution R5
	Rect<1> edge_elem_rect = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
	int i = 0;
	for (PointInRectIterator<1> pir(edge_elem_rect); pir(); pir++){
		X(i) = (double) ( start + (*pir).x )/(Net);
		acc_mesh[*pir] = X(i);
		i++;
	    acc_sol[*pir] = 0;
	    
	}


	// write inner and edge nodes R2 and R3
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
	acc_write[current_subdomain] = true;

	//Loop over elements
	for(int e=0; e<ned; e++){				
		double kv = field(e)/fabs(X(e+1)-X(e)); // k(x_e)/dx
		if(e==0){	// first sudomain	
			A(e,e) += kv;

		}else if (e==ned-1){ // last subdomain
			A(e-1,e-1) += kv;

		} else { //inner subdomains
			A(e-1,e-1) += kv;
			A(e-1,e) -= kv;
			A(e,e-1) -= kv;
			A(e,e) += kv;
		}

	}

	MatrixXd L = A.llt().matrixL();

	for (int e1=0; e1<ned-1; e1++){
		int i1 = current_subdomain * (ned-1) + e1;
		for(int e2=0; e2<ned-1; e2++){
			acc_matrix[Point<2>(i1, e2)] = L(e1,e2);
		}
	}

}


// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------




void loc_solver_task(const Task *task,
			         const std::vector<PhysicalRegion> &regions,
			         Context ctx, Runtime *runtime){


	/* create accessors to regions */
	const FieldAccessor<READ_ONLY, double, 1> acc_edge_left(regions[0], EDGE_LEFT_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_edge_right(regions[0], EDGE_RIGHT_ID);
	const FieldAccessor<READ_WRITE, double, 1> acc_inner_left(regions[1], INNER_LEFT_ID);
	const FieldAccessor<READ_WRITE, double, 1> acc_inner_right(regions[1], INNER_RIGHT_ID);
	const FieldAccessor<READ_WRITE, bool, 1> acc_done(regions[1], DONE_ID);
	const FieldAccessor<READ_WRITE, bool, 1> acc_write(regions[1], WRITE_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_field(regions[2], FIELD_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_source(regions[2], SOURCE_ID);
	const FieldAccessor<READ_ONLY, double, 1> acc_mesh(regions[3], MESH_ID);
	const FieldAccessor<WRITE_ONLY, double, 1> acc_sol(regions[3], SOLUTION_ID);
	const FieldAccessor<READ_ONLY, double, 2> acc_matrix(regions[4], MASS_MATRIX_ID);



	// Get indices 
	int current_subdomain = task->index_point.point_data[0];

	Indices indices = *((const Indices*)task->args);

	int ned = indices.elements_per_subdomain;
	int no = indices.num_overlapping_elem;


	bool done = acc_done[current_subdomain];

	if(done==false){

		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------
		//               SOLVE FE PROBLEM AND UPDATE INNER NODES 
		// ----------------------------------------------------------------------
		// ----------------------------------------------------------------------


		// We use the same region to store the mesh (defined at each edge of element),
		// and the field and source (defined both at the middle of the element)
		VectorXd X = VectorXd::Zero(ned+1);
		VectorXd field = VectorXd::Zero(ned);
		VectorXd Source = VectorXd::Zero(ned-1);

		Rect<1> rect2 = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
		int e=0;
		for (PointInRectIterator<1> pir(rect2); pir(); pir++){
			field(e) = acc_field[*pir];
			e++;
		}

		Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
		int i=0;
		for (PointInRectIterator<1> pir(rect); pir(); pir++){
			X(i) = acc_mesh[*pir];
			i++;
		}
		

		// Building source
		int start = current_subdomain * (ned-no);
		for(int e=0; e<ned; e++){ // loop over elements subdomain
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



		

		/* Dirichlet boundary values at edges for FE system. */
		double BCvalue_edge = field(0)/fabs(X(1)-X(0));
		double BCvalue_right = field(ned-1)/fabs(X(ned)-X(ned-1));

		/* get edges values from regions */
		double Ul = acc_edge_left[current_subdomain];
		double Ur = acc_edge_right[current_subdomain];


		// applies boundary values in the source vector
		VectorXd b = Source; b(0) += BCvalue_edge * Ul; b(ned-2) += BCvalue_right * Ur;

		// just making sure we are not deleting any rows in the FE system
		if(b.rows()!=ned-1) cout << "Mismatch\n";

		MatrixXd L = MatrixXd::Zero(ned-1, ned-1);
		VectorXd y = VectorXd::Zero(ned-1);
		VectorXd x = VectorXd::Zero(ned-1);


		for (int e1=0; e1<ned-1; e1++){
			int i1 = current_subdomain * (ned-1) + e1;
			for(int e2=0; e2<ned-1; e2++){
				L(e1, e2) = acc_matrix[Point<2>(i1, e2)];
			}
		}
		MatrixXd Lt = L.transpose();

		/* solves portion of Mx=b for inner points of current
		subdomain. Boundary values will be added later.
		This is the heaviest computation and should be done 
		in parallel. */

		// forward substitution		
		for(int m=0; m<ned-1; m++){
			double sum = 0;
			for(int i=0; i<=m-1; i++){
				sum -= L(m,i) * y(i);
			}
			y(m) = (b(m) + sum) / L(m,m);
		}

		// backward substitution
		for(int m=ned-1 -1; m>=0; m--){
			double sum = 0;
			for(int i=ned-1 -1; i>m; i--){
				sum -= Lt(m,i) * x(i);
			}
			x(m) = (y(m) + sum) / Lt(m,m);
		}

		
		// solving M_d x_d = b_d
		VectorXd Usol = VectorXd(ned+1); // initiate vector with solution values
		Usol.segment(1,ned-1)= x;

		Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges


		/* getting local gaps */
		double gap_left = fabs(acc_inner_left[current_subdomain] - Usol(no));
		double gap_right = fabs(acc_inner_right[current_subdomain] - Usol(ned-no));


		acc_inner_left[current_subdomain] = Usol(no);
		acc_inner_right[current_subdomain] = Usol(ned-no);

		double tol = indices.tolerance;
		if ((gap_left < tol) && (gap_right < tol)== 1){
			acc_done[current_subdomain] = true;



			//display
			if (acc_write[current_subdomain]){
				int i=0;
				Rect<1> edge_elem_rect = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
				for (PointInRectIterator<1> pir(edge_elem_rect); pir(); pir++){
					acc_sol[*pir] = Usol(i);
					i++;
				}
				acc_write[current_subdomain] = false;			
			}

		}

	} // END OF IF BOOL == FALSE

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

	if (done==false){

		int num_pieces = indices.num_pieces;

		if (current_subdomain>0) acc_edge_left[current_subdomain] = acc_inner_right[current_subdomain-1];
		if (current_subdomain<num_pieces-1) acc_edge_right[current_subdomain] = acc_inner_left[current_subdomain+1];

	}

}

// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------




void display_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime){

	const FieldAccessor<READ_WRITE, double, 1> acc_sol(regions[0], SOLUTION_ID);

	Indices indices = *((const Indices*)task->args);
	int Net = indices.num_total_elements;
	int ned = indices.elements_per_subdomain;
	int no = indices.num_overlapping_elem;
	int num_pieces = indices.num_pieces;



	VectorXd solution = VectorXd::Zero(Net+1);
	int id = 0;
	for(int n=0; n<num_pieces; n++){
		int start = -n * (no+1);
		if (n==0){
			for(int e=0; e<ned+1; e++){
				id = n * (ned+1) + e;
				solution[start+e] = acc_sol[e];
			}
		} else{
			for(int e=0; e<ned+1; e++){
				id = n * (ned+1) + e;
				solution[start+id] = acc_sol[id];

			}
		}
		
	}

	cout << "final solution \n";
	cout << solution.transpose() << endl;


}







