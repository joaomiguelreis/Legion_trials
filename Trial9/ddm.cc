#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include "../../../StochTk++/includes/pcb++.h"
#include "../../../StochTk++/includes/cub++.h"
#include "../../../StochTk++/includes/mstoch++.h"
#include "ddm.h"
using namespace std;
using namespace Eigen;



// ##########################################################################
// ##########################################################################


/* This routine set up the subdomain indices and mesh */
void Subdomain::Set(unsigned const dom_index, unsigned const ned_in, unsigned no_in, const VectorXd &X_Global){
		index = dom_index;					//Index of the subdomain
		ned = ned_in;						//Number of elements in the subdomains
		no	= no_in;						//Number of overlapping elements (on each size).
		eglo_s = index * (ned-no);
		eglo_e = eglo_s + ned;
		lbc = -1; // trick for SetAccelerator
		rbc = -1;
		X = X_Global.segment(eglo_s,ned+1);
	};


// ##########################################################################
// ##########################################################################

/* This routine computes the sparse cholesky decomposition of the portion of the mass matrix M, 
from the FE system Mx=b, relative to a specific subdomain. It based on the triplet T that stores the 
values that approximate the entries of M (FE integrals). 
*/
void Subdomain::BuildOperator(const VectorXd &K_Global){
	// K_Global:coefficient field


	field = K_Global.segment(eglo_s,ned); // segment field restricted to subdomain
	vector<T> cprec; //inititate the vector with values of FEM integral
	UBC = VectorXd::Zero(2); /* vector with boundary values at edges. */

	//Loop over elements
	for(int e=0; e<ned; e++){				
		double kv = field(e)/fabs(X(e+1)-X(e)); // k(x_e)/dx	

		if(e==0){	// first sudomain	
			cprec.push_back(T(e,e,kv)); 
			UBC(0) = kv;

		}else if (e==ned-1){ // last subdomain
			cprec.push_back(T(e-1,e-1,kv));
			UBC(1) = kv;

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

	OPer = SpMat(ned-1,ned-1);	// Assembly sparse matrix:
	OPer.setFromTriplets(cprec.begin(), cprec.end()); /* fill the sparse matrix with cprec 
														 Oper_(e1,e2) = kappa_value */
	chol.compute(OPer); /*  find cholesky decomposition. This will be used to solve the FE 
							linear system */
};


// ##########################################################################
// ##########################################################################

 
/* This routines computes the source vector b from the FE linear system Mx=b */
void Subdomain::SetSource(const VectorXd &S_Global){
	// S_Global: source function defined global domain

	Source = VectorXd::Zero(ned-1);

	// loop over elements subdomain
	for(int e=0; e<ned; e++){
		double f = S_Global(eglo_s+e)*.5*fabs(X(e+1)-X(e));	//Source term
		if(e==0){
			Source(e  ) += f;													
		}else if (e==ned-1){
			Source(e-1) += f;
		} else {
			Source(e-1) += f;							
			Source(e  ) += f;
		}
	} //next element
};


// ##########################################################################
// ##########################################################################



/* This routine solves the subproblems. Usol is a member of class subdomain. This routine
updates Usol from a new pair of boundary values. Usol is defined for all elements
of the current subdomain. Does u^k -> u^k+1*/
void Subdomain::SetSolution(const double Ul, const double Ur){
		// Ul, Ur: left and right boundary values of subdomain

		// applies boundary values in the source vector
		VectorXd b = Source; b(0) += UBC(0)*Ul; b(ned-2) += UBC(1)*Ur;

		// just making sure we are not deleting any rows in the FE system
		if(b.rows()!=ned-1) cout << "Mismatch\n";

		// solving M_d x_d = b_d
		Usol = VectorXd(ned+1); // initiate vector with solution values
		Usol.segment(1,ned-1) = chol.solve(b); /* solves portion of Mx=b for inner points of current
												  subdomain. Boundary values will be added later.
												  This is the heaviest computation and should be done 
												  in parallel. */
		Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges
	};

VectorXd Subdomain::SetSolution_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime){


	// Ul, Ur: left and right boundary values of subdomain

	printf("Executing SetSolution_task \n");

	VectorXd localBCVal = *((const VectorXd*)task->local_args);

	double Ul = localBCVal[0];
	double Ur = localBCVal[1];

	// applies boundary values in the source vector
	VectorXd b = Source; b(0) += UBC(0)*Ul; b(ned-2) += UBC(1)*Ur;

	// just making sure we are not deleting any rows in the FE system
	if(b.rows()!=ned-1) cout << "Mismatch\n";

	// solving M_d x_d = b_d
	Usol = VectorXd(ned+1); // initiate vector with solution values
	Usol.segment(1,ned-1) = chol.solve(b); /* solves portion of Mx=b for inner points of current
											  subdomain. Boundary values will be added later.
											  This is the heaviest computation and should be done 
											  in parallel. */
	Usol(0) = Ul; Usol(ned) = Ur; // adding boundary values at the edges

	return Usol;
};

// /*static*/
// void Subdomain::register_loc_solver_task(void)
// {
//   TaskVariantRegistrar registrar(LOC_SOLVER_TASK_ID, "SetSolution_task");
//   registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
//   registrar.set_leaf(true);
//   runtime->register_task_variant<VectorXd, Subdomain::SetSolution_task>(registrar, "SetSolution_task");
// }

// ##########################################################################
// ##########################################################################
// 
//          CLASS PARTITION
// 
// ##########################################################################
// ##########################################################################


void PartitionDomain::Set(vector<Subdomain>* SD) { 


	SDom = SD;
	for(unsigned id=0; id<Nd; id++) (*SDom)[id].Set(id,Ne+No,No,XGlobal);
	IndBC = VectorXi(Nbc);
	IGL = VectorXi(Nd);
	IGR = VectorXi(Nd);		
	int count=0;
	for(unsigned id=0; id<Nd; id++){
		IGL(id) = (*SDom)[id].Left();
		IGR(id) = (*SDom)[id].Right();
		if(id>0){ 
			IndBC(count) = IGL(id);
			(*SDom)[id].IndNeigh.push_back((*SDom)[id].index - 1);
			(*SDom)[id].lbc = count;
			count++;
		}			
		if(id<Nd-1){ 
			IndBC(count) = IGR(id);
			(*SDom)[id].IndNeigh.push_back((*SDom)[id].index + 1); 
			(*SDom)[id].rbc = count;
			count++;
		} 
	}
	cout << "BC indexes : " << IndBC.transpose() << endl;
	cout << "\n";
};



// ##########################################################################
// ##########################################################################


/* THE CENTRAL ROUTINE. This routine computes the gap vector. The gap corresponds to the distance between
the cboundary values of the current iterate and the next interate vector. 
The next iterate vector is computed inside this routine by a sub-routine */
VectorXd PartitionDomain::DoOneSchwartz(VectorXd &U){
	// U: The current interate U^k

	// Before changinf U^k, store it!
	VectorXd Uold = U;

	// loop over subdomains. MUST BE DONE IN PARALLEL
	for(unsigned id=0; id<Nd; id++){
		double Ul, Ur; Ul = Uold(IGL(id)); Ur = Uold(IGR(id)); /* Fixes boudary values of
				 												of current subdomain */

		(*SDom)[id].SetSolution(Ul,Ur); /* adds subsolution to subsolution vector. 
										Heaviest computation and it is why the loop 
										must be done in paralllel */

		unsigned shift = IGL(id); // shift is the indexof the first element of the subdomain id
		VectorXd uloc = (*SDom)[id].GetSol(); // retrieve solution
		for(unsigned i=1; i<uloc.rows()-1; i++) U(shift+i) = uloc(i); // update U^k -> U^k+1
	}
	return GetGap(Uold,U);
};

VectorXd PartitionDomain::DoOneSchwartz_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime){
	// U: The current interate U^k

	printf("Executing DoOneSchwartz_task\n");

	VectorXd U = *((const VectorXd*)task->args);

	// Before changinf U^k, store it!
	VectorXd Uold = U;
	cout << IGL(0) << endl;
	cout << " here " << endl;

	// loop over subdomains. MUST BE DONE IN PARALLEL
	Rect<1> index_launch(0,Nd-1);
	ArgumentMap local_args;
	for (int index=0; index<Nd; index++){

		Point<1> point(index);
		VectorXd GBC(2);
		
		GBC(0) = U(IGL(index));
		GBC(1) = U(IGR(index));
		cout << GBC.transpose() << endl;

		local_args.set_point(point, TaskArgument(&GBC, sizeof(VectorXd)));

	}



	IndexTaskLauncher loc_solver_launcher(LOC_SOLVER_TASK_ID, index_launch, TaskArgument(NULL, 0), local_args);
	loc_solver_launcher.tag |= Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
	FutureMap futureUloc = runtime->execute_index_space(ctx, loc_solver_launcher);
	futureUloc.wait_all_results();

	int id=0;
	for (PointInDomainIterator<1> itr(index_launch); itr(); itr++){

		unsigned shift = IGL(id); // shift is the indexof the first element of the subdomain id
		id++;
		VectorXd uloc = futureUloc.get_result<VectorXd>(*itr);
		for(unsigned i=1; i<uloc.rows()-1; i++) U(shift+i) = uloc(i); // update U^k -> U^k+1
	}

	return GetGap(Uold,U);
};


// ##########################################################################
// ##########################################################################


/* This routine is an extension of the next routine. It sets up thhe FE system
in addition to what the next routine does */
unsigned PartitionDomain::SolveSchwartz(VectorXd &U, const VectorXd &KG, const VectorXd &SG, 
	double prec){
	SetOperators(KG);
	SetSources(SG);
	return SolveSchwartz(U, prec);
};


// ##########################################################################
// ##########################################################################


/* This routine preforms Schwarz iterations over the global domain until a given
precision is acheived. This returns the  number of SM iterations preformed */
unsigned PartitionDomain::SolveSchwartz(VectorXd &U, double prec){
	// U: initial solution to start iterations
	// prec: precision to be reached

	double diff=1.;
	unsigned iter = 0; 
	// diff = DoOneSchwartz(U).norm();
	// iter++;
	while(diff>prec){
		diff = DoOneSchwartz(U).norm();
		iter++;
	}
	// printf("Problem solved in %6u Schwartz iterations\n",iter);// uncomment to give number
																  // SM iterations per sample
	return iter;
};

/* This routine preforms Schwarz iterations over the global domain until a given
precision is acheived. This returns the  number of SM iterations preformed */
void PartitionDomain::SolveSchwartz_task(const Task *task,
			       						const std::vector<PhysicalRegion> &regions,
			       						Context ctx, Runtime *runtime){
	// U: initial solution to start iterations
	// prec: precision to be reached

	printf("Executing SolveSchwartz_task\n");

	VectorXd U = *((const VectorXd*)task->args);
	// double prec = Getprec();
	cout << "-  prec: " << prec << endl;
	cout << "Nd " << Nd << endl;

	double diff=1.;
	unsigned iter = 0; 
	// diff = DoOneSchwartz(U).norm();
	// iter++;
	while(diff>prec){
		TaskLauncher schwarz_launcher(DO_ONE_SCHWARZ_TASK_ID, TaskArgument(&U, sizeof(VectorXd)));
		schwarz_launcher.tag |= Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
		Future FutureGap = runtime->execute_task(ctx, schwarz_launcher);
		VectorXd Gap = FutureGap.get_result<VectorXd>();
		diff = Gap.norm();
		iter++;
	}
	// printf("Problem solved in %6u Schwartz iterations\n",iter);// uncomment to give number
																  // SM iterations per sample
	// return iter;
};



// /*static*/
// void PartitionDomain::register_schwarz_solver_task(void)
// {
// 	TaskVariantRegistrar registrar(SCHWARZ_SOLVER_TASK_ID, "SetSolution_task",true);
// 	registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
// 	// registrar.set_leaf(true);
// 	Runtime::register_task_variant<SolveSchwartz_task>(registrar);
// }


// /*static*/
// void PartitionDomain::register_one_schwarz_task(void)
// {
//   TaskVariantRegistrar registrar(DO_ONE_SCHWARZ_TASK_ID, "DoOneSchwartz_task");
//   registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
//   // registrar.set_leaf(PartitionDomain::LEAF);
//   Runtime::register_task_variant<VectorXd, DoOneSchwartz_task>(registrar, "DoOneSchwartz_task");
// }


// // ##########################################################################
// // ##########################################################################


// unsigned PartitionDomain::SolveAccSchwartz(VectorXd &U, const VectorXd &KG, const VectorXd &SG, 
// 	unsigned every, double prec){
// 	SetOperators(KG);
// 	SetSources(SG);
// 	return SolveAccSchwartz(U, every, prec);
// };


// /* This routine solves the acceleration system Pu=g. It is the analogous to the routine 
// DoOneSchwartz.  It requires a LUD decomposition of the preconditioner P. This is another 
// heavy routine. Pu=g is a system for the boundary points, whereas M_d x_d = b_d is defined 
// for all elements of each subdomain. */
// void PartitionDomain::ApplyCorrector(VectorXd &Usol, VectorXd &gap){

// 	// Before changinf U^k, store it!
// 	VectorXd Usave = Usol;
// 	VectorXd up = LUD.solve(gap); // solve Pu=g with LUD decomposition of P
// 	for(unsigned ib=0; ib<Nbc; ib++) Usol(IndBC(ib)) += up(ib); // updating boundary values
// 	gap = GetGap(Usave, Usol);
// };

// /* This routine runs the PSM method and returns the number of iterations preformed. 
// It is analogous to the SolveSchwartz. */
// unsigned PartitionDomain::SolveAccSchwartz(VectorXd &U, unsigned every, double prec){
// 	// U: Initial solution
// 	// every: number of SM iterations between each acceleration. Most efficient with 0
// 	// prec: tolerance to be acheived

// 	double diff=1.;
// 	unsigned iter = 0;
// 	iter+=1;
// 	VectorXd gap;
// 	while(diff>prec){
// 		if(iter>1 && iter%(every+1)==0){
// 			ApplyCorrector(U, gap);
// 			diff = gap.norm();
// 		}else{
// 			gap = DoOneSchwartz(U);
// 			diff = gap.norm();
// 		}
// 		iter++;
// 		if (iter>1000) break;
// 	}

// 	// printf("Problem solved in %6u (Accelerated) Schwartz iterations - Error %12.6e \n",iter,diff);
// 	return iter;
// };



// /* This routine assembles the preconditioner P based on a finite representation of kappa */
// void PartitionDomain::SetAccelerator(VectorXd const &KG){
// 	// KG: finite representation of kappa defined in the all domain

// 	/* initialising operators*/
// 	Acc = MatrixXd::Zero(Nbc,Nbc);
// 	VectorXd S0 = VectorXd::Zero(Net);
// 	SetOperators(KG);  
// 	SetSources(S0);

// 	/* Loop over subdomains. This computes the elementary solutions per subdomain and
// 	stores as entries of P. Here P is written as Acc and it is a member of the class
// 	Partition */
// 	for(unsigned id=0; id<Nd; id++){
// 		int lbc = (*SDom)[id].LBC();

// 		 left elementary solution 
// 		if(lbc>=0){
// 			// set BCs
// 			VectorXd U0 = VectorXd::Zero(Net+1);		
// 			VectorXd U1 = U0;
// 			U0(IGL(id)) = 1;

// 			// solve elementary problem (just inner points)
// 			(*SDom)[id].SetSolution(1,0);

// 			// update U1
// 			unsigned shift = IGL(id);
// 			VectorXd uloc = (*SDom)[id].GetSol();
// 			for(unsigned i=1; i<uloc.rows()-1; i++) U1(shift+i) = uloc(i);

// 			/* Storing values of elementary solutions at boundary edges of current 
// 			subdomain. Matrix Acc is not sparse. Since U0 is zero everywhere GetGap is a
// 			nice way to get values at the boudaries in a single line */
// 			Acc.col(lbc) = GetGap(U1,U0);
// 		}

// 		int rbc = (*SDom)[id].RBC(); // change to right edge

// 		/* right elementary solution */
// 		if(rbc>=0){
// 			VectorXd U0 = VectorXd::Zero(Net+1);		
// 			VectorXd U1 = U0;
// 			U0(IGR(id)) = 1;
// 			(*SDom)[id].SetSolution(0,1);
// 			unsigned shift = IGL(id);
// 			VectorXd uloc = (*SDom)[id].GetSol();
// 			for(unsigned i=1; i<uloc.rows()-1; i++) U1(shift+i) = uloc(i);
// 			Acc.col(rbc) = GetGap(U1,U0);
// 		}
// 	}

// 	// find LU decomposition of P
// 	LUD.compute(Acc);
// 	if(LUD.isInvertible()==0) cout << "Acc not invertible\n"; /* if this message shows up it is 
// 																that some operator (kappa, source, ...)
// 																isn't appropriately set up. */
// };


// void PartitionDomain::SetPC(int const &n_kl, int const &level){
// 	CUB<double> Cub(n_kl,0);
// 	Cub.Initialize(level);
// 	Cub.Finalize();
// 	Cub.Set_Poly();
// 	Base = PCB(Cub.poly,'U');
// 	PSP = Cub.Get_Pseudo_NISP();
// 	Pts = Cub.GetCubPts();
// 	printf("Basis dimension %d Ndim %d Nord %d \n",Base.Npol(),Base.Ndim(), Base.Nord());
// 	printf("PSP will use %d nodes \n",(int) PSP.cols());
// 	cout << "\n";

// };


// void PartitionDomain::SetSorrugate(vector<MatrixXd> &VAcc, MatrixXd const &Modes){

// 		for(unsigned in=0; in<PSP.cols(); in++){
// 			VectorXd Xi = Pts.col(in);
// 			VectorXd KF = VectorXd::Ones(Net)*mu_kl;

// 			for(unsigned k=0; k<n_kl; k++) KF += Modes.col(k)*inv_cdf_normal(Xi(k));

// 			KF.array() = KF.array().exp();
// 			SetAccelerator(KF); 					//Use the truncated KA to build the accelerator
// 			VAcc[in] = Acc; 
// 		}
// 		MStoch<PCB,MatrixXd> Surr(&Base);
// 		for(unsigned ip=0; ip<Base.Npol(); ip++){
// 			MatrixXd Accm = VAcc[0]*0;
// 			for(unsigned in=0; in<Pts.cols(); in++){
// 				if(PSP(ip,in)!=0) Accm += VAcc[in]*PSP(ip,in);
// 			}
// 			Surr.set_mode(ip,Accm);
// 		}
// 		AccS = Surr;


// };

