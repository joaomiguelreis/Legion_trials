typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

using namespace Eigen;
using namespace std;


class Subdomain{

	friend class PartitionDomain;

public:
	Subdomain(){};
	
	/* Set up of main members of a subdomain*/
	void Set(unsigned const dom_index, unsigned const ned_in, unsigned no_in, const VectorXd &X_Global);



	/* Global indices of left/right edges of subdomain*/
	unsigned Left() const 	{ return eglo_s; };
	unsigned Right() const 	{ return eglo_e; };	
	/* Value of the solution at the left/right edge of subdomain*/
	int LBC() const { return lbc; };
	int RBC() const { return rbc; };	

	/* get mesh of subdomain */
	VectorXd GetMesh() const { return X; };
	/*get solution of local problem*/
	VectorXd GetSol() const { return Usol; };
	/*get number of elements in the subdomain*/
	unsigned GetNel() const { return ned; };
	/*get index of subdoamin*/
	unsigned GetIndex() const { return index; };

	/* see comments in subdomain.cpp*/
	void BuildOperator(const VectorXd &K_Global);
	void SetSource(const VectorXd &S_Global);
	void SetSolution(const double Ul, const double Ur);
	void SetSolution(const VectorXd &K_Global, const VectorXd &S_Global, 
										const double Ul, const double Ur){ 
		BuildOperator(K_Global);
		SetSource(S_Global);
		SetSolution(Ul,Ur);
	};


private:
	unsigned index, ned, no;
	vector<int> IndNeigh;
	unsigned eglo_s, eglo_e;  	// edge global elements
	int lbc, rbc;			// index of reduced BCs 
	VectorXd X;
	VectorXd field;
	VectorXd Source;
	VectorXd Usol;
	SpMat OPer; // sparse version of mass matrix
	SimplicialCholesky<SpMat> chol; // cholesky decomp of (sparse) mass matrix
	VectorXd UBC; // vector with values of solution at edges
};

class PartitionDomain {

public:
	PartitionDomain(unsigned nd, unsigned no, const VectorXd &XM, double const nkl, double const mu){
		// nd: number of subdomains
		// no: number of overlapping elements
		// XM: spatial mesh 

		Nd = nd;
		No = no;
		Net= XM.rows()-1;
		Nbc= 2*(Nd-1);
		XGlobal = XM;
		Ne = (Net-No)/Nd;
		if(Net!= Ne*Nd + No){
			cout << "Inconsistant parameters to build the partition\n";
			return;
		}
		n_kl = nkl;
		mu_kl = mu;
	};

	// Utilities to work with boundary values
	VectorXi GetIndBC() const {return IndBC;};
	VectorXi GetIGL() const {return IGL;};
	VectorXi GetIGR() const {return IGR;};
	VectorXd GetBCVal(VectorXd const &U) const {
		VectorXd Val(Nbc);
		for(unsigned ib=0; ib<Nbc; ib++) Val(ib) = U(IndBC(ib));
		return Val;
	};
	void SetBCVal(VectorXd &U,VectorXd const &Ubc) const {
		for(unsigned ib=0; ib<Nbc; ib++) U(IndBC(ib)) = Ubc(ib);
		return;
	};
	VectorXd GetGap(VectorXd const &U1, VectorXd const &U2) const {
		VectorXd Gap = GetBCVal(U2) - GetBCVal(U1);
		return Gap;
	};
	

	// FEM linear system
	void Set(vector<Subdomain>* SD);
	void SetOperators(VectorXd const &KG){
		KGlobal = KG;
		for(unsigned id=0; id<Nd; id++) (*SDom)[id].BuildOperator(KGlobal);
	};
	void SetSources(VectorXd const &SG){
		FGlobal = SG;
		for(unsigned id=0; id<Nd; id++) (*SDom)[id].SetSource(FGlobal);
	};

	// SM
	VectorXd DoOneSchwartz(VectorXd &U);
	unsigned SolveSchwartz(VectorXd &U, const VectorXd &KG, const VectorXd &SG, 
		double prec = 1.e-8);
	unsigned SolveSchwartz(VectorXd &U, double prec = 1.e-8);

	// // PSM
	// void ApplyCorrector(VectorXd &Usol, VectorXd &gap);
	// unsigned SolveAccSchwartz(VectorXd &U, const VectorXd &KG, const VectorXd &SG, unsigned every,
	// 						double prec = 1.e-8);
	// unsigned SolveAccSchwartz(VectorXd &U, unsigned every, double prec);

	// // Preconditioner
	// void SetAcc(MatrixXd const &A){ Acc=A; LUD.compute(A); };
	// MatrixXd GetAcc() const { return Acc;};
	// void SetAccelerator(VectorXd const &KG);
	// void SetPC(int const &n_kl, int const &level);
	// MatrixXd GetQuadPts(){return Pts;};
	// MatrixXd GetPSP(){return PSP;};
	// void SetSorrugate(vector<MatrixXd> &VAcc, MatrixXd const &Modes);
	// MStoch<PCB,MatrixXd> GetSorrugate(){return AccS;};


private:
	unsigned Nd;			//Number of subdomains
	unsigned No;			//Number of overlapping points
	unsigned Ne;			//Number of elements per subdomains
	unsigned Net;			//Total number of elements
	unsigned Nbc;			//Number of inner boundary points

	double n_kl;
	double mu_kl;

	VectorXi IGL, IGR;
	VectorXi IndBC;
	VectorXd XGlobal;
	VectorXd FGlobal;
	MatrixXd Modes;
	VectorXd KGlobal;
	vector<Subdomain>* SDom; // vector with subdomains
	MatrixXd Acc;
	FullPivLU<MatrixXd> LUD;
	PCB Base;
	MatrixXd PSP;
	MatrixXd Pts;
	// MStoch<PCB,MatrixXd> AccS;
};

