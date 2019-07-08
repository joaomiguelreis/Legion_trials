
#include <iomanip>
#include <fstream>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;


MatrixXd SetKLModes(int const nel, VectorXd &Lam, double len_kl, double var_f){
	MatrixXd K(nel,nel);
	MatrixXd M = MatrixXd::Zero(nel,nel);
	double xi,xj,xij,h,cxx;
    double x,y;
	h = 1./(double)(nel);									// Size of element
	for(unsigned i=0; i<nel; i++){
		xi = h*(i+.5);
		for(unsigned j=i; j<nel; j++){
            xj = h*(j+.5);
            xij = fabs(xj - xi)/ len_kl;
			cxx = var_f*exp( - xij*xij / 2. )*h*h;   // Squared exponential
			// cxx = var_f*exp( - xij )*h*h;   // exponential

			K(i,j) = cxx;
			if(i!=j) K(j,i) = K(i,j); 
		}
		M(i,i) = h;
	}
	
	GeneralizedSelfAdjointEigenSolver<MatrixXd> eig(K,M); /* K g = \lambda M g */
	MatrixXd Ev = eig.eigenvectors();
	VectorXd Eg = eig.eigenvalues();
	
	MatrixXd Modes(nel,nel);
	Lam = VectorXd(nel);

    // Finding the KL modes from eigen values and eigen functions,
	for(int i=0; i<nel; i++){
		int ie=i;
		for(int j=i+1; j<nel; j++){
			if(Eg(j)>Eg(ie)) ie=j;
		}
		if(ie!=i){
			double e = Eg(i); Eg(i)=Eg(ie); Eg(ie)=e;
			VectorXd E= Ev.col(i); Ev.col(i) = Ev.col(ie); Ev.col(ie) = E;
		}
		if(Eg(i)<0 ) Eg(i) =0;
		Lam(i) = sqrt(abs(Eg(i)));
		Modes.col(i) = Ev.col(i);
	}

	cout << "First 10 modes (square root of eigenvalues): \n";
	cout << Lam.head(10).transpose() << endl;
	cout << "\n";
	return Modes;
};