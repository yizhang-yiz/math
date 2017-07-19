#include <stan/math/rev/mat.hpp>
//#include <stan/math/prim/arr/fun/linear_interpolation.hpp>
#include <gtest/gtest.h>

TEST(linear_interpolation, constant) {
  int nx = 5, nout = 3;
  std::vector<double> x(nx), y(nx), xout(nout), yout;
  double youtTrue;

  for(int i = 0; i < nx; i++){
    x[i] = i;
    y[i] = i;
  }

  xout[0] = 1.5;
  xout[1] = 2.5;
  xout[2] = 4.5;

  yout = stan::math::linear_interpolation(xout, x, y);
  for(int i = 0; i < nout; i++){
      if(xout[i] <= x[0]){
	youtTrue = x[0];
      }else if(xout[i] >= x[nx - 1]){
	youtTrue = x[nx - 1];
      }else{
	youtTrue = xout[i];
      }
      EXPECT_FLOAT_EQ(youtTrue, yout[i]);
  }
}

TEST(linear_interpolation, gradient){
  using stan::math::var;
  int nx = 5, nout = 3;
  std::vector<double> x(nx), xout(nout), thisGrad(nx);
  Eigen::MatrixXd trueJac = Eigen::MatrixXd::Zero(3, 5); 

  for(int i = 0; i < nx; i++){
    x[i] = i;
  }

  xout[0] = 1.5;
  xout[1] = 2.5;
  xout[2] = 4.5;

  trueJac(0, 2) = (xout[0] - x[1]) / (x[2] - x[1]);
  trueJac(0, 1) = 1 - trueJac(0, 2);

  trueJac(1, 3) = (xout[1] - x[2]) / (x[3] - x[2]);
  trueJac(1, 2) = 1 - trueJac(1, 3);

  trueJac(2, 4) = 1;

  for(int i = 0; i < nout; i++){
    std::vector<var> y(nx), yout;
    for(int i = 0; i < nx; i++){
      y[i] = i;
    }
    yout = stan::math::linear_interpolation(xout, x, y);

    yout[i].grad(y, thisGrad);
    
    for(int j = 0; j < nx; j++){
      EXPECT_EQ(trueJac(i, j), thisGrad[j]);
    }
  }
}
  
