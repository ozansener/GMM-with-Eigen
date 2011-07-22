#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "GMM.h"
using Eigen::MatrixXd;
using namespace std;
int main()
{
  GMM g(4,5);
  ifstream inFile;
  inFile.open("GMMData.txt",ifstream::in);
  MatrixXd m(1,4);
  float otVec[4];
  for(int i=0;i<300;i++){
      inFile>>otVec[0];
      inFile>>otVec[1];
      inFile>>otVec[2];
      inFile>>otVec[3];
      g.insertData(otVec,1);
      }

    g.iterateGMM(70);
    g.printModels();
}
