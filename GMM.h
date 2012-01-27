#ifndef _GMM_H
#define _GMM_H
#include "Eigen/Dense"
#include <vector>
using Eigen::MatrixXd;
class GMM{
private:
    bool* mpbIsModelValid;
    bool mbIsFirstRun;
    std::vector<float> mpfInputData;
    std::vector<float> mpfWeights;
    int mnNumOfInputs;
    int mnDimension;
    int mnK;
    float* mpfDeterminants;
    float* mpfMeanVector;
    MatrixXd* mpfCovVectors;
    MatrixXd* mpfInvCovVectors;
    float getLikelihood(int inputId,int modelId);
    float* normSumK;
public:
    GMM();
    GMM(int dimension,int k);
    bool initializeAndClear(int dimension,int k);
    void insertData(float* inVector,int inSize);
    void clear();
    void iterateGMM(int numIter);
    float getLikelihood(float* inputVec);
    float getLikelihood2(float* inputVec);
    void printModels();
    void initLastMean();
};
#endif
