#include "GMM.h"
#include "Eigen/Dense"
#include <iostream>
#include <vector>
#define PI 3.14159265
using Eigen::MatrixXd;
using namespace std;

GMM::GMM(){
    mpbIsModelValid=0;
    mbIsFirstRun = true;
    mnDimension=0;
    mnK=0;
    mpfDeterminants=0;
    mpfMeanVector=0;
    mpfCovVectors=0;
    mpfInvCovVectors=0;
    mnNumOfInputs=0;
    mpfInputData.clear();
    }
GMM::GMM(int dimension,int k){
    mbIsFirstRun = true;
    if((k<1)||(dimension<1)){
        cerr<<"Zero dimension or zero number of classes"<<endl;
        exit(0);
    }
    mnDimension=dimension;
    mnK=k;

    mpbIsModelValid = new bool[mnK];

    mpfDeterminants=new float[mnK];
    mpfMeanVector=new float[mnK*mnDimension];

    mpfCovVectors= new MatrixXd[mnK];
    mpfInvCovVectors = new MatrixXd[mnK];

    for(int i=0;i<mnK;i++){
        mpfCovVectors[i].resize(mnDimension,mnDimension);
        mpfInvCovVectors[i].resize(mnDimension,mnDimension);
    }
    normSumK = new float[mnK];
    mnNumOfInputs=0;
    mpfInputData.clear();
    }
bool GMM::initializeAndClear(int dimension,int k){
    mbIsFirstRun = true;
    if((k<1)||(dimension<1)){
        cerr<<"Zero dimension or zero number of classes"<<endl;
        return false;
    }

    if(mnK!=0){
        delete [] mpfDeterminants;
        delete [] mpfMeanVector;
        delete [] mpfCovVectors;
        delete [] mpfInvCovVectors;
        delete [] mpbIsModelValid;
        }
    mnDimension=dimension;
    mnK=k;
        normSumK = new float[mnK];
    mpbIsModelValid = new bool[mnK];
    mpfDeterminants=new float[mnK];
    mpfMeanVector=new float[mnK*mnDimension];
    mpfCovVectors= new MatrixXd[mnK];
    mpfInvCovVectors = new MatrixXd[mnK];
    for(int i=0;i<mnK;i++){
        mpfCovVectors[i].resize(mnDimension,mnDimension);
        mpfInvCovVectors[i].resize(mnDimension,mnDimension);
    }
    mnNumOfInputs=0;
    mpfInputData.clear();
    return true;
    }
void GMM::insertData(float* inVector,int inSize){
    mnNumOfInputs+=inSize;
    for(int i=0;i<inSize*mnDimension;i++)
        mpfInputData.push_back(inVector[i]);

    }
void GMM::clear(){
    mnNumOfInputs=0;
    mpfInputData.clear();
    mbIsFirstRun = true;
    }


float GMM::getLikelihood(int inputId,int modelId){
    float mulSum;
    float logLL;


    if(!mpbIsModelValid[modelId])
        return 0;
    logLL = pow(mpfDeterminants[modelId],0.5);
    logLL *= pow(2*PI,mnDimension*0.5);
    logLL = 1/logLL;
    mulSum = 0;
    for(int a=0;a<mnDimension;a++)
        for(int b=0;b<mnDimension;b++)
            mulSum += (mpfInputData[inputId*mnDimension+a]-mpfMeanVector[mnDimension*modelId+a])*(mpfInputData[inputId*mnDimension+b]-mpfMeanVector[mnDimension*modelId+b])*mpfInvCovVectors[modelId](a,b);
    logLL *= exp((-0.5)*mulSum);

    if (logLL<0)
        logLL = 0;
    if(isnan(logLL))
        logLL =  0;
    return logLL;


    }
float GMM::getLikelihood2(float* inputVec){

    float mulSum;
    float logLL;
    float maxLL;
    maxLL = -100000;

    for(int j=0;j<mnK;j++)
    {

        if(!mpbIsModelValid[j])
            continue;
        logLL = pow(mpfDeterminants[j],0.5);
        logLL *= pow(2*PI,mnDimension*0.5);
        logLL = 1/logLL;
        mulSum = 0;
        for(int a=0;a<mnDimension;a++)
            for(int b=0;b<mnDimension;b++)
                mulSum += (inputVec[a]-mpfMeanVector[mnDimension*j+a])*(inputVec[b]-mpfMeanVector[mnDimension*j+b])*mpfInvCovVectors[j](a,b);

        logLL *= exp((-0.5)*mulSum);

        if(logLL>maxLL)
            maxLL=logLL;
    }
    if(maxLL<0)
        maxLL = 0;

    return maxLL;


    }

float GMM::getLikelihood(float* inputVec){
  //  float a=0;
 //   for(int i=0;i<mnK;i++)
 //       a+=normSumK[i];
//    cout<<"AAAA"<<a<<endl;
    float mulSum;
    float logLL;
    float maxLL;
    maxLL = -100000;
    int b=0;
    for(int j=0;j<mnK;j++)
    {
        if(!mpbIsModelValid[j])
            continue;
        logLL = pow(mpfDeterminants[j],0.5);
        logLL *= pow(2*PI,mnDimension*0.5);
        logLL = 1/logLL;
        mulSum = 0;
        for(int a=0;a<mnDimension;a++)
            for(int b=0;b<mnDimension;b++)
                mulSum += (inputVec[a]-mpfMeanVector[mnDimension*j+a])*(inputVec[b]-mpfMeanVector[mnDimension*j+b])*mpfInvCovVectors[j](a,b);

        logLL *= exp((-0.5)*mulSum);
        logLL *= normSumK[j];
        if(logLL>maxLL)
            maxLL=logLL;


    }
    if(maxLL<0)
        maxLL = 0;

    return maxLL;


    }

void GMM::initLastMean(){
    if(!mbIsFirstRun)
        return;
    mbIsFirstRun=false;
    //Initialize all models again
    for(int i=0;i<mnK;i++)
        mpbIsModelValid[i]=true;

    mpfWeights.resize((mpfInputData.size()/mnDimension)*mnK);
    // Expectation
    for(int k=0;k<mnK;k++){
        if(!mpbIsModelValid)
            continue;
        for(int l=0;l<mnDimension;l++)
            mpfCovVectors[k](l,l)=mpfCovVectors[k](l,l)+pow(10,-6);
        //cout<<"Covariance"<<k<<endl<<mpfCovVectors[k]<<endl;
        mpfDeterminants[k]=mpfCovVectors[k].determinant();
        mpfInvCovVectors[k]=mpfCovVectors[k].inverse();
        for(int j=0;j<((int)mpfInputData.size())/mnDimension;j++){
            mpfWeights[j*mnK+k]=getLikelihood(j,k);
            if(isnan(mpfWeights[j*mnK+k]))
                mpfWeights[j*mnK+k] = 0;
        }
    }

    int normSum;
    ///////////////////////////////////
    //Normalize weights
    for(int j=0;j<(((int)mpfInputData.size())/mnDimension);j++)
    {
        normSum = 0;
        for(int k=0;k<mnK;k++)
            normSum+=mpfWeights[j*mnK+k];
        if(normSum!=0)
        {
            for(int k=0;k<mnK;k++)
                mpfWeights[j*mnK+k]=mpfWeights[j*mnK+k]/normSum;
        }
    }





}


void GMM::iterateGMM(int numIter){
    float normSum;
   // srand(time(NULL));
    if(mbIsFirstRun){
        //Allocate vectors
        mpfWeights.resize((mpfInputData.size()/mnDimension)*mnK);
        //Initialize all models
        for(int i=0;i<mnK;i++)
            mpbIsModelValid[i]=true;
        //Initialize all weights
        for(int i=0;i<(int)mpfWeights.size();i++)
            mpfWeights[i]=rand()%1000;
        //Normalize weights
        for(int i=0;i<(((int)mpfInputData.size())/mnDimension);i++)
        {
            normSum = 0;
            for(int j=0;j<mnK;j++)
                normSum+=mpfWeights[i*mnK+j];
            if(normSum!=0)
            {
                for(int j=0;j<mnK;j++)
                    mpfWeights[i*mnK+j]=mpfWeights[i*mnK+j]/normSum;
            }
        }
        mbIsFirstRun = false;
    }else{
        //Initialize all models again
        for(int i=0;i<mnK;i++)
            mpbIsModelValid[i]=true;
        //If number of inputs is changed, update the weights
        //Insert new ones randomly
        if((((int)mpfInputData.size())/mnDimension)!=(((int)mpfWeights.size())/mnK)){
                int oldSize = mpfWeights.size();
                mpfWeights.resize((mpfInputData.size()/mnDimension)*mnK);
                for(int i=oldSize;i<(int)mpfWeights.size();i++)
                    mpfWeights[i]=rand()%1000;
                for(int i=oldSize/mnK;i<((int)mpfWeights.size())/mnK;i++)
                {
                    normSum = 0;
                    for(int j=0;j<mnK;j++)
                        normSum+=mpfWeights[i*mnK+j];
                    if(normSum!=0)
                    {
                        for(int j=0;j<mnK;j++)
                            mpfWeights[i*mnK+j]=mpfWeights[i*mnK+j]/normSum;
                    }
                }
            }
        }

        for(int i=0;i<numIter;i++){
            //Clear containers
            for(int j=0;j<mnK;j++){
                for(int k=0;k<mnDimension;k++)
                    mpfMeanVector[j*mnDimension+k] = 0;
                normSumK[j]=0;
                mpfCovVectors[j]= MatrixXd::Zero(mnDimension,mnDimension);
            }
            //Normalize weights
            for(int j=0;j<(((int)mpfInputData.size())/mnDimension);j++)
            {
                normSum = 0;
                for(int k=0;k<mnK;k++)
                    normSum+=mpfWeights[j*mnK+k];
                if(normSum!=0)
                {
                    for(int k=0;k<mnK;k++)
                        mpfWeights[j*mnK+k]=mpfWeights[j*mnK+k]/normSum;
                }
            }
            //Maximization
            for(int j=0;j<((int)mpfInputData.size())/mnDimension;j++)
            {
                for(int k=0;k<mnK;k++)
                {
                    for(int l=0;l<mnDimension;l++)
                    {
                        for(int z=0;z<mnDimension;z++)
                            mpfCovVectors[k](l,z)+=mpfWeights[j*mnK+k]*mpfInputData[j*mnDimension+l]*mpfInputData[j*mnDimension+z];

                        mpfMeanVector[k*mnDimension+l]+=mpfWeights[j*mnK+k]*mpfInputData[j*mnDimension+l];
                    }
                    normSumK[k]+=mpfWeights[j*mnK+k];
                }
            }
            //If not used at all ignore
            for(int k=0;k<mnK;k++)
                if(normSumK[k]==0)
                    mpbIsModelValid[k]=false;
            //Normalize
            for(int k=0;k<mnK;k++){
                if(!mpbIsModelValid)
                    continue;
                for(int j=0;j<mnDimension;j++){
                    mpfMeanVector[k*mnDimension+j]=mpfMeanVector[k*mnDimension+j]/normSumK[k];
                    for(int l=0;l<mnDimension;l++)
                        mpfCovVectors[k](j,l)=mpfCovVectors[k](j,l)/normSumK[k];
                }
            }
            //Compute covariance
            for(int k=0;k<mnK;k++)
                for(int l=0;l<mnDimension;l++)
                    for(int z=0;z<mnDimension;z++)
                        mpfCovVectors[k](l,z)=mpfCovVectors[k](l,z) - mpfMeanVector[k*mnDimension+l]*mpfMeanVector[k*mnDimension+z];
            // Convergence check
            // Expectation
            for(int k=0;k<mnK;k++){
                if(!mpbIsModelValid)
                    continue;
                for(int l=0;l<mnDimension;l++)
                        mpfCovVectors[k](l,l)=mpfCovVectors[k](l,l)+pow(10,-6);
                //cout<<"Covariance"<<k<<endl<<mpfCovVectors[k]<<endl;
                mpfDeterminants[k]=mpfCovVectors[k].determinant();
                /*
                for(int nI=0;nI<mnDimension;nI++)
                    for(int nJ=0;nJ<mnDimension;nJ++)
                        if(nI==nJ)
                            mpfInvCovVectors[k](nI,nJ)=1/mpfCovVectors[k](nI,nJ);
                        else
                            mpfInvCovVectors[k](nI,nJ)=0;
*/
                mpfInvCovVectors[k]=mpfCovVectors[k].inverse();
                //cout<<"Inv Covariance"<<k<<endl<<mpfInvCovVectors[k]<<endl;
                //cout<<mpfMeanVector[k*mnDimension+0]<<" "<<mpfMeanVector[k*mnDimension+1]<<" "<<mpfMeanVector[k*mnDimension+2]<<" "<<mpfMeanVector[k*mnDimension+3]<<endl;
                for(int j=0;j<((int)mpfInputData.size())/mnDimension;j++){
                    mpfWeights[j*mnK+k]=getLikelihood(j,k);
                    if(isnan(mpfWeights[j*mnK+k]))
                        mpfWeights[j*mnK+k] = 0;
                }
                }
        }

        float a;
        for(int i=0;i<mnK;i++)
            a+=normSumK[i];
        for(int i=0;i<mnK;i++)
            normSumK[i]=normSumK[i]/a;
}

void GMM::printModels(){
    for(int k=0;k<mnK;k++){
        if(mpbIsModelValid[k])
        cout<<endl<<"Model "<<k<<":"<<endl<<"\t";
        for(int i=0;i<mnDimension;i++)
            cout<<mpfMeanVector[k*mnDimension+i]<<"\t";
        }

    }
