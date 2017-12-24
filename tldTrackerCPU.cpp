/***********************************************************************
  $FILENAME    : tldTrackerCPU.cpp

  $TITLE       : TLD tracker class implementation for CPU

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Defines the TLD tracker class for running on CPU

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef USE_OCL_

#include <stdexcept>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Constants.h"
#include "BoundingBox.hpp"
#include "Distance.hpp"
#include "Statistics.hpp"
#include "Linkage.hpp"
#include "Image.hpp"
#include "tldTracker.hpp"

using namespace std;
using namespace cv;

#define PROFILINING_START_TIME( )                    \
                start = clock()

#define PROFILINING_STOP_TIME( string )               \
                stop = clock();                       \
                elapsedTime = (float)(stop - start) / \
                (float)CLOCKS_PER_SEC * 1000.0f;      \
                nowis = time(0);                      \
                localtm = localtime(&nowis);          \
                strftime (timeBuffer, 256, "%b/%e/%Y,%H:%M:%S", localtm); \
                profilingLogFile << timeBuffer << string << elapsedTime << "\n"


tld::cpu::tracker::tracker( trackerApp* pApp )
{
  m_pApp = pApp;
  initDone_ = false;
  count_ = 0;

  // print a welcome message
  cout << "Standalone tracker based on TLD (Tracking, learning, detection (aka Predator)) algorithm.\n"
      << "TLD has been developed by Zdenek Kalal.\n"
      << "TLD has been converted to C++ by Armin Zare Zadeh.\n";

  cout << "Current instance: CPU\n";

}

tld::cpu::tracker::~tracker()
{
#ifdef PROFILING_ON_
  profilingLogFile.close();
#endif

  delete lucaskanade; lucaskanade = 0;
  delete fern; fern = 0;

  // Free the display image
  tmpBlurImg.release();

  detector.grid.release();
  detector.scales.release();
  detector.features.x.release();
  detector.tmp.conf.release();
  detector.tmp.patt.release();

  detector.dt.bb.release();
  detector.dt.patt.release();
  detector.dt.idx.release();
  detector.dt.conf1.release();
  detector.dt.conf2.release();
  detector.dt.isin.release();
  detector.dt.patch.release();

  trainer.X.release();
  trainer.Y.release();
  trainer.pEx.release();
  trainer.nEx.release();
  trainer.pex.release();
  trainer.nex.release();

  TRDT.curBBox.release();
  TRDT.prvBBox.release();
  TRDT.xFJ.release();
  TRDT.xFJJ.release();
  TRDT.xFI.release();
  TRDT.xFII.release();
  TRDT.idxF_.release();
  TRDT.tBB.release();
  TRDT.dBB.release();
  TRDT.dConf.release();

  patchPatt1D.release();
  isin_NN1D.release();

#ifdef FERN_OPENCV_ON_
  TRDT.object.release();
#endif
}

void
tld::cpu::tracker::process ( )
{
  count_++;
  if( initDone_ ){
    TRDT.curBBox.copyTo(TRDT.prvBBox);
    TRDT.prvValid = TRDT.curValid;
    TRDT.prvSize = TRDT.curSize;
    TRDT.prvConf = TRDT.curConf;

//    if ( count_ == 2 ){
    processFrame();
//    }
  }else{
    if ( count_ == 1 ){
      init();

    }
    initDone_ = true;
  }
}

void
tld::cpu::tracker::init( )
{
#ifdef PROFILING_ON_
  profilingLogFile.open ("profilinglogCPU.txt", ofstream::out | ofstream::app);
#endif

  lucaskanade = new LucasKanade(m_pApp->curGrayImg);
  fern = new Fern( );

  // INITIALIZE DETECTOR =====================================================
  TRDT.numM = 10; // numMxnumN grid of points within BBox
  TRDT.numN = 10;
  TRDT.numMN = TRDT.numM * TRDT.numN;
  TRDT.xFI.create(2, TRDT.numMN, CV_32FC1);
  TRDT.xFI.setTo(Scalar(0));
  TRDT.xFII.create(2, TRDT.numMN, CV_32FC1);
  TRDT.xFII.setTo(Scalar(0));
  TRDT.xFJ.create(2, TRDT.numMN, CV_32FC1);
  TRDT.xFJ.setTo(Scalar(0));
  TRDT.xFJJ.create(4, TRDT.numMN, CV_32FC1);
  TRDT.xFJJ.setTo(Scalar(0));
  TRDT.idxF_.create(1, TRDT.numMN, CV_32SC1);
  TRDT.idxF_.setTo(Scalar(0));
  TRDT.idxF = (int*)TRDT.idxF_.data;
  TRDT.numMN = 0;
  TRDT.tBB.create(4, 1, CV_32FC1);
  TRDT.dBB.create(4, DT_MAX_BB, CV_32FC1);
  TRDT.dConf.create(1, DT_MAX_BB, CV_32FC1);
  TRDT.DTNum = 0;

  TRDT.prvConf = 0;
  TRDT.curConf = 0;
  TRDT.curSize = 0;
  TRDT.prvSize = 0;
  TRDT.prvValid = 0;
  TRDT.curValid = 0;

  TRDT.bbox_rect = m_pApp->tldargs.source.bbox;
  TRDT.curBBox.create(4, 1, CV_32FC1);
  rect2BBox<float>(m_pApp->tldargs.source.bbox, TRDT.curBBox);

  patchPatt1D.create( m_pApp->tldargs.model.patchsize[0]*m_pApp->tldargs.model.patchsize[1], 1, CV_32FC1 );
  isin_NN1D.create(3, 1, CV_32SC1);

  //  saveMatrix<float>(TRDT.curBBox);

  // Scanning grid
  // output: detector.grid, detector.scales
  // Grid format :
  //     (bb[0],bb[1]) : X1,Y1
  //     (bb[2],bb[3]) : X2,Y2
  //     bb[4] : pointer to features for this scale ((bb[4]-1)*2*nFEAT*nTREES)
  //     bb[5] : number of left-right bboxes, will be used for searching neighbours
  // Example: grid: 6x30255, scales: 2x11
  bboxScan<float>( TRDT.curBBox, m_pApp->curGrayImg.cols, m_pApp->curGrayImg.rows, m_pApp->tldargs.model.min_win, detector.grid, detector.scales );

  // Features
  detector.nGrid = detector.grid.cols;
  // output: detector.features
  // Example: features.x: 52x10, features.type = 1; "forest"
  generateFeatures( m_pApp->tldargs.model.num_trees, m_pApp->tldargs.model.num_features, detector.features.x );

  //  saveMatrix(detector.features.x);

  detector.features.type = 1; //"forest"

  // Initialize Detector
  fern->init(m_pApp->curGrayImg, detector.grid, detector.features.x, detector.scales); // allocate structures

  // Temporal structures for fern
  //  detector.temporal.conf = zeros(1, detector.nGrid);
  detector.tmp.conf.create(1, detector.nGrid, CV_32FC1); // fern
  detector.tmp.conf.setTo(Scalar(0.));
  //  detector.temporal.patt = zeros(m_pApp->tldargs.model.num_trees, detector.nGrid);
  detector.tmp.patt.create(m_pApp->tldargs.model.num_trees, detector.nGrid, CV_32SC1); // fern
  detector.tmp.patt.setTo(Scalar(0));

  // initialize detection structure
  detector.dt.bb.create(4, DT_MAX_BB, CV_32FC1); // bounding boxes
  detector.dt.patt.create(detector.tmp.patt.rows, DT_MAX_BB, CV_32SC1); // corresponding codes of the Ensemble Classifier
  detector.dt.patt.setTo(Scalar(0));
  detector.dt.idx.create(1, DT_MAX_BB, CV_32SC1); // indexes of detected bounding boxes within the scanning grid
  detector.dt.idx.setTo(Scalar(NANN));
  detector.dt.conf1.create(1, DT_MAX_BB, CV_32FC1); // Relative Similarity (for final nearest neighbor classifier)
  detector.dt.conf1.setTo(Scalar(NANN));
  detector.dt.conf2.create(1, DT_MAX_BB, CV_32FC1); // Conservative Similarity (for integration with tracker)
  detector.dt.conf2.setTo(Scalar(NANN));
  detector.dt.isin.create(4, DT_MAX_BB, CV_32SC1); // detected (isin=1) or rejected (isin=0) by nearest neighbor classifier
  detector.dt.isin.setTo(Scalar(NANN));
  // For Nearest Neighbor
  detector.dt.patch.create(m_pApp->tldargs.model.patchsize[0]*m_pApp->tldargs.model.patchsize[1], DT_MAX_BB, CV_32FC1); // Corresponding patches
  detector.dt.patch.setTo(Scalar(NANN));

  // RESULTS =================================================================

  // Initialize Trajectory
  //  TrajectoryLocal *traj;
  //  ACE_NEW_RETURN (traj, TrajectoryLocal (m_pApp->tldargs.source.bb, 1, 1, 1), -1);
  //  insert2TrajList (traj);

  // TRAIN DETECTOR ==========================================================

  // Initialize structures
  imgsize = Size(m_pApp->curGrayImg.cols,m_pApp->curGrayImg.rows);
  Mat overlap;
  bboxOverlap<float>(TRDT.curBBox, detector.grid, overlap);

  // Generate Positive Examples
  //[pX,pEx,bbP] = generatePositiveData(trainer.overlap, videoFrame, m_pApp->tldargs.p_par_init);
  Mat pX; // training data for fern EX: 10x200
  // pEx : patchPatt1D training data for Nearest Neighbor EX: 225x1
  Mat bbP; // 4x1
  generatePositiveData(overlap, 0, pX, patchPatt1D, bbP);
  //generatePositiveData(overlap, 0, trainer.X.pX, trainer.pEx, bbP);

  //pY = ones(1,size(pX,2));

  // Correct initial bbox
  //traj->bb = bbP;

  TRDT.prvBBox = TRDT.curBBox.clone();

  bbP.copyTo(TRDT.curBBox);
  //tld.bb(:,1) = bbP(1:4,:);

  // Variance threshold Ex: 1323
  trainer.var = (float)variance1D<float>(patchPatt1D);
  trainer.var = trainer.var/2.0;
  //var(pEx(:,1))/2

  // Generate Negative Examples
  Mat nX; // 10x4698
  Mat nEx; // 225x100
  generateNegativeData(overlap, nX, nEx);

  Mat nX1; // 10x2349
  Mat nX2; // 10x2349
  Mat nEx1; // 225x50
  Mat nEx2; // 225x50
  // Split Negative Data to Training set and Validation set
  splitNegativeData(nX, nEx, nX1, nX2, nEx1, nEx2);

  //  nY1  = zeros(1,size(nX1,2));

  // Generate Apriori Negative Examples
  //[anX,anEx] = generateAprioriData(tld);
  //anY = zeros(1,size(anX,2));
  // disp(['# apriori N patterns: ' num2str(size(anX,2))]);
  // disp(['# apriori N patches : ' num2str(size(anEx,2))]);

  trainer.pEx  = patchPatt1D.clone(); // save positive patches for later
  trainer.nEx  = nEx.clone(); // save negative patches for later

  if (pX.rows != nX1.rows){
    cout << "\nLM_ERROR:pX->rows != nX1->rows: " << pX.rows << "," << nX1.rows << "\n"
        << endl;
    CV_Assert( pX.rows == nX1.rows );
  }

  Mat X(pX.rows, pX.cols+nX1.cols, CV_32SC1); // 10x2549
  Mat Y(1, X.cols, CV_32SC1);
  for (int j=0; j<X.cols; j++){
    if (j<pX.cols){ //
      Y.at<int>(0,j) = 1;
      for (int i=0; i<X.rows; i++){ // ... <
        X.at<int>(i,j) = pX.at<int>(i,j);
      }
    } else { //  =< ... <
      Y.at<int>(0,j) = 0;
      for (int i=0; i<X.rows; i++){
        X.at<int>(i,j) = nX1.at<int>(i,j-pX.cols);
      }
    }
  }
  //  X    = [pX nX1];
  //  Y    = [pY nY1];

  int* idx = new int[X.cols];
  randomPerm<int>( X.cols, idx);
  //  idx         = randperm(size(tld.X{1},2));

  trainer.X.create(X.rows, X.cols, CV_32SC1); // training data for fern
  for (int j=0; j<X.cols; j++){
    for (int i=0; i<X.rows; i++){
      trainer.X.at<int>(i,j) = X.at<int>(i,idx[j]);
    }
  }
  //  tld.X{1}    = tld.X{1}(:,idx);
  trainer.Y.create(Y.rows, Y.cols, CV_32SC1);
  for (int j=0; j<Y.cols; j++){
    for (int i=0; i<Y.rows; i++){
      trainer.Y.at<int>(i,j) = Y.at<int>(i,idx[j]);
    }
  }
  //  tld.Y{1}    = tld.Y{1}(:,idx);

  // Train using training set ------------------------------------------------

  // Fern
  int bootstrap = 2;
  //  fern(2,tld.X{1},tld.Y{1},tld.model.thr_fern,bootstrap);
  fern->update( trainer.X, trainer.Y, m_pApp->tldargs.model.thr_fern, bootstrap );

  // Nearest Neighbor

  //  tld = trainNearestNeighbor(pEx,nEx1,tld);
  trainNearestNeighbor(patchPatt1D, nEx1);

  m_pApp->tldargs.model.num_init = trainer.pex.cols;
  //model.num_init = size(tld.pex,2);

  // Estimate thresholds on validation set  ----------------------------------

  // Fern
  //  conf_fern = fern(3,nX2);
  Mat conf_fern;
  fern->evaluate( nX2, conf_fern ); // 1x2349

  float maxtemp = -10000;
  for (int j=0; j<conf_fern.cols; j++){
    if (conf_fern.at<float>(0,j) > maxtemp){
      maxtemp = conf_fern.at<float>(0,j);
    }
  }
  maxtemp = maxtemp/m_pApp->tldargs.model.num_trees; // 0.4
  m_pApp->tldargs.model.thr_fern = fmax(maxtemp, m_pApp->tldargs.model.thr_fern); // 0.5 todo
  //  tld.model.thr_fern = max(max(conf_fern)/tld.model.num_trees,tld.model.thr_fern);

  // Nearest neighbor
  Mat isin;
  Mat conf1;
  Mat conf2;
  NearestNeighbor (nEx2, isin, conf1, conf2);
  //  conf_nn = NearestNeighbor(nEx2,tld);

  maxtemp = -10000;
  for (int j=0; j<conf1.cols; j++){
    if (conf1.at<float>(0,j) > maxtemp){
      maxtemp = conf1.at<float>(0,j); // 0.446
    }
  }
  m_pApp->tldargs.model.thr_nn = fmax(maxtemp, m_pApp->tldargs.model.thr_nn); // 0.65 todo max 2 min
  //  tld.model.thr_nn = max(tld.model.thr_nn,max(conf_nn));
  m_pApp->tldargs.model.thr_nn_valid = fmax(m_pApp->tldargs.model.thr_nn_valid, m_pApp->tldargs.model.thr_nn); // todo
  //  tld.model.thr_nn_valid = max(tld.model.thr_nn_valid,tld.model.thr_nn);

#ifdef FERN_OPENCV_ON_
  ObjectDetectorTrain();
#endif

  delete []idx;

  conf1.release();
  conf2.release();
  isin.release();
  conf_fern.release();
  nEx.release();

  bbP.release();
  nX.release();
  nX1.release();
  nX2.release();
  nEx1.release();
  nEx2.release();
  X.release();
  Y.release();
  overlap.release();

  return;
}

void
tld::cpu::tracker::generateFeatures ( const int nTREES, const int nFEAT, OutputArray _features )
{
  float SHI = 1.0/5; // 0.20000
  int SCA = 1;
  float OFF = SHI;

  // 0:SHI:1 = 0.00000   0.20000   0.40000   0.60000   0.80000   1.00000
  float temp1[6] = {0.00000, 0.20000, 0.40000, 0.60000, 0.80000, 1.00000};
  //x = repmat(nTuples(0:SHI:1,0:SHI:1),2,1);
  Mat x1(4, 36, CV_32FC1);
  int col = 0;
  for (int jj=0; jj<6; jj++){
    for (int ii = 0; ii<6; ii++){
      x1.at<float>(0,col) = temp1[jj];
      x1.at<float>(2,col) = temp1[jj];
      col++;
    }
  }
  col = 0;
  for (int ii = 0; ii<6; ii++){
    for (int jj=0; jj<6; jj++){
      x1.at<float>(1,col) = temp1[jj];
      x1.at<float>(3,col) = temp1[jj];
      col++;
    }
  }

  Mat x(x1.rows, x1.cols*2, CV_32FC1); // 4x72
  for (int j=0; j<x.cols; j++){
    if (j<x1.cols){
      for (int i=0; i<x.rows; i++){
        x.at<float>(i,j) = x1.at<float>(i,j);
      }
    }else{
      for (int i=0; i<x.rows; i++){
        x.at<float>(i,j) = x1.at<float>(i,j-x1.cols)+SHI/2.0;
      }
    }
  }
  // x = [x1 x1 + SHI/2];

  x1.release();

  // k = x->cols;
  //k = size(x,2);

  float *random_ = new float[x.cols];

  srand ( time(0) );
  Mat r(x.rows, x.cols, CV_32FC1); // 4x72
  for (int j=0; j<x.cols; j++){
    random_[j] = SCA*(rand()/(float(RAND_MAX)+1))+OFF;
  }
  for (int j=0; j<r.cols; j++){ // 72
    r.at<float>(0,j) = x.at<float>(0,j);
    r.at<float>(1,j) = x.at<float>(1,j);
    r.at<float>(2,j) = x.at<float>(2,j)+random_[j];
    r.at<float>(3,j) = x.at<float>(3,j);
  }
  // r = x; r(3,:) = r(3,:) + (SCA*rand(1,k)+OFF);

  srand ( time(0) );
  Mat l(x.rows, x.cols, CV_32FC1); // 4x72
  for (int j=0; j<x.cols; j++){
    random_[j] = SCA*(rand()/(float(RAND_MAX)+1))+OFF;
  }
  for (int j=0; j<l.cols; j++){
    l.at<float>(0,j) = x.at<float>(0,j);
    l.at<float>(1,j) = x.at<float>(1,j);
    l.at<float>(2,j) = x.at<float>(2,j)-random_[j];
    l.at<float>(3,j) = x.at<float>(3,j);
  }
  // l = x; l(3,:) = l(3,:) - (SCA*rand(1,k)+OFF);

  srand ( time(0) );
  Mat t(x.rows, x.cols, CV_32FC1); // 4x72
  for (int j=0; j<x.cols; j++){
    random_[j] = SCA*(rand()/(float(RAND_MAX)+1))+OFF;
  }
  for (int j=0; j<t.cols; j++){
    t.at<float>(0,j) = x.at<float>(0,j);
    t.at<float>(1,j) = x.at<float>(1,j);
    t.at<float>(2,j) = x.at<float>(2,j);
    t.at<float>(3,j) = x.at<float>(3,j)-random_[j];
  }
  // t = x; t(4,:) = t(4,:) - (SCA*rand(1,k)+OFF);

  srand ( time(0) );
  Mat b(x.rows, x.cols, CV_32FC1); // 4x72
  for (int j=0; j<x.cols; j++){
    random_[j] = SCA*(rand()/(float(RAND_MAX)+1))+OFF;
  }
  for (int j=0; j<b.cols; j++){
    b.at<float>(0,j) = x.at<float>(0,j);
    b.at<float>(1,j) = x.at<float>(1,j);
    b.at<float>(2,j) = x.at<float>(2,j);
    b.at<float>(3,j) = x.at<float>(3,j)+random_[j];
  }
  // b = x; b(4,:) = b(4,:) + (SCA*rand(1,k)+OFF);

  delete []random_;
  x.release();

  x.create(r.rows, r.cols*4, CV_32FC1); // 4x288
  for (int j=0; j<x.cols; j++){
    if (j<r.cols){ // 72
      for (int i=0; i<x.rows; i++){ // ... < 72
        CV_Assert( j < r.cols );
        x.at<float>(i,j) = r.at<float>(i,j);
      }
    } else if (j>=r.cols && j<r.cols*2){ // 72 =< ... < 72*2
      for (int i=0; i<x.rows; i++){
        CV_Assert( j-r.cols < l.cols );
        x.at<float>(i,j) = l.at<float>(i,j-r.cols);
      }
    }else if (j>=r.cols*2 && j<r.cols*3){ // 72*2 =< ... < 72*3
      for (int i=0; i<x.rows; i++){
        CV_Assert( j-r.cols*2 < t.cols );
        x.at<float>(i,j) = t.at<float>(i,j-r.cols*2);
      }
    }else if (j>=r.cols*3 && j<r.cols*4){ // 72*3 =< ...
      for (int i=0; i<x.rows; i++){
        CV_Assert( j-r.cols*3 < b.cols );
        x.at<float>(i,j) = b.at<float>(i,j-r.cols*3);
      }
    }
  }
  // x = [r l t b];

  r.release();
  l.release();
  t.release();
  b.release();

  int cnt = 0;
  Mat idx(1, x.cols, CV_32SC1);
  for (int j=0; j<x.cols; j++){
    if (x.at<float>(0,j)<1 &&
        x.at<float>(0,j)>0 &&
        x.at<float>(1,j)<1 &&
        x.at<float>(1,j)>0
    ){
      idx.at<int>(0,j) = 1;
      cnt++;
    }else{
      idx.at<int>(0,j) = 0;
    }
  }
  // idx = all(x([1 2],:) < 1 & x([1 2],:) > 0,1);

  Mat xx(x.rows, cnt, CV_32FC1); // 4x...
  cnt = 0;
  for (int j=0; j<x.cols; j++){
    if (idx.at<int>(0,j) == 1){
      for (int i=0; i<x.rows; i++){
        xx.at<float>(i,cnt) = x.at<float>(i,j);
      }
      cnt++;
    }
  }
  // x = x(:,idx);

  x.release();
  idx.release();

  for (int j=0; j<xx.cols; j++){
    for (int i=0; i<xx.rows; i++){
      if (xx.at<float>(i,j) > 1){
        xx.at<float>(i,j) = 1;
      }else if (xx.at<float>(i,j) < 0){
        xx.at<float>(i,j) = 0;
      }
    }
  }
  // x(x > 1) = 1;
  // x(x < 0) = 0;

  int numF = xx.cols;
  // numF = size(x,2);
  int* perm = new int[numF];
  randomPerm<int>( numF, perm);
  x.create(xx.rows, xx.cols, CV_32FC1);
  for (int j=0; j<xx.cols; j++){
    for (int i=0; i<xx.rows; i++){
      x.at<float>(i,j) = xx.at<float>(i,perm[j]);
    }
  }
  //x = x(:,randperm(numF));

  delete []perm;
  xx.release();

  CV_Assert( nFEAT*nTREES <= x.cols );

  Mat xxx(4, nFEAT*nTREES, CV_32FC1);
  for (int j=0; j<xxx.cols; j++){
    for (int i=0; i<xxx.rows; i++){
      // 0.59,0.8,0.9,...
      // 0.8,0.59,0.1,...
      // 0.93,1,0.11,...
      // 0.8,0.59,0.1,...
      xxx.at<float>(i,j) = x.at<float>(i,j);
    }
  }
  // x = x(:,1:nFEAT*nTREES); 10 13

  x.release();

  _features.create( 4*nFEAT, nTREES, CV_32FC1 ); // 4x13 10
  Mat features = _features.getMat();

  int tmp_col = 0, tmp_row = 0;
  for (int i=0; i<features.cols; i++){ // cols:10
    for (int j=0; j<features.rows; j++){ // rows:4x13
      CV_Assert( tmp_row < xxx.rows && tmp_col < xxx.cols );
      features.at<float>(j,i) = xxx.at<float>(tmp_row,tmp_col);
      tmp_row++;
      if (tmp_row >= xxx.rows) {
        tmp_col++;
        if (tmp_col >= xxx.cols) tmp_col = 0;
        tmp_row = 0;
      }
    }
  }
  // x = reshape(x,4*nFEAT,nTREES);

  //features = x;

  xxx.release();

}

void
tld::cpu::tracker::generatePositiveData ( InputArray _overlap, const int isRunTime, OutputArray _pX, OutputArray _pEx, OutputArray _bbP0 )
{
  Mat overlap_ = _overlap.getMat();
  float* overlap = (float*)overlap_.data;

  // Get closest bbox
  int idxM;
  float temp = 0.0;
  int idxPP_len = 0;
  Mat idxPP_(1, overlap_.cols, CV_32SC1);
  int* idxPP = (int*)idxPP_.data;
  for (int j = 0; j < overlap_.cols; j++){
    if (overlap[j] > temp){
      temp = overlap[j];
      idxM = j;
    }
    // [~,idxP] = max(overlap);
    if (overlap[j] > 0.6){
      idxPP[idxPP_len] = j;
      idxPP_len++;
    }
    // idxP = find(overlap > 0.6);
  }

  _bbP0.create( 4, 1, CV_32FC1 );
  Mat bbP0 = _bbP0.getMat();

  for (int j = 0; j < 4; j++){
    bbP0.at<float>(j,0) = detector.grid.at<float>(j,idxM);
  }
  //bbP0 = detector.grid(1:4,idxP);

  // Get overlapping bboxes
  int num_closest, num_warps;
  if (isRunTime == 0) {
    num_closest = m_pApp->tldargs.p_par_init.num_closest;
    num_warps = m_pApp->tldargs.p_par_init.num_warps;
  }else {
    num_closest = m_pApp->tldargs.p_par_update.num_closest;
    num_warps = m_pApp->tldargs.p_par_update.num_warps;
  }

  // idxP = find(overlap > 0.6);
  if (idxPP_len > num_closest) {
    for (int i = 0; i < idxPP_len-1; i++){
      for (int j = i+1; j < idxPP_len; j++){
        if (overlap[idxPP[j]] > overlap[idxPP[i]]){
          int temp = idxPP[i];
          idxPP[i] = idxPP[j];
          idxPP[j] = temp;
        }
      }
    }
    // [~,sIdx] = sort(overlap(idxPP),'descend');
    // idxPP = idxPP(sIdx(1:opt.p_par_init.num_closest));
  } else{
    idxPP_.release();
    bbP0.release();
    cout << "\ngeneratePositiveData: idxP_len <= num_closest!\n"
        << endl;
    //CV_Assert( idxPP_len > num_closest );
    return;
  }

  Mat idxP_(1, num_closest, CV_32SC1);
  int* idxP = (int*)idxP_.data;
  //idxPtest = cvCreateMat(1, num_closest, CV_32SC1);// todo
  for (int j = 0; j < num_closest; j++){
    idxP[j] = idxPP[j];
    //Coord_INT(idxPtest, 0, j) = Coord_INT(idxPP, 0, j);// todo
  }

  Mat bbP(detector.grid.rows, num_closest, CV_32FC1);
  for (int j = 0; j < num_closest; j++){
    for (int i = 0; i < detector.grid.rows; i++){
      bbP.at<float>(i,j) = detector.grid.at<float>(i,idxP[j]);
    }
  }
  // bbP  = detector.grid(:,idxPP);
  // if isempty(bbP), return;

  //  int inf = std::numeric_limits<int>::infinity();

  // Get hull
  Mat bbH(4, 1, CV_32FC1);
  for (int i = 0; i < 2; i++){
    float temp = 100000.;
    for (int j = 0; j < bbP.cols; j++){
      if (bbP.at<float>(i,j) < temp){
        temp = bbP.at<float>(i,j);
      }
    }
    bbH.at<float>(i,0) = temp;
  }
  for (int i = 2; i < 4; i++){
    float temp = 0.;
    for (int j = 0; j < bbP.cols; j++){
      if (bbP.at<float>(i,j) > temp){
        temp = bbP.at<float>(i,j);
      }
    }
    bbH.at<float>(i,0) = temp;
  }
  // bbH  = bb_hull(bbP);

  // training data for Nearest Neighbor
  //pEx : 225x1
  getPattern_1D ( bbP0, 0, _pEx );

//  Mat pEx = _pEx.getMat();

  // pEx : opt.model.patchsize[0]*opt.model.patchsize[1] x bbP0->cols
  if (m_pApp->tldargs.model.fliplr){
    //pEx = [pEx getPattern(videoFrame, bbP0, opt.model.patchsize,1)];
  }

  m_pApp->curBlurImg.copyTo(tmpBlurImg);
  // training data for fern
  _pX.create( detector.features.x.cols, num_closest*num_warps, CV_32SC1 ); // #Trees x num_closest*num_warps 10x200
  Mat pX = _pX.getMat();
  int colnum = 0;
  for (int i = 0; i<num_warps; i++){
    if (i > 0){
      Mat patch_blur;
      // patch_input = img_patch(im0.input,bbH,randomize,p_par);
      if (isRunTime == 0) {
        imgPatch<float>(m_pApp->curBlurImg, patch_blur, bbH, m_pApp->tldargs.p_par_init); // 43x26
      }else {
        imgPatch<float>(m_pApp->curBlurImg, patch_blur, bbH, m_pApp->tldargs.p_par_update);
      }
      for(int y=0; y<patch_blur.rows; y++) { // row : height y
        for(int x=0; x<patch_blur.cols; x++) { // col : width x
          tmpBlurImg.at<uchar>(y+(int)bbH.at<float>(1,0),x+(int)bbH.at<float>(0,0)) = patch_blur.at<uchar>(y,x);
        }
      }
      //tmpBlurImg(rows,cols) = patch_blur;

      Mat pxcell; // 10x10
      Mat status; // 1x10

      // Measures on blured image
      // GET PATTERNS: patterns = fern(5,img_gray,idx,minVar)
      //  pX  = [pX fern(5, im1, idxPP, 0)];
      fern->getPattern( m_pApp->curGrayImg, tmpBlurImg, idxP_, 0, pxcell, status );
      for (int jj=0; jj<pxcell.cols; jj++){ // col
        for (int ii = 0; ii<pxcell.rows; ii++){ // row
          CV_Assert( colnum < pX.cols );
          pX.at<int>(ii,colnum) = pxcell.at<int>(ii,jj);
        }
        colnum++;
      }
      patch_blur.release();
      pxcell.release();
      status.release();
    }else{
      Mat pxcell; // 10x10
      Mat status; // 1x10

      // Measures on blured image
      // GET PATTERNS: patterns = fern(5,img_gray,idx,minVar)
      //  pX  = [pX fern(5, im1, idxPP, 0)];
      fern->getPattern( m_pApp->curGrayImg, m_pApp->curBlurImg, idxP_, 0, pxcell, status );
      for (int jj=0; jj<pxcell.cols; jj++){ // col
        for (int ii = 0; ii<pxcell.rows; ii++){ // row
          CV_Assert( colnum < pX.cols );
          pX.at<int>(ii,colnum) = pxcell.at<int>(ii,jj);
        }
        colnum++;
      }

      pxcell.release();
      status.release();
    }
    // Measures on input image
    //pEx(:,i) = tldGetPattern(im1,bbP0,tld.model.patchsize);
    //pEx = [pEx tldGetPattern(im1,tld.grid(1:4,idxP),tld.model.patchsize)];
  }

  idxP_.release();
  idxPP_.release();
  bbH.release();

}

// get patch under bounding box (bb), normalize it size, reshape to a column
// vector and normalize to zero mean and unit variance (ZMUV)
void
tld::cpu::tracker::getPattern ( InputArray _bb, OutputArray _pattern )
{
  Mat bb = _bb.getMat();

  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb.rows == 6 || bb.rows == 4 );

  // initialize output variable
  int nBB = bb.cols;
  //nBB = size(bb,2);

  _pattern.create( m_pApp->tldargs.model.patchsize[0]*m_pApp->tldargs.model.patchsize[1], bb.cols, CV_32FC1 );
  Mat pattern = _pattern.getMat();
  // CvMat* pattern = zeros(prod(patchsize),nBB);

  // for every bounding box
  for (int i=0; i<nBB; i++){

    // sample patch
    Mat patch;
    imgPatch<float>(m_pApp->curGrayImg, patch, bb, i);

    // normalize size to 'patchsize' and nomalize intensities to ZMUV
    patch2Pattern(patch, m_pApp->tldargs.model.patchsize, pattern, i);
    patch.release();

  }

}

// get patch under bounding box (bb), normalize it size, reshape to a column
// vector and normalize to zero mean and unit variance (ZMUV)
void
tld::cpu::tracker::getPattern_1D ( InputArray _bb, const int colBBox, OutputArray _pattern )
{
  Mat bb = _bb.getMat();

  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb.rows == 6 || bb.rows == 4 );
  CV_Assert( colBBox < bb.cols );

  // sample patch
  Mat patch;
  imgPatch<float>(m_pApp->curGrayImg, patch, bb, colBBox);

//  _pattern.create( m_pApp->tldargs.model.patchsize[0]*m_pApp->tldargs.model.patchsize[1], 1, CV_32FC1 );
  Mat pattern = _pattern.getMat();
  // normalize size to 'patchsize' and nomalize intensities to ZMUV
  patch2Pattern(patch, m_pApp->tldargs.model.patchsize, pattern, 0);
  patch.release();
}

void
tld::cpu::tracker::patch2Pattern ( InputArray _patch, const int *patchsize, OutputArray _pattern, const int col)
{
  Mat patch = _patch.getMat();
  Mat pattern = _pattern.getMat();

  CV_Assert( col < pattern.cols );

  // width : patchsize[0]
  // height : patchsize[1]
  Mat patchImage( patchsize[0], patchsize[1], m_pApp->curGrayImg.type());

  resize(patch, patchImage, patchImage.size());

  //cvResize(patch, patchImage);
  //patch   = imresize(patch,patchsize); // 'bilinear' is faster

  //  CvMat* pattern = cvCreateMat(patchsize[0]*patchsize[1], 1, CV_32SC1);
  int cnt = 0;
  double mean = 0;
  for(int x=0; x<patchsize[0]; x++){ // width : patchsize[0]
    for(int y=0; y<patchsize[1]; y++){ // height : patchsize[1]
      CV_Assert( cnt < pattern.rows );
      pattern.at<float>(cnt,col) = (float)(patchImage.at<uchar>(y,x) & 0x000000FF);
      mean += (double)(pattern.at<float>(cnt,col));
      cnt++;
    }
  }
  // pattern = double(patch(:));
  mean /= (double)(pattern.rows);

  patchImage.release();

  for(int i=0; i<pattern.rows; i++){
    pattern.at<float>(i,col) = pattern.at<float>(i,col) - (float)mean;
  }
  // pattern = pattern - mean(pattern);

}

// Measure patterns on all bboxes that are far from initial bbox
void
tld::cpu::tracker::generateNegativeData ( InputArray _overlap, OutputArray _nX, OutputArray _nEx )
{
  Mat overlap = _overlap.getMat();

  int idxN_len = 0;
  Mat idxNN(1, overlap.cols, CV_32SC1);
  for (int j = 0; j < overlap.cols; j++){
    if (overlap.at<float>(0,j) < m_pApp->tldargs.n_par.overlap){
      idxNN.at<int>(0,idxN_len) = j;
      idxN_len++;
    }
    // idxN = find(overlap<tld.n_par.overlap);
  }
  if ( idxN_len == 0 ){
    idxNN.release();
    CV_Assert( idxN_len != 0 );
    return;
  }
  Mat idxNNN(1, idxN_len, CV_32SC1);
  for (int j = 0; j < idxN_len; j++){
    idxNNN.at<int>(0,j) = idxNN.at<int>(0,j);
  }
  idxNN.release();
  //idxN        = find(overlap<tld.n_par.overlap);

  Mat nXX;  // 10x29861
  Mat status;  // 1x29861
  fern->getPattern( m_pApp->curGrayImg, m_pApp->curBlurImg, idxNNN, trainer.var/2.0, nXX, status );
  //[nX,status] = fern(5,videoFrame,idxN,tld.var/2);

  idxN_len = 0;
  for (int j = 0; j < status.cols; j++){
    if (status.at<int>(0,j) == 1){
      idxN_len++;
    }
    //  idxN = idxN(status==1); // bboxes far and with big variance
  }
  if ( idxN_len == 0 ){
    nXX.release();
    status.release();
    CV_Assert( idxN_len != 0 );
    return;
  }
  Mat idxN(1, idxN_len, CV_32SC1); // 1x4698
  //idxNtest = cvCreateMat(1, idxN_len, CV_32SC1); // todo
  _nX.create(nXX.rows, idxN_len, CV_32SC1); // 10x4698
  Mat nX = _nX.getMat();
  int cols = 0;
  for (int j = 0; j < status.cols; j++){
    if (status.at<int>(0,j) == 1){
      idxN.at<int>(0,cols) = idxNNN.at<int>(0,j);
      //Coord_INT(idxNtest, 0, cols) = Coord_INT(idxNNN, 0, j); // todo
      for (int i = 0; i < nXX.rows; i++){
        nX.at<int>(i,cols) = nXX.at<int>(i,j);
      }
      cols++;
    }
  }
  //  idxN        = idxN(status==1); // bboxes far and with big variance
  //  nX          = nX(:,status==1);
  status.release();
  nXX.release();
  idxNNN.release();

  // Randomly select 'num_patches' bboxes and measure patches
  Mat temp(1, idxN_len, CV_32SC1);
  for (int j = 0; j < idxN_len; j++){
    temp.at<int>(0,j) = j;
  }
  Mat idx;
  randomValues<int>(temp, m_pApp->tldargs.n_par.num_patches, idx); // 1x100

  //  idx = randvalues(1:length(idxN), opt.n_par.num_patches);
  temp.release();

  Mat bb(4, idx.cols, CV_32FC1); // 4x100
  for (int j = 0; j < idx.cols; j++){
    for (int i = 0; i < 4; i++){
      bb.at<float>(i,j) = detector.grid.at<float>(i,idxN.at<int>(0,idx.at<int>(0,j)));
    }
  }
  //  bb  = detector.grid(:,idxN(idx));
  idxN.release();
  idx.release();
  getPattern(bb, _nEx); // nEx : 225x100
  bb.release();

}

// Splits negative data to training and validation set
void
tld::cpu::tracker::splitNegativeData ( InputArray _nX, InputArray _nEx, OutputArray _nX1, OutputArray _nX2, OutputArray _nEx1, OutputArray _nEx2 )
{
  Mat nX = _nX.getMat();
  Mat nEx = _nEx.getMat();

  CV_Assert( (nEx.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  int N = nX.cols;
  //N = size(nX,2);

  int* idx = new int[N];
  randomPerm<int>( N, idx );
  //  idx  = randperm(N);

  Mat nXX(nX.rows, nX.cols, CV_32SC1);
  for (int j=0; j<nXX.cols; j++){
    for (int i=0; i<nXX.rows; i++){
      nXX.at<int>(i,j) = nX.at<int>(i,idx[j]);
    }
  }
  //  nX   = nX(:,idx);

  _nX1.create(nXX.rows, nXX.cols/2, CV_32SC1);
  Mat nX1 = _nX1.getMat();

  for (int j=0; j<nX1.cols; j++){
    for (int i=0; i<nX1.rows; i++){
      nX1.at<int>(i,j) = nXX.at<int>(i,j);
    }
  }
  //  nX1  = nX(:,1:N/2);

  _nX2.create(nXX.rows, nXX.cols-nX1.cols, CV_32SC1);
  Mat nX2 = _nX2.getMat();
  for (int j=0; j<nX2.cols; j++){
    for (int i=0; i<nX2.rows; i++){
      nX2.at<int>(i,j) = nXX.at<int>(i,j+nX1.cols);
    }
  }
  //  nX2  = nX(:,N/2+1:end);

  delete []idx;
  nXX.release();

  N = nEx.cols;
  //  N    = size(nEx,2);

  int* idx1 = new int[N];
  randomPerm<int>( N, idx1);
  //  idx  = randperm(N);

  Mat nExx(nEx.rows, nEx.cols, CV_32FC1);
  for (int j=0; j<nExx.cols; j++){
    for (int i=0; i<nExx.rows; i++){
      nExx.at<float>(i,j) = nEx.at<float>(i,idx1[j]);
    }
  }
  //  nEx  = nEx(:,idx);

  _nEx1.create(nExx.rows, nExx.cols/2, CV_32FC1);
  Mat nEx1 = _nEx1.getMat();
  for (int j=0; j<nEx1.cols; j++){
    for (int i=0; i<nEx1.rows; i++){
      nEx1.at<float>(i,j) = nExx.at<float>(i,j);
    }
  }
  //  nEx1 = nEx(:,1:N/2);

  _nEx2.create(nExx.rows, nExx.cols-nEx1.cols, CV_32FC1);
  Mat nEx2 = _nEx2.getMat();
  for (int j=0; j<nEx2.cols; j++){
    for (int i=0; i<nEx2.rows; i++){
      nEx2.at<float>(i,j) = nExx.at<float>(i,j+nEx1.cols);
    }
  }
  //  nEx2 = nEx(:,N/2+1:end);

  delete []idx1;
  nExx.release();

}

void
tld::cpu::tracker::trainNearestNeighbor ( InputArray _pEx, InputArray _nEx )
{
  Mat pEx = _pEx.getMat();
  Mat nEx = _nEx.getMat();

  CV_Assert( (pEx.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( (nEx.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  int nP = pEx.cols; // get the number of positive example
  int nN = nEx.cols; // get the number of negative examples

  Mat xx(pEx.rows, pEx.cols+nEx.cols, CV_32FC1); // 225x101
  Mat yy(1, pEx.cols+nEx.cols, CV_32SC1);

  for (int j=0; j<xx.cols; j++){
    if (j<pEx.cols){
      yy.at<int>(0,j) = 1;
      for (int i=0; i<xx.rows; i++){ // ... <
        xx.at<float>(i,j) = pEx.at<float>(i,j);
      }
    } else {
      yy.at<int>(0,j) = 0;
      for (int i=0; i<xx.rows; i++){
        xx.at<float>(i,j) = nEx.at<float>(i,j-pEx.cols);
      }
    }
  }
  //x = [pEx,nEx];
  //y = [ones(1,nP), zeros(1,nN)];

  //Permutate the order of examples
  int* idx = new int[nP+nN];
  randomPerm<int>( nP+nN, idx);
  //idx = randperm(nP+nN);

  Mat x(xx.rows, xx.cols+1, CV_32FC1);
  Mat y(1, yy.cols+1, CV_32SC1);

  // always add the first positive patch as the first (important in initialization)
  for (int i=0; i<pEx.rows; i++){
    x.at<float>(i,0) = pEx.at<float>(i,0);
  }
  y.at<int>(0,0) = 1;
  for (int j=0; j<nP+nN; j++){
    y.at<int>(0,j+1) = yy.at<int>(0,idx[j]);
    for (int i=0; i<x.rows; i++){
      x.at<float>(i,j+1) = xx.at<float>(i,idx[j]);
    }
  }
  //if (~isempty(pEx)){
  // x   = [pEx(:,1) x(:,idx)];
  // y   = [1 y(:,idx)];
  //}

  xx.release();
  yy.release();
  delete []idx;

  trainer.pex.release();
  //  trainer.pex = 0;
  trainer.nex.release();
  //  trainer.nex = 0;

  PEX pex;
  int pexcnt = 0;
  NEX nex;
  int nexcnt = 0;
  float conf1;
  float conf2;
  int col = 0;
  for (int k = 0; k < 1; k++){ // Bootstrap
    for (int i = 0; i<y.cols; i++){
      if (!pex.empty()){
        trainer.pex.release();
        trainer.pex.create(x.rows, pex.size(), CV_32FC1);
        col = 0;
        for( PEX::iterator iter = pex.begin(); iter != pex.end(); ++iter ){
          for (int j = 0; j<x.rows; j++){ // row
            trainer.pex.at<float>(j,col) = (*iter).at<float>(j,0);
          }
          col++;
        }
      }
      if (!nex.empty()){
        trainer.nex.release();
        trainer.nex.create(x.rows, nex.size(), CV_32FC1);
        col = 0;
        for( NEX::iterator iter = nex.begin(); iter != nex.end(); ++iter ){
          for (int j = 0; j<x.rows; j++){ // row
            trainer.nex.at<float>(j,col) = (*iter).at<float>(j,0);
          }
          col++;
        }
      }

      // measure Relative similarity
      NearestNeighbor_1D (x, i, pex.empty(), nex.empty(), isin_NN1D, conf1, conf2);
      //[conf1,~,isin] = NearestNeighbor(x(:,i),tld);

      // Positive
      if (y.at<int>(0,i) == 1 && conf1 <= m_pApp->tldargs.model.thr_nn){ // 0.65
        Mat pexcell(x.rows, 1, CV_32FC1);
        for (int j = 0; j < x.rows; j++){
          pexcell.at<float>(j,0) = x.at<float>(j,i);
        }

        //if ( isnan(isin(2))){
        if (isin_NN1D.at<int>(1,0) == NANN){
          pex.push_back(pexcell);
          pexcnt++;
          //tld.pex = x(:,i);
          continue;
        }

        int test1 = pex.size();
        int test2 = isin_NN1D.at<int>(1,0);
        PEX::iterator iter = pex.begin();
        for (int j = 0; j < isin_NN1D.at<int>(1,0); j++){
          ++iter;
        }
        pex.insert (iter,pexcell);
        pexcnt++;
        //tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; // add to model
      }

      // Negative
      if (y.at<int>(0,i) == 0 && conf1 > 0.5){
        //if (y(i) == 0 && conf1 > 0.5){
        //tld.nex = [tld.nex x(:,i)];
        Mat nexcell(x.rows, 1, CV_32FC1);
        for (int j = 0; j < x.rows; j++){
          nexcell.at<float>(j,0) = x.at<float>(j,i);
        }
        nex.push_back(nexcell);
        nexcnt++;
      }

    }
  }
  trainer.pex.create(x.rows, pex.size(), CV_32FC1);
  col = 0;
  for( PEX::iterator iter = pex.begin(); iter != pex.end(); ++iter ){
    for (int i = 0; i<x.rows; i++){ // row
      trainer.pex.at<float>(i,col) = (*iter).at<float>(i,0);
    }
    col++;
  }

  for( PEX::iterator iter = pex.begin(); iter != pex.end(); ++iter ){
    (*iter).release();
    //    delete *iter;
  }

  trainer.nex.create(x.rows, nex.size(), CV_32FC1);
  col = 0;
  for( NEX::iterator iter = nex.begin(); iter != nex.end(); ++iter ){
    for (int i = 0; i<x.rows; i++){ // row
      trainer.nex.at<float>(i,col) = (*iter).at<float>(i,0);
    }
    col++;
  }

  for( NEX::iterator iter = nex.begin(); iter != nex.end(); ++iter ){
    (*iter).release();
    //    delete *iter;
  }

  x.release();
  y.release();

  return;
}

// function [conf1,conf2,isin] = tldNN(x,tld)
// 'conf1' ... full model (Relative Similarity)
// 'conf2' ... validated part of model (Conservative Similarity)
// 'isnin' ... inside positive ball, id positive ball, inside negative ball
void
tld::cpu::tracker::NearestNeighbor_1D (InputArray _patchPatt, const int colPatchPatt, const int pex_isempty, const int nex_isempty, OutputArray _isin, float &conf1, float &conf2)
{
  Mat patchPatt = _patchPatt.getMat();

  CV_Assert( (patchPatt.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( colPatchPatt < patchPatt.cols );

  Mat isin = _isin.getMat();
  isin.setTo(Scalar(NANN));
  //isin = nan(3,size(x,2));

  // IF positive examples in the model are not defined THEN everything is negative
  if (pex_isempty) {
    conf1 = 0.;
    conf2 = 0.;
    return;
  }
  //  if (isempty(tld.pex)){
  //  conf1 = zeros(1,size(x,2));
  //    conf2 = zeros(1,size(x,2));
  //    return;
  //  }

  // IF negative examples in the model are not defined THEN everything is positive
  if (nex_isempty) {
    conf1 = 1.;
    conf2 = 1.;
    return;
  }
  //  if (isempty(tld.nex)){
  //  conf1 = ones(1,size(x,2));
  //    conf2 = ones(1,size(x,2));
  //    return;
  //  }

  conf1 = NANN;
  //  conf1 = nan(1,size(x,2));
  conf2 = NANN;
  //  conf2 = nan(1,size(x,2));

  Mat temp(patchPatt.rows, 1, CV_32FC1);
  for (int i = 0; i < patchPatt.rows; i++){
    temp.at<float>(i,0) = patchPatt.at<float>(i,colPatchPatt);
  }

  // measure NCC to positive examples
  Mat nccP;
  distanceCol<float>(temp, trainer.pex, 1, nccP); // 1xpex-cols
  // nccP = distance(x(:,i),tld.pex,1);
  //CV_Assert( nccP->rows == 1 );
  // measure NCC to negative examples
  Mat nccN;
  distanceCol<float>(temp, trainer.nex, 1, nccN); // 1xnex-cols
  // nccN = distance(x(:,i),tld.nex,1);
  //CV_Assert( nccN->rows == 1 );

  temp.release();

  // set isin
  float maxnccP = -100000;
  for (int j=0; j<nccP.cols; j++){ // for pex
    // get the index of the maximal correlated positive patch
    if (nccP.at<float>(0,j) > maxnccP){
      maxnccP = nccP.at<float>(0,j);
      isin.at<int>(1,0) = j;
    }
    // IF the query patch is highly correlated with any positive patch in the model THEN it is considered to be one of them
    if (nccP.at<float>(0,j) > m_pApp->tldargs.model.ncc_thesame){
      isin.at<int>(0,0) = 1;
    }
  }
  // if (any(nccP > tld.model.ncc_thesame)){ isin(1,i) = 1;  }
  // [~,isin(2,i)] = max(nccP);
  float maxnccN = -100000;
  for (int j=0; j<nccN.cols; j++){
    if (nccN.at<float>(0,j) > maxnccN){
      maxnccN = nccN.at<float>(0,j);
    }
    // IF the query patch is highly correlated with any negative patch in the model THEN it is considered to be one of them
    if (nccN.at<float>(0,j) > m_pApp->tldargs.model.ncc_thesame){
      isin.at<int>(2,0) = 1;
    }
  }
  // if (any(nccN > tld.model.ncc_thesame)){ isin(3,i) = 1;  }

  // measure Relative Similarity
  float dN = 1 - maxnccN;
  float dP = 1 - maxnccP;
  conf1 = dN / (dN + dP);
  // dN = 1 - max(nccN);
  // dP = 1 - max(nccP);
  // conf1(i) = dN / (dN + dP);

  // measure Conservative Similarity
  int maxidx = ceil(m_pApp->tldargs.model.valid*trainer.pex.cols);
  int cnt = 0;
  float maxP = -10000;
  for (int j=0; j<nccP.cols; j++){
    if (nccP.at<float>(0,j) > maxP){
      maxP = nccP.at<float>(0,j);
      cnt++;
      if (cnt>=maxidx){
        j=nccP.cols;
        continue;
      }
    }
  }
  // maxP = max(nccP(1:ceil(tld.model.valid*size(tld.pex,2))));

  dP = 1 - maxP;
  conf2 = dN / (dN + dP);

  nccP.release();
  nccN.release();

  return;
}

// function [conf1,conf2,isin] = tldNN(x,tld)
// 'conf1' ... full model (Relative Similarity)
// 'conf2' ... validated part of model (Conservative Similarity)
// 'isnin' ... inside positive ball, id positive ball, inside negative ball
void
tld::cpu::tracker::NearestNeighbor (InputArray _x, OutputArray _isin, OutputArray _conf1, OutputArray _conf2)
{
  Mat x = _x.getMat();

  CV_Assert( (x.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  if (_isin.empty()){
    _isin.create(3, x.cols, CV_32SC1);
  }
  Mat isin = _isin.getMat();
  isin.setTo(Scalar(NANN));
  //isin = nan(3,size(x,2));

  Mat conf1;
  Mat conf2;

  if (trainer.pex.empty()) {
    _conf1.create(1, x.cols, CV_32FC1);
    conf1 = _conf1.getMat();
    conf1.setTo(Scalar(0.));
    _conf2.create(1, x.cols, CV_32FC1);
    conf2 = _conf2.getMat();
    conf2.setTo(Scalar(0.));
    CV_Assert( !trainer.pex.empty() );
    return;
  }
  //  if (isempty(tld.pex)){ // IF positive examples in the model are not defined THEN everything is negative
  //    conf1 = zeros(1,size(x,2));
  //  conf2 = zeros(1,size(x,2));
  //  return;
  //  }

  if (trainer.nex.empty()) {
    _conf1.create(1, x.cols, CV_32FC1);
    conf1 = _conf1.getMat();
    conf1.setTo(Scalar(1.));
    _conf2.create(1, x.cols, CV_32FC1);
    conf2 = _conf2.getMat();
    conf2.setTo(Scalar(1.));
    CV_Assert( !trainer.nex.empty() );
    return;
  }
  //  if (isempty(tld.nex)){ // IF negative examples in the model are not defined THEN everything is positive
  //    conf1 = ones(1,size(x,2));
  //  conf2 = ones(1,size(x,2));
  //  return;
  //  }

  _conf1.create(1, x.cols, CV_32FC1);
  conf1 = _conf1.getMat();
  conf1.setTo(Scalar(NANN));
  //  conf1 = nan(1,size(x,2));
  _conf2.create(1, x.cols, CV_32FC1);
  conf2 = _conf2.getMat();
  conf2.setTo(Scalar(NANN));
  //  conf2 = nan(1,size(x,2));

  Mat temp(x.rows, 1, CV_32FC1);

  for (int ii = 0; ii < x.cols; ii++){ // for every patch that is tested
    for (int i = 0; i < x.rows; i++){
      temp.at<float>(i,0) = x.at<float>(i,ii);
    }

    Mat nccP;
    distanceCol<float>(temp, trainer.pex, 1, nccP); // 1x1
    // nccP = distance(x(:,i),tld.pex,1); // measure NCC to positive examples
    CV_Assert( nccP.rows == 1 );
    Mat nccN;
    distanceCol<float>(temp, trainer.nex, 1, nccN); // 1x1
    // nccN = distance(x(:,i),tld.nex,1); // measure NCC to negative examples
    CV_Assert( nccN.rows == 1 );

    // set isin
    float maxnccP = -10000;
    for (int j=0; j<nccP.cols; j++){ // for pex
      // get the index of the maximal correlated positive patch
      if (nccP.at<float>(0,j) > maxnccP){
        maxnccP = nccP.at<float>(0,j);
        isin.at<int>(1,ii) = j;
      }
      // IF the query patch is highly correlated with any positive patch in the model THEN it is considered to be one of them
      if (nccP.at<float>(0,j) > m_pApp->tldargs.model.ncc_thesame){
        isin.at<int>(0,ii) = 1;
      }
    }
    // if (any(nccP > tld.model.ncc_thesame)){ isin(1,i) = 1;  }
    // [~,isin(2,i)] = max(nccP);
    float maxnccN = -10000;
    for (int j=0; j<nccN.cols; j++){
      if (nccN.at<float>(0,j) > maxnccN){
        maxnccN = nccN.at<float>(0,j);
      }
      // IF the query patch is highly correlated with any negative patch in the model THEN it is considered to be one of them
      if (nccN.at<float>(0,j) > m_pApp->tldargs.model.ncc_thesame){
        isin.at<int>(2,ii) = 1;
      }
    }
    // if (any(nccN > tld.model.ncc_thesame)){ isin(3,i) = 1;  }

    // measure Relative Similarity
    float dN = 1 - maxnccN;
    float dP = 1 - maxnccP;
    conf1.at<float>(0,ii) = dN / (dN + dP);
    // dN = 1 - max(nccN);
    // dP = 1 - max(nccP);
    // conf1(i) = dN / (dN + dP);

    // measure Conservative Similarity
    int maxidx = ceil(m_pApp->tldargs.model.valid*trainer.pex.cols);
    int cnt = 0;
    float maxP = -100000;
    for (int j=0; j<nccP.cols; j++){
      if (nccP.at<float>(0,j) > maxP){
        maxP = nccP.at<float>(0,j);
        cnt++;
        if (cnt>=maxidx){
          j=nccP.cols;
          continue;
        }
      }
    }
    // maxP = max(nccP(1:ceil(tld.model.valid*size(tld.pex,2))));

    dP = 1 - maxP;
    conf2.at<float>(0,ii) = dN / (dN + dP);

    nccP.release();
    nccN.release();
  }
  temp.release();

}

void
tld::cpu::tracker::processFrame ( )
{
#ifdef PROFILING_ON_
  PROFILINING_START_TIME( );
#endif
  // 190.0 ms
  // TRACKER  ----------------------------------------------------------------
  tracking ( ); // frame-to-frame tracking (MedianFlow)
#ifdef PROFILING_ON_
  PROFILINING_STOP_TIME( ",TLD,tracking," );
#endif

#ifdef PROFILING_ON_
  PROFILINING_START_TIME( );
#endif
  // 1250.0 ms
  // DETECTOR ----------------------------------------------------------------
  detection ( ); // detect appearances by cascaded detector (variance filter -> ensemble classifier -> nearest neighbor)
#ifdef PROFILING_ON_
  PROFILINING_STOP_TIME( ",TLD,detection," );
#endif
#ifdef FERN_OPENCV_ON_
  ObjectDetector();
#endif

  // INTEGRATOR --------------------------------------------------------------
  if (TRDT.TR){ // if tracker is defined
    // copy tracker's result
    for (int i=0; i<4; i++){
      TRDT.curBBox.at<float>(i,0) = TRDT.tBB.at<float>(i,0);
    }
    //  tld.bb(:,I)  = tBB;

    TRDT.curConf = TRDT.tConf;
    //  tld.conf(I)  = tConf;

    TRDT.curSize = 1;
    //  tld.size(I)  = 1;

    TRDT.curValid = TRDT.tValid;
    //  tld.valid(I) = tValid;

    if (TRDT.DT){ // if detections are also defined
      Mat cBB; // 4x1
      Mat cConf; // 1x1
      Mat cSize; // 1x1
      bboxClusterConfidence<float>(TRDT.dBB, TRDT.dConf, TRDT.DTLen, cBB, cConf, cSize); // cluster detections
      //[cBB,cConf,cSize] = bboxClusterConfidence(dBB,dConf);

      // get indexes of all clusters that are far from tracker and are more confident than the tracker
      Mat overlap;
      bboxOverlap<float>(TRDT.curBBox, cBB, overlap);
      Mat id_(1, overlap.cols, CV_32SC1);
      int* id = (int*)id_.data;

      int id_len = 0;
      for (int j=0; j < overlap.cols; j++){
        if (overlap.at<float>(0,j) <= 0.5 && cConf.at<float>(0,j) >= TRDT.curConf){
          id[id_len] = j;
          id_len++;
        }
      }
      //id = bboxOverlap(tld.bb(:,I),cBB) < 0.5 & cConf > tld.conf(I);
      Mat overlap1_;
      Mat idTr_;
      if (id_len == 1){ // if there is ONE such a cluster, re-initialize the tracker
        for (int j=0; j < 4; j++){
          TRDT.curBBox.at<float>(j,0) = cBB.at<float>(j,id[0]);
        }
        TRDT.curConf  = cConf.at<float>(0,id[0]);
        TRDT.curSize  = cSize.at<int>(0,id[0]);
        TRDT.curValid = 1;//todo 0 -> 1
      } else {// otherwise adjust the tracker's trajectory
        bboxOverlap<float>(TRDT.tBB, 1, detector.dt.bb, detector.dt.num_dt, overlap1_);
        float* overlap1 = (float*)overlap1_.data;
        idTr_.create(1, overlap1_.cols, CV_32SC1);
        int* idTr = (int*)idTr_.data;
        int idTr_len = 0;
        // get indexes of close detections
        for (int j=0; j < overlap1_.cols; j++){
          if (overlap1[j] > 0.7){
            idTr[idTr_len] = j;
            idTr_len++;
          }
        }
        //idTr = bboxOverlap(tBB,tld.dt{I}.bb) > 0.7;

        // weighted average trackers trajectory with the close detections
        float mean[4]; mean[0] = 0.; mean[1] = 0.; mean[2] = 0.; mean[3] = 0.;
        if (idTr_len>0){
          for (int i=0; i < idTr_len; i++){
            for (int j=0; j < 4; j++){
              mean[j] += detector.dt.bb.at<float>(j,idTr[i]);
            }
          }
        }
        for (int j=0; j < 4; j++){
          mean[j] += TRDT.tBB.at<float>(j,0)*10.;
        }
        for (int j=0; j < 4; j++){
          TRDT.curBBox.at<float>(j,0) = (float)(mean[j]/(idTr_len+10.));
        }
        //tld.bb(:,I) = mean([repmat(tBB,1,10) tld.dt{I}.bb(:,idTr)],2);

      }
      overlap1_.release();
      idTr_.release();
      overlap.release();
      id_.release();
      cBB.release();
      cConf.release();
      cSize.release();
    }
  } else {// if tracker is not defined
    if (TRDT.DT){ // and detector is defined
      Mat cBB;
      Mat cConf;
      Mat cSize;
      bboxClusterConfidence<float>(TRDT.dBB, TRDT.dConf, TRDT.DTLen, cBB, cConf, cSize); // cluster detections
      //[cBB,cConf,cSize] = bboxClusterConfidence(dBB,dConf);

      if (cConf.cols == 1){ // and if there is just a single cluster, re-initialize the tracker
        cBB.copyTo(TRDT.curBBox);
        TRDT.curConf  = cConf.at<float>(0,0);
        TRDT.curSize  = cSize.at<int>(0,0);
        TRDT.curValid = 1;//todo 0 -> 1
      }
      cBB.release();
      cConf.release();
      cSize.release();
    }
  }

  // LEARNING ----------------------------------------------------------------

  if (m_pApp->tldargs.control.update_detector && TRDT.curValid == 1) {
    //   learning();
  }

  //  if (tld.control.drop_img && I > 2) {tld.img{I-1} = {}; } // forget previous image

}

// Estimates motion of bounding box BB1 from frame I to frame J
void
tld::cpu::tracker::tracking (  )
{
  // initialize output variables
  TRDT.tConf   = 0; // confidence of prediction
  TRDT.tValid  = 0; // is the predicted bounding box valid? if yes, learning will take place ...

  if (TRDT.curBBox.empty() || !bboxIsDef<float>(TRDT.curBBox)){
    TRDT.TR = 0;
    return;
  } // exit function if BB1 is not defined

  // estimate BB2
  //TRDT.xFI 2x100
  bboxPointsCustom<float>(TRDT.curBBox, TRDT.numM, TRDT.numN, 5, TRDT.xFII); // generate 10x10 grid of points within BB1 with margin 5 px

//  saveMatrix<float>(BB1,false);
//  saveMatrix<float>(TRDT.xFII,false);

  //TRDT.xFJJ 4x100
  lucaskanade->lk(m_pApp->prvGrayImg, m_pApp->curGrayImg, TRDT.xFII, TRDT.xFII, TRDT.xFJJ); // track all points by Lucas-Kanade tracker from frame I to frame J, estimate Forward-Backward error, and NCC for each point

//  saveMatrix<float>(xFJ,false);

  // rwocol = 0 -> row = 2
  float medFB = (float)median1D<float>(TRDT.xFJJ, 0, 2); // get median of Forward-Backward error
  //  medFB  = median2(xFJ(3,:));

  // rwocol = 0 -> row = 3
  float medNCC = (float)median1D<float>(TRDT.xFJJ, 0, 3); // get median for NCC
  //  medNCC = median2(xFJ(4,:));

  // get indexes of reliable points
  TRDT.numMN = 0;
  for (int i=0; i<TRDT.xFJJ.cols; i++){
    if (TRDT.xFJJ.at<float>(2,i) <= medFB && TRDT.xFJJ.at<float>(3,i) >= medNCC){
      TRDT.idxF[TRDT.numMN] = i;
      TRDT.numMN++;
    }
  }
  //  idxF   = xFJ(3,:) <= medFB & xFJ(4,:)>= medNCC;

  if (TRDT.numMN == 0){
    TRDT.TR = 0;
    return;
  }
//  saveMatrix<int>(TRDT.idxF_,false);

  for (int i=0; i<TRDT.numMN; i++){
    for (int j=0; j<2; j++){
      TRDT.xFI.at<float>(j,i) = TRDT.xFII.at<float>(j,TRDT.idxF[i]);
      TRDT.xFJ.at<float>(j,i) = TRDT.xFJJ.at<float>(j,TRDT.idxF[i]);
    }
  }
//  saveMatrix<float>(TRDT.xFI,false);
//  saveMatrix<float>(TRDT.xFJ,false);
  // BB2 : 4x1
  // estimate BB2 using the reliable points only
  bboxPredict<float>(TRDT.curBBox, TRDT.xFI, TRDT.xFJ, TRDT.numMN, TRDT.tBB);
  //  BB2    = bboxPredict(BB1,xFI(:,idxF),xFJ(1:2,idxF));
  //  tld.xFJ = xFJ(:,idxF); // save selected points (only for display purposes)

//  saveMatrix<float>(TRDT.tBB,false);

  // detect failures
  if ((!bboxIsDef<float>(TRDT.tBB) || bboxIsOut<float>(TRDT.tBB, imgsize)) ||
      (m_pApp->tldargs.control.maxbbox > 0 && medFB > 10)
  ){
    TRDT.TR = 0;
    return;
  } // bounding box out of image or too unstable predictions

  TRDT.TR = 1;

  // estimate confidence and validity
  //patchPatt1D 225x1
  // sample patch in current image
  getPattern_1D ( TRDT.tBB, 0, patchPatt1D );
  float conf1 = 0;
  float conf2 = 0;
  // estimate its Conservative Similarity (considering 50% of positive patches only)
  NearestNeighbor_1D (patchPatt1D, 0, 0, 0, isin_NN1D, conf1, conf2);
  //  [~,Conf] = NearestNeighbor(patchJ,tld);

  TRDT.tConf = conf1;

  // Validity
  TRDT.tValid    = TRDT.prvValid; // copy validity from previous frame
  if (conf1 >= m_pApp->tldargs.model.thr_nn_valid){
    TRDT.tValid = 1;
  } // tracker is inside the 'core'
}

// scanns the image(I) with a sliding window, returns a list of bounding
// boxes and their confidences that match the object description
void
tld::cpu::tracker::detection ( )
{
  TRDT.DT = 0;

#ifdef PROFILING_ON_
  PROFILINING_START_TIME( );
#endif
  // evaluates Ensemble Classifier: saves sum of posteriors to 'tld.tmp.conf', saves measured codes to 'tld.tmp.patt',
  // does not considers patches with variance < tmd.var
  fern->detect( m_pApp->curGrayImg, m_pApp->curBlurImg, m_pApp->tldargs.control.maxbbox, trainer.var, detector.tmp.conf, detector.tmp.patt );
#ifdef PROFILING_ON_
  PROFILINING_STOP_TIME( ",TLD,fern->detect," );
#endif

  // get indexes of bounding boxes that passed through the Ensemble Classifier
  int idx_dt_len = 0;
  Mat idx_dt_(1, detector.tmp.conf.cols, CV_32SC1);
  int* idx_dt = (int*)idx_dt_.data;
  for (int j = 0; j < detector.tmp.conf.cols; j++){
    if (detector.tmp.conf.at<float>(0,j) >= m_pApp->tldargs.model.num_trees*m_pApp->tldargs.model.thr_fern){ // 10*0.5
      idx_dt[idx_dt_len] = j;
      idx_dt_len++;
    }
  }
  //  idx_dt = find(detector.tmp.conf > tld.model.num_trees*tld.model.thr_fern);

  //  saveMatrix(detector.tmp.conf);

  // speedup: if there are more than 100 detections, pict 100 of the most confident only
  if (idx_dt_len > DT_MAX_BB){
    for (int i = 0; i < idx_dt_len-1; i++){
      for (int j = i+1; j < idx_dt_len; j++){
        if (detector.tmp.conf.at<float>(0,idx_dt[j]) > detector.tmp.conf.at<float>(0,idx_dt[i])){
          int temp = idx_dt[i];
          idx_dt[i] = idx_dt[j];
          idx_dt[j] = temp;
        }
      }
    }
    //[~,sIdx] = sort(tld.tmp.conf(idx_dt),'descend');
    idx_dt_len = DT_MAX_BB;
    //    idx_dt = idx_dt(sIdx(1:100));
  }

  detector.dt.num_dt = idx_dt_len; // get the number detected bounding boxes so-far
  TRDT.DTNum = detector.dt.num_dt;
  // if nothing detected, return
  if (detector.dt.num_dt == 0) {
    //  tld.dt{I} = dt;
    idx_dt_.release();
    TRDT.DT = 0;
    TRDT.DTLen = 0;
    return;
  }

  // initialize detection structure
  for (int i = 0; i<detector.dt.num_dt; i++){
    for (int j = 0; j<4; j++){ // bounding boxes
      detector.dt.bb.at<float>(j,i) = detector.grid.at<float>(j,idx_dt[i]);
    }
    //dt.bb     = tld.grid(1:4,idx_dt);

    for (int j = 0; j<detector.dt.patt.rows; j++){ // corresponding codes of the Ensemble Classifier
      detector.dt.patt.at<int>(j,i) = detector.tmp.patt.at<int>(j,idx_dt[i]);
    }
    //dt.patt   = tld.tmp.patt(:,idx_dt);

    detector.dt.idx.at<int>(0,i) = idx_dt[i]; // indexes of detected bounding boxes within the scanning grid
    //dt.idx    = find(idx_dt);
  }

  idx_dt_.release();

  for (int i = 0; i<detector.dt.num_dt; i++){ // for every remaining detection
    getPattern_1D(detector.dt.bb, i, patchPatt1D); // measure patch

    float conf1 = 0;
    float conf2 = 0;
    NearestNeighbor_1D (patchPatt1D, 0, 0, 0, isin_NN1D, conf1, conf2); // evaluate nearest neighbor classifier
    //[conf1, conf2, isin] = NearestNeighbor(ex,tld);

    //fill detection structure
    detector.dt.conf1.at<float>(0,i) = conf1;// Relative Similarity (for final nearest neighbor classifier)
    detector.dt.conf2.at<float>(0,i) = conf2;// Conservative Similarity (for integration with tracker)

    //dt.isin(:,i)  = isin;
    for (int j = 0; j < detector.dt.isin.rows; j++){
      detector.dt.isin.at<int>(j,i) = isin_NN1D.at<int>(j,0);// detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
    }

    for (int j = 0; j < detector.dt.patch.rows; j++){
      detector.dt.patch.at<float>(j,i) = patchPatt1D.at<float>(j,0);// Corresponding patches
    }
    //dt.patch(:,i) = ex;
  }

  // get all indexes that made it through the nearest neighbor
  TRDT.DTLen = 0;
  Mat idx(1, detector.dt.num_dt, CV_32SC1);
  for (int j = 0; j < detector.dt.num_dt; j++){
    if (detector.dt.conf1.at<float>(0,j) >= m_pApp->tldargs.model.thr_nn){
      idx.at<int>(0,TRDT.DTLen) = j;
      TRDT.DTLen++;
    }
  }
  //  idx = dt.conf1 > tld.model.thr_nn;

  if (TRDT.DTLen > 0){
    // output
    for (int i = 0; i<TRDT.DTLen; i++){
      for (int j = 0; j<4; j++){ // bounding boxes
        TRDT.dBB.at<float>(j,i) = detector.dt.bb.at<float>(j,idx.at<int>(0,i));
      }
      TRDT.dConf.at<float>(0,i) = detector.dt.conf2.at<float>(0,idx.at<int>(0,i)); // conservative confidences
    }
    //  BB    = dt.bb(:,idx); // bounding boxes
    //  Conf  = dt.conf2(:,idx); // conservative confidences
    //  tld.dt{I} = dt; // save the whole detection structure
    TRDT.DT = 1;
  }
  idx.release();
}

void
tld::cpu::tracker::learning ( )
{
  // Check consistency -------------------------------------------------------
  getPattern_1D ( TRDT.curBBox, 0, patchPatt1D ); // get current patch

  float pConf1 = 0;
  float pConf2 = 0;
  NearestNeighbor_1D (patchPatt1D, 0, 0, 0, isin_NN1D, pConf1, pConf2); // measure similarity to model
  //  [pConf1,~,pIsin] = NearestNeighbor(pPatt,tld);

  if (pConf1 < 0.5){ // too fast change of appearance
    cout << "\nLearning:Too fast change of appearance.\n"
        << endl;
    TRDT.curValid = 0;
    return;
  }
  float var = (float)variance1D<float>(patchPatt1D);
  if (var < trainer.var){ // too low variance of the patch
    cout << "\nLearning:Too low variance of the patch.\n"
        << endl;
    TRDT.curValid = 0;
    return;
  }
  if (isin_NN1D.at<int>(2,0) == 1){ // patch is in negative data
    cout << "\nLearning:Patch is in negative data.\n"
        << endl;
    TRDT.curValid = 0;
    return;
  }

  // Update ------------------------------------------------------------------

  // generate positive data
  // measure overlap of the current bounding box with the bounding boxes on the grid
  Mat overlap; // 1x28052
  bboxOverlap<float>(TRDT.curBBox, detector.grid, overlap);

  // generate positive examples from all bounding boxes that are highly overlapping with current bounding box
  Mat pX; //
  // pEx : patchPatt1D
  Mat bbP; // 4x1
  generatePositiveData(overlap, 1, pX, patchPatt1D, bbP);
  //  [pX,pEx] = generatePositiveData(tld,overlap,img,tld.p_par_update);

  if (pX.empty()){
    overlap.release();
    return;
  }
  //  pY       = ones(1,size(pX,2)); // labels of the positive patches
  bbP.release();

  // generate negative data
  int idx_len = 0;
  Mat idx(1, overlap.cols, CV_32SC1);
  // get indexes of negative bounding boxes on the grid (bounding boxes on the grid that are far from current bounding box and which confidence was larger than 0)
  for (int j = 0; j < idx.cols; j++){
    if (overlap.at<float>(0,j) < m_pApp->tldargs.n_par.overlap && detector.tmp.conf.at<float>(0,j) >= 1){
      idx.at<int>(0,idx_len) = j;
      idx_len++;
    }
  }
  //  idx      = overlap < tld.n_par.overlap & tld.tmp.conf >= 1;

  overlap.release();

  if (idx_len>0){
    CV_Assert( pX.rows == detector.tmp.patt.rows );
    Mat X(pX.rows, pX.cols+idx_len, CV_32SC1); //
    for (int j=0; j<X.cols; j++){
      if (j<pX.cols){
        for (int i=0; i<X.rows; i++){
          X.at<int>(i,j) = pX.at<int>(i,j);
        }
      }else{
        for (int i=0; i<X.rows; i++){
          X.at<int>(i,j) = detector.tmp.patt.at<int>(i,idx.at<int>(0,j-pX.cols));
        }
      }
    }

    Mat Y(1, pX.cols+idx_len, CV_32SC1);
    for (int j=0; j<Y.cols; j++){
      if (j<pX.cols){
        Y.at<int>(0,j) = 1;
      }else{
        Y.at<int>(0,j) = 0;
      }
    }

    CV_Assert( Y.cols == X.cols );

    fern->update( X, Y, m_pApp->tldargs.model.thr_fern, 2 ); // update the Ensemble Classifier (reuses the computation made by detector)
    //  fern(2,[pX tld.tmp.patt(:,idx)],[pY zeros(1,sum(idx))],tld.model.thr_fern,2);

    X.release();
    Y.release();
  }
  pX.release();
  idx.release();

  if (detector.dt.num_dt > 0){
    // measure overlap of the current bounding box with detections
    bboxOverlap<float>(TRDT.curBBox, 1, detector.dt.bb, detector.dt.num_dt, overlap);
    //overlap  = bb_overlap(bb,tld.dt{I}.bb);

    int idxx_len = 0;
    Mat idxx(1, overlap.cols, CV_32SC1);
    // get negative patches that are far from current bounding box
    for (int j = 0; j < idxx.cols; j++){
      if (overlap.at<float>(0,j) < m_pApp->tldargs.n_par.overlap){
        idxx.at<int>(0,idxx_len) = j;
        idxx_len++;
      }
    }

    if (idxx_len > 0){
      Mat nEx(detector.dt.patch.rows, idxx_len, CV_32FC1);
      for (int i = 0; i < idxx_len; i++){
        for (int j = 0; j < detector.dt.patch.rows; j++){
          nEx.at<float>(j,i) = detector.dt.patch.at<float>(j,idxx.at<int>(0,i));
        }
      }
      //  nEx      = tld.dt{I}.patch(:,overlap < opt.n_par.overlap);

      trainNearestNeighbor(patchPatt1D, nEx); // update nearest neighbor
      //  tld = trainNearestNeighbor(pEx,nEx,tld);

      nEx.release();
    }
    idxx.release();
    overlap.release();
  }
}

#ifdef FERN_OPENCV_ON_
void
tld::cpu::tracker::ObjectDetectorTrain( )
{
  Size patchSize(SINGLETRACKEROPT_PATCHSIZE_X, SINGLETRACKEROPT_PATCHSIZE_Y);
  ldetector.radius = 7;
  ldetector.threshold = 20;
  ldetector.nOctaves = 2;
  ldetector.nViews = 2000;
  ldetector.baseFeatureSize = patchSize.width;
  ldetector.clusteringDistance = 2;
//  LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);

  TRDT.DTFound =false;

  TRDT.object.release();
  imgPatch<float>(m_pApp->curGrayImg, TRDT.object, TRDT.curBBox, 0);

//  const char* object_filename = "bbox.png";

  ldetector.setVerbose(true);

  vector<Mat> objpyr;
  blurKSize = 3; // 3
  sigma = 0;
  GaussianBlur(TRDT.object, TRDT.object, Size(blurKSize, blurKSize), sigma, sigma);
  buildPyramid(TRDT.object, objpyr, ldetector.nOctaves-1);

  vector<KeyPoint> objKeypoints_;
  PatchGenerator gen(0,256,5,true,0.8,1.2,-CV_PI/2,CV_PI/2,-CV_PI/2,CV_PI/2);

//  string model_filename = format("%s_model.xml.gz", object_filename);
//  printf("Trying to load %s ...\n", model_filename.c_str());
//  FileStorage fs(model_filename, FileStorage::READ);
//  if( fs.isOpened() )
//  {
//    detector_.read(fs.getFirstTopLevelNode());
//    printf("Successfully loaded %s.\n", model_filename.c_str());
//  }
//  else
//  {
//    printf("The file not found and can not be read. Let's train the model.\n");
    std::cout << "Fern: train the model:\n" << "Step 1. Finding the robust keypoints ..." << "\n";
    ldetector.setVerbose(true);
    ldetector.getMostStable2D(TRDT.object, objKeypoints_, 100, gen);
    std::cout << "Done.\nStep 2. Training ferns-based planar object detector ...\n";
    detector_.setVerbose(true);

    detector_.train(objpyr, objKeypoints_, patchSize.width, 100, 11, 10000, ldetector, gen);
//    printf("Done.\nStep 3. Saving the model to %s ...\n", model_filename.c_str());
//    if( fs.open(model_filename, FileStorage::WRITE) )
//      detector_.write(fs, "ferns_model");
//  }
//  fs.release();

}

void
tld::cpu::tracker::ObjectDetector()
{
  double imgscale = 1;
  Mat image;

  resize(m_pApp->curGrayImg, image, Size(), 1./imgscale, 1./imgscale, INTER_CUBIC);

  Size patchSize(SINGLETRACKEROPT_PATCHSIZE_X, SINGLETRACKEROPT_PATCHSIZE_Y);
//  LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);
//  ldetector.setVerbose(true);

  vector<Mat> imgpyr;
  GaussianBlur(image, image, Size(blurKSize, blurKSize), sigma, sigma);
  buildPyramid(image, imgpyr, ldetector.nOctaves-1);

//  printf("Now find the keypoints in the image, try recognize them and compute the homography matrix\n");
  Mat H;

//  double t = (double)getTickCount();
  TRDT.objKeypoints = detector_.getModelPoints();
  ldetector(imgpyr, TRDT.imgKeypoints, 300);

//  std::cout << "Object keypoints: " << TRDT.objKeypoints.size() << "\n";
//  std::cout << "Image keypoints: " << TRDT.imgKeypoints.size() << "\n";
  TRDT.DTFound = detector_(imgpyr, TRDT.imgKeypoints, H, TRDT.dst_corners, &TRDT.pairs);
//  t = (double)getTickCount() - t;
//  printf("%gms\n", t*1000/getTickFrequency());

  return;
}
#endif

#endif
