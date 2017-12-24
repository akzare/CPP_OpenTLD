/***********************************************************************
  $FILENAME    : BoundingBox.hpp

  $TITLE       : Bounding box template functions

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Includes utility template funtions for constructing 
                 the bounding boxes for the tracked object

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef BOUNDING_BOX_H_
#define BOUNDING_BOX_H_

#include <opencv2/core/core.hpp>      // Basic OpenCV structures

#include <list>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <fstream>

#include "Constants.h"

#include "Distance.hpp"
#include "Statistics.hpp"
#include "Linkage.hpp"

using namespace std;
using namespace cv;

typedef std::list<Mat> BBOXES;
typedef std::list<Mat> SCA;

template<class TEMPL>
void bboxHeight( InputArray _bb, OutputArray _bbH )
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  _bbH.create( 1, bb.cols , bb.type() );
  Mat bbH = _bbH.getMat();

  for(int i=0; i<bbH.cols; i++){
    bbH.at<TEMPL>(0,i) = bb.at<TEMPL>(3,i)-bb.at<TEMPL>(1,i)+1.;
    // bb(3,:)-bb(1,:)+1;
  }
}

template<class TEMPL>
TEMPL bboxHeight1D( InputArray _bb )
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb.cols == 1 );
  return (bb.at<TEMPL>(3,0)-bb.at<TEMPL>(1,0)+1.);
}

template<class TEMPL>
void bboxWidth( InputArray _bb, OutputArray _bbW )
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  _bbW.create( 1, bb.cols, bb.type() );
  Mat bbW = _bbW.getMat();

  for(int i=0; i<bbW.cols; i++){
    bbW.at<TEMPL>(0,i) = bb.at<TEMPL>(2,i)-bb.at<TEMPL>(0,i)+1.;
    // bb(2,:)-bb(0,:)+1;
  }
}

template<class TEMPL>
TEMPL bboxWidth1D( InputArray _bb )
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb.cols == 1 );
  return (bb.at<TEMPL>(2,0)-bb.at<TEMPL>(0,0)+1.);
}

template<class TEMPL>
void bboxSize( InputArray _bb, OutputArray _size )
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  _size.create( 2, bb.cols, bb.type() );
  Mat bbSize = _size.getMat();

  for(int i=0; i<bbSize.cols; i++){
    bbSize.at<TEMPL>(0,i) = bb.at<TEMPL>(3,i)-bb.at<TEMPL>(1,i)+1.;
    bbSize.at<TEMPL>(1,i) = bb.at<TEMPL>(2,i)-bb.at<TEMPL>(0,i)+1.;
    // s = [bb(3,:)-bb(1,:)+1; bb(2,:)-bb(0,:)+1];
  }
}

template<class TEMPL>
void bboxCenter( InputArray _bb, OutputArray _center )
{
  Mat bb = _bb.getMat();

  CV_Assert( bb.cols == 1 );
//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32SC1 );

  _center.create( 2, 1, bb.type() );
  Mat center = _center.getMat();

  center.at<TEMPL>(0,0) = (bb.at<TEMPL>(0,0)+bb.at<TEMPL>(2,0))/2;
  center.at<TEMPL>(1,0) = (bb.at<TEMPL>(1,0)+bb.at<TEMPL>(3,0))/2;
  // center = 0.5 * [bb(0,:)+bb(2,:); bb(1,:)+bb(3,:)];
}

template<class TEMPL>
float bboxOverlap1D(InputArray _bb1, const int col1, InputArray _bb2, const int col2)
{
  Mat bb1 = _bb1.getMat();
  Mat bb2 = _bb2.getMat();

//  CV_Assert( (bb1.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb1.cols > col1 && bb2.cols > col2 );

  if (bb1.at<TEMPL>(0,col1) > bb2.at<TEMPL>(2,col2)) { return 0.; }
  //  if (bb1[0] > bb2[2]) { return 0.0; }
  if (bb1.at<TEMPL>(1,col1) > bb2.at<TEMPL>(3,col2)) { return 0.; }
  //  if (bb1[1] > bb2[3]) { return 0.0; }
  if (bb1.at<TEMPL>(2,col1) < bb2.at<TEMPL>(0,col2)) { return 0.; }
  //  if (bb1[2] < bb2[0]) { return 0.0; }
  if (bb1.at<TEMPL>(3,col1) < bb2.at<TEMPL>(1,col2)) { return 0.; }
  //  if (bb1[3] < bb2[1]) { return 0.0; }

  float colInt =  (float)(min(bb1.at<TEMPL>(2,col1), bb2.at<TEMPL>(2,col2))) - (float)(max(bb1.at<TEMPL>(0,col1), bb2.at<TEMPL>(0,col2))) + 1.;
  //  int colInt =  min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]) + 1;
  float rowInt =  (float)(min(bb1.at<TEMPL>(3,col1), bb2.at<TEMPL>(3,col2))) - (float)(max(bb1.at<TEMPL>(1,col1), bb2.at<TEMPL>(1,col2))) + 1.;
  //  int rowInt =  min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]) + 1;

  float intersection = (float)(colInt * rowInt);
  float area1 = (float)((bb1.at<TEMPL>(2,col1)-bb1.at<TEMPL>(0,col1)+1.)*(bb1.at<TEMPL>(3,col1)-bb1.at<TEMPL>(1,col1)+1.));
  //  int area1 = (bb1[2]-bb1[0]+1)*(bb1[3]-bb1[1]+1);
  float area2 = (float)((bb2.at<TEMPL>(2,col2)-bb2.at<TEMPL>(0,col2)+1.)*(bb2.at<TEMPL>(3,col2)-bb2.at<TEMPL>(1,col2)+1.));
  //  int area2 = (bb2[2]-bb2[0]+1)*(bb2[3]-bb2[1]+1);
  return (float)((intersection*1.0) / (area1 + area2 - intersection));
}

// bb_overlap(bb), overlap matrix
template<class TEMPL>
void bboxOverlap(InputArray _bb, OutputArray _overlap)
{
  Mat bb_ = _bb.getMat();

//  CV_Assert( (bb_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  // Output
  _overlap.create( 1, (bb_.cols)*(bb_.cols-1)/2, CV_32FC1 );
  Mat overlap = _overlap.getMat();

  int col = 0;
  for (int i = 0; i < bb_.cols-1; i++){
    for (int j = i+1; j < bb_.cols; j++){
      CV_Assert( col < overlap.cols );
      overlap.at<float>(0,col) = bboxOverlap1D<TEMPL>(bb_, i, bb_, j);
      //*overlap++ = bb_overlap_(bb1 + bb1_->rows*i, bb2 + bb2_->rows*j);
      col++;
    }
  }
}

// bb_overlap(bb1,bb2), overlap matrix
template<class TEMPL>
void bboxOverlap(InputArray _bb1, InputArray _bb2, OutputArray _overlap)
{
  Mat bb1_ = _bb1.getMat();
  Mat bb2_ = _bb2.getMat();

//  CV_Assert( (bb1_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (bb2_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  // Output
  _overlap.create( bb1_.cols, bb2_.cols, CV_32FC1 );
  Mat overlap = _overlap.getMat();

  for (int i = 0; i < bb1_.cols; i++){
    for (int j = 0; j < bb2_.cols; j++){
      overlap.at<float>(i,j) = bboxOverlap1D<TEMPL>(bb1_, i, bb2_, j);
      //*overlap++ = bb_overlap_(bb1 + bb1_->rows*i, bb2 + bb2_->rows*j);
    }
  }
}

template<class TEMPL>
void bboxOverlap(InputArray _bb1, const int size1, InputArray _bb2, const int size2, OutputArray _overlap)
{
  Mat bb1_ = _bb1.getMat();
  Mat bb2_ = _bb2.getMat();

//  CV_Assert( (bb1_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (bb2_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb1_.cols >= size1 && bb2_.cols >= size2 );
  // Output
  _overlap.create( size1, size2, CV_32FC1 );
  Mat overlap = _overlap.getMat();

  for (int i = 0; i < size1; i++){
    for (int j = 0; j < size2; j++){
      overlap.at<float>(i,j) = bboxOverlap1D<TEMPL>(bb1_, i, bb2_, j);
      //*overlap++ = bb_overlap_(bb1 + bb1_->rows*i, bb2 + bb2_->rows*j);
    }
  }
}

template<class TEMPL>
void bboxDistance(InputArray _bb, OutputArray _overlap)
{
  Mat bb = _bb.getMat();

//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  bboxOverlap<TEMPL>(bb, _overlap);
  Mat overlap = _overlap.getMat();
  for (int j=0; j<overlap.rows; j++){
    for (int i=0; i<overlap.cols; i++){
      overlap.at<TEMPL>(j,i) = 1 - overlap.at<TEMPL>(j,i);
    }
  }
  //  return (1.0 - bboxOverlap(bb));
}

template<class TEMPL>
void bboxDistance(InputArray _bb1, InputArray _bb2, OutputArray _overlap)
{
  Mat bb1 = _bb1.getMat();
  Mat bb2 = _bb2.getMat();

//  CV_Assert( (bb1.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (bb2.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  bboxOverlap<TEMPL>(bb1, bb2, _overlap);
  Mat overlap = _overlap.getMat();
  for (int j=0; j<overlap.rows; j++){
    for (int i=0; i<overlap.cols; i++){
      overlap.at<TEMPL>(j,i) = 1 - overlap.at<TEMPL>(j,i);
    }
  }
  //  return (1.0 - bboxOverlap(bb1, bb2));
}

template<class TEMPL>
void bbox2Rect(InputArray _bb_in, Rect& rect_out)
{
  Mat bb_in = _bb_in.getMat();

//  CV_Assert( (bb_in.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb_in.cols == 1 && bb_in.rows == 4 );

  rect_out.x = (int)(bb_in.at<TEMPL>(0,0));
  rect_out.y = (int)(bb_in.at<TEMPL>(1,0));
  rect_out.width = (int)(bb_in.at<TEMPL>(2,0) - rect_out.x);
  rect_out.height = (int)(bb_in.at<TEMPL>(3,0) - rect_out.y);
}

template<class TEMPL>
void rect2BBox(const Rect rect_in, OutputArray _bb_out)
{
  Mat bb_out = _bb_out.getMat();

//  CV_Assert( (bb_out.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb_out.cols == 1 && bb_out.rows == 4 );

  bb_out.at<TEMPL>(0,0) = (TEMPL)(rect_in.x);
  bb_out.at<TEMPL>(1,0) = (TEMPL)(rect_in.y);
  bb_out.at<TEMPL>(2,0) = (TEMPL)(rect_in.x + rect_in.width);
  bb_out.at<TEMPL>(3,0) = (TEMPL)(rect_in.y + rect_in.height);
}

template<class TEMPL>
bool bboxIsDef(InputArray _bb)
{
  Mat bb = _bb.getMat();
//  CV_Assert( (bb.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  if (bb.at<TEMPL>(0,0) >= 0 &&
      bb.at<TEMPL>(0,0) < MAX_IMAGE_WIDTH &&
      bb.at<TEMPL>(1,0) >= 0 &&
      bb.at<TEMPL>(1,0) < MAX_IMAGE_HEIGHT &&
      bb.at<TEMPL>(2,0) > bb.at<TEMPL>(0,0) &&
      bb.at<TEMPL>(2,0) < MAX_IMAGE_WIDTH &&
      bb.at<TEMPL>(3,0) > bb.at<TEMPL>(1,0) &&
      bb.at<TEMPL>(3,0) < MAX_IMAGE_HEIGHT
  ){
    return true;
  }else{
    return false;
  }
}

template<class TEMPL>
bool bboxIsOut(InputArray _bb_in, const Size imsize)
{
  Mat bb_in = _bb_in.getMat();

//  CV_Assert( (bb_in.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  if (bb_in.at<TEMPL>(0,0) >= imsize.width ||
      bb_in.at<TEMPL>(1,0) >= imsize.height ||
      bb_in.at<TEMPL>(2,0) >= imsize.width ||
      bb_in.at<TEMPL>(3,0) >= imsize.height ||
      bb_in.at<TEMPL>(2,0) <= 1 ||
      bb_in.at<TEMPL>(3,0) <= 1
  ){
    return true;
  } else{
    return false;
  }
}

template<class TEMPL>
void saveMatrix(InputArray _matrix, const bool isHex)
{
  if (_matrix.empty()){
    return;
  }
  Mat matrix = _matrix.getMat();

  ofstream fout("test.txt");

  if (isHex){
    for (int i=0; i<matrix.rows; i++){
      for (int j=0; j<matrix.cols; j++){
        fout << hex << (int)(matrix.at<TEMPL>(i,j)) << ",";
      }
      fout << "\n";
    }
  }else{
    for (int i=0; i<matrix.rows; i++){
      for (int j=0; j<matrix.cols; j++){
        fout << (TEMPL)(matrix.at<TEMPL>(i,j)) << ",";
      }
      fout << "\n";
    }
  }
  fout.close();
}

template<class TEMPL>
void bboxScan( InputArray _srcBB, const int imgwidth, const int imgheight, const int MINBB, OutputArray _detectorGrid, OutputArray _detectorScales )
{
  Mat srcBB = _srcBB.getMat();

//  CV_Assert( (srcBB.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  // detector.curBBox, cvSize(cvGetSize(curGrayImg).width, cvGetSize(curGrayImg).height), opt.model.min_win, detector.grid, detector.scales
  float SHIFT = 0.1;
  float SCALE[21];
  // = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,0.57870,0.69444,0.83333,1.00000,1.20000,1.44000,1.72800,2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
  for (int i=0; i<21; i++){
    SCALE[i] = (float)pow(1.2, (i-10)*1.0);
  }

  int col;

  // Check if input bbox is smaller than minimum
  //if (fmin(TRDT.bbox_rect.width, TRDT.bbox_rect.height) < MINBB){
  if (fmin(bboxWidth1D<TEMPL>(srcBB), bboxHeight1D<TEMPL>(srcBB)) < MINBB){
    cerr << "\nError(bboxScan): Input bbox is smaller than minimum!\n"
        << endl;
    CV_Assert( fmin(bboxWidth1D<TEMPL>(srcBB), bboxHeight1D<TEMPL>(srcBB)) >= MINBB );
    return;
  }

  float bbW[21], bbH[21];
  for (int i=0; i<21; i++){
    bbW[i] = bboxWidth1D<TEMPL>(srcBB) * SCALE[i];
    bbH[i] = bboxHeight1D<TEMPL>(srcBB) * SCALE[i];
  }

  float bbSHH[21], bbSHW[21];
  for (int i=0; i<21; i++){
    bbSHH[i] = SHIFT * fmin(bbH[i], bbH[i]);
    bbSHW[i] = SHIFT * fmin(bbH[i], bbW[i]);
  }

  float bbF[] = {2., 2., imgwidth, imgheight}; // 2 2 imsize(2) imsize(1)

  BBOXES bbs;
  SCA sca;
  int idx = 0; // 1 or 0?

  for (int i = 0; i<21; i++){
    if (bbW[i] < MINBB || bbH[i] < MINBB) { continue; }
    if (cvRound(((bbF[2]-bbW[i])-bbF[0])/bbSHW[i]) <= 0 || cvRound(((bbF[3]-bbH[i])-bbF[1])/bbSHH[i]) <= 0) { continue; }

    Mat left(1, cvRound(((bbF[2]-bbW[i])-bbF[0])/bbSHW[i]), CV_32FC1);
    for (int ii=0; ii<left.cols; ii++){
      left.at<TEMPL>(0,ii) = (TEMPL)(bbF[0]+ii*bbSHW[i]);
    }
    // left = cvRound(bbF(0):bbSHW(i):(bbF(2)-bbW(i)-1));

    Mat top(1, cvRound(((bbF[3]-bbH[i])-bbF[1])/bbSHH[i]), CV_32FC1);
    for (int ii=0; ii<top.cols; ii++){
      top.at<TEMPL>(0,ii) = (TEMPL)(bbF[1]+ii*bbSHH[i]);
    }
    // top = cvRound(bbF(1):bbSHH(i):(bbF(3)-bbH(i)-1));

    // Computes all possible ntupples.
    Mat grid(2, top.cols*left.cols, srcBB.type());
    col = 0;
    for (int jj=0; jj<top.cols; jj++){
      for (int ii = 0; ii<left.cols; ii++){
        grid.at<TEMPL>(0,col) = top.at<TEMPL>(0,jj);
        col++;
      }
    }
    col = 0; // 10370, 6909, 4640, 2914, 1824, 1178, 700, 400, 224, 100, 38
    for (int ii = 0; ii<top.cols; ii++){
      for (int jj=0; jj<left.cols; jj++){
        grid.at<TEMPL>(1,col) = left.at<TEMPL>(0,jj);
        col++;
      }
    }

    //if (isempty(grid)) { continue; }
    Mat bbscell(6, grid.cols, srcBB.type());
    // (grid[0],grid[1]) : X1,Y1
    // (grid[2],grid[3]) : X2,Y2
    // grid[4] : pointer to features for this scale
    // grid[5] : number of left-right bboxes, will be used for searching neighbours
    for (int ii = 0; ii<grid.cols; ii++){
      bbscell.at<TEMPL>(0,ii) = grid.at<TEMPL>(1,ii); // X1: 2,5,7,10,12,...
      bbscell.at<TEMPL>(1,ii) = grid.at<TEMPL>(0,ii); // Y1: 2,2,2,2,2,...
      bbscell.at<TEMPL>(2,ii) = grid.at<TEMPL>(1,ii)+bbW[i]-1.; // X2: 27,30,32,35,37,...
      bbscell.at<TEMPL>(3,ii) = grid.at<TEMPL>(0,ii)+bbH[i]-1.; // Y2: 43,43,43,43,43,...
      bbscell.at<TEMPL>(4,ii) = (TEMPL)idx; // pointer to features for this scale: 0,0,0,0,0,...
      bbscell.at<TEMPL>(5,ii) = (TEMPL)left.cols; // number of left-right bboxes, will be used for searching neighbours: 170,170,170,170,170,...
    }

    bbs.push_back(bbscell);
    Mat scacell(2, 1, srcBB.type());
    scacell.at<TEMPL>(0,0) = (TEMPL)bbH[i]; // 42,50,60,73,...
    scacell.at<TEMPL>(1,0) = (TEMPL)bbW[i]; // 26,31,37,45,...
    sca.push_back(scacell);
    idx = idx + 1;
    // delete all pointers
    left.release();
    top.release();
    grid.release();
  }

  int max_col = 0;
  for( BBOXES::iterator iter = bbs.begin(); iter != bbs.end(); ++iter ){
    max_col += (*iter).cols;
  }

  _detectorGrid.create( 6, max_col, srcBB.type() ); // width=col, height=row
  Mat detectorGrid = _detectorGrid.getMat();

//  cout << "\ndetectorGrid\n"
//       << detectorGrid.rows
//       << "\n"
//       << detectorGrid.cols
//       << endl;

  col = 0;
  for( BBOXES::iterator iter = bbs.begin(); iter != bbs.end(); ++iter ){
    for (int jj=0; jj<(*iter).cols; jj++){
      for (int ii = 0; ii<6; ii++){
        CV_Assert( col < detectorGrid.cols );
        detectorGrid.at<TEMPL>(ii,col) = (*iter).at<TEMPL>(ii,jj);
      }
      col++;
    }
    //cvReleaseMat(&(*iter));
  }

  for( BBOXES::iterator iter = bbs.begin(); iter != bbs.end(); ++iter ){
    (*iter).release();
  //    delete *iter;
  }

  max_col = 0;
  for( SCA::iterator iter = sca.begin(); iter != sca.end(); ++iter ){
    max_col += (*iter).cols;
  }

  _detectorScales.create( 2, max_col, srcBB.type() );
  Mat detectorScales = _detectorScales.getMat();

  col = 0;
  for( SCA::iterator iter = sca.begin(); iter != sca.end(); ++iter ){
    for (int jj=0; jj<(*iter).cols; jj++){ // col
      for (int ii = 0; ii<2; ii++){ // row
        CV_Assert( col < detectorScales.cols );
        detectorScales.at<TEMPL>(ii,col) = (*iter).at<TEMPL>(ii,jj);
      }
      col++;
    }
    //cvReleaseMat(&(*iter));
  }

  for( SCA::iterator iter = sca.begin(); iter != sca.end(); ++iter ){
    (*iter).release();
  //    delete *iter;
  }
}

// Clustering of tracker and detector responses
// First cluster returned corresponds to the tracker
template<class TEMPL>
void bboxClusterConfidence(InputArray _iBB, InputArray _iConf, const int numBB, OutputArray _oBB, OutputArray _oConf, OutputArray _oSize)
{
  Mat iBB = _iBB.getMat();
  Mat iConf = _iConf.getMat();

  TEMPL SPACE_THR = 0.5;

  if (numBB == 0){
    return;
  }

  Mat T;
  float temp_dist;
  Mat bbd;
  Mat Z;
  int col = 0;

  char method[3] = "si";
  char flag[9] = "distance";
  // iBB : EX: 4x3

  switch (numBB)
  //  switch (test->cols)
  {
  case 0:
//    T = 0;
    break;
  case 1:
    T.create(1, 1, CV_32SC1);
    T.setTo(Scalar(1));
    break;
  case 2:
    T.create(2, 1, CV_32SC1);
    T.setTo(Scalar(1));
    //T = ones(2,1);
//    bboxDistance<TEMPL>(iBB, temp_dist);
    temp_dist = bboxOverlap1D<TEMPL>(iBB, 0, iBB, 1);
    temp_dist = 1 - temp_dist;

    if (temp_dist > SPACE_THR){
      T.at<int>(1,0) = 2;
    }
    break;
  default:
    // bbd : EX: 1x6
//    bboxDistance<TEMPL>(iBB, bbd);
    bbd.create( 1, (numBB)*(numBB-1)/2, CV_32FC1 );
    col = 0;
    for (int i = 0; i < numBB-1; i++){
      for (int j = i+1; j < numBB; j++){
        CV_Assert( col < bbd.cols );
        bbd.at<float>(0,col) = bboxOverlap1D<TEMPL>(iBB, i, iBB, j);
        col++;
      }
    }
    for (int i=0; i<bbd.cols; i++){
      bbd.at<TEMPL>(0,i) = 1 - bbd.at<float>(0,i);
    }

    // Z : EX: 3x3
    linkage<TEMPL>(bbd, method, Z);
//    saveMatrix<TEMPL>(Z, false);

    clusterCutoffCriterion(Z, SPACE_THR, flag, T);
  }

  Mat idx_cluster;
  uniqueMat<int>(T,idx_cluster);
  int num_clusters = idx_cluster.rows;

  _oBB.create( 4, num_clusters, iBB.type() );
  Mat oBB = _oBB.getMat();

  _oConf.create( 1, num_clusters, CV_32FC1 );
  Mat oConf = _oConf.getMat();

  _oSize.create( 1, num_clusters, CV_32SC1 );
  Mat oSize = _oSize.getMat();

  Mat idx_(T.rows, 1, CV_32SC1);
  int* idx = (int*)idx_.data;

  float mean[4]; mean[0] = 0.; mean[1] = 0.; mean[2] = 0.; mean[3] = 0.;
  float mean_db = 0;
  int num = 0;
  int sum = 0;
  for (int k = 0; k<num_clusters; k++){
    mean[0] = 0.; mean[1] = 0.; mean[2] = 0.; mean[3] = 0.;
    mean_db = 0; num = 0; sum = 0;
    for (int j = 0; j < idx_.rows; j++){
      if (idx_cluster.at<int>(k,0) == T.at<int>(j,0)){
        idx[j] = 1;
        num++;
      } else{
        idx[j] = 0;
      }
    }
    //idx = T == idx_cluster(i);

    for (int i = 0; i < idx_.rows; i++){
      if (idx[i] == 1){
        CV_Assert( i < numBB && i < iConf.cols );
        for (int j = 0; j < 4; j++){
          mean[j] += (float)(iBB.at<TEMPL>(j,i));
          //          mean_int[j] += Coord_INT(test, j, i);
        }
        mean_db += iConf.at<float>(0,i);
        sum =+ idx[i];
      }
    }

    for (int j = 0; j < 4; j++){
      mean[j] /= (float)num;
      oBB.at<TEMPL>(j,k) = (TEMPL)mean[j];
    }
    //oBB(:,i)  = mean(iBB(1:4,idx),2);

    mean_db /= num;
    oConf.at<float>(0,k) = (float)mean_db;
    //oConf(i)  = mean(iConf(idx));
    oSize.at<int>(0,k) = sum;
    //oSize(i)  = sum(idx);
  }
  idx_cluster.release();
  bbd.release();
  Z.release();
  T.release();
  idx_.release();

}

// Generates numM x numN points on BBox.
template<class TEMPL>
void bboxPoints(InputArray _bb, const int numM, const int numN, const int margin, OutputArray _pt)
{
  Mat bb_ = _bb.getMat();
  TEMPL* bb = (TEMPL*)bb_.data;

//  CV_Assert( (bb_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( bb_.cols == 1 );

  Mat temp_(4, 1, bb_.type());
  TEMPL* temp = (TEMPL*)temp_.data;
  temp[0] = (TEMPL)(bb[0] + margin);
  temp[1] = (TEMPL)(bb[1] + margin);
  temp[2] = (TEMPL)(bb[2] - margin);
  temp[3] = (TEMPL)(bb[3] - margin);
  // bb(1:2) = bb(1:2)+margin;
  // bb(3:4) = bb(3:4)-margin;

  if (numM == 1 && numN ==1){
    bboxCenter<TEMPL>(temp_, _pt);
    temp_.release();
    return;
  }

  if (numM == 1 && numN > 1){
    //int stepW = (Coord_INT(temp, 2, 0)-Coord_INT(temp, 0, 0)) / (numN - 1);
    TEMPL stepW = (TEMPL)((temp[2]-temp[0]) / (TEMPL)(numN - 1.));
    Mat in1(1, numN, temp_.type());
    for (int j=0; j<in1.cols; j++){
      in1.at<TEMPL>(0,j) = (TEMPL)((temp[0]+j*stepW));
    }
    Mat in2(1, 1, temp_.type());
    in2.at<TEMPL>(0,0) = (temp[1]+temp[3])/2;
    nTuples<TEMPL>( in1, in2, _pt );
    // pt = nTuples(bb(0):stepW:bb(2),c(1));
    in1.release();
    in2.release();
    temp_.release();
    return;
  }

  if (numM > 1 && numN == 1){
    //int stepH = (Coord_INT(temp, 3, 0)-Coord_INT(temp, 1, 0)) / (numM - 1);
    TEMPL stepH = (TEMPL)((temp[3]-temp[1]) / (TEMPL)(numM - 1.));
    Mat in1(1, numM, temp_.type());
    for (int j=0; j<in1.cols; j++){
      in1.at<TEMPL>(0,j) = (TEMPL)((temp[1]+j*stepH));
    }
    Mat in2(1, 1, temp_.type());
    in2.at<TEMPL>(0,0) = (temp[0]+temp[2])/2;
    nTuples<TEMPL>( in2, in1, _pt );
    //    pt = nTuples(c(0),(bb(1):stepH:bb(3)));
    in1.release();
    in2.release();
    temp_.release();
    return;
  }

  TEMPL stepW = (TEMPL)((temp[2]-temp[0]) / (TEMPL)(numN - 1.));
  //  stepW = (bb(2)-bb(0)) / (numN - 1);
  TEMPL stepH = (TEMPL)((temp[3]-temp[1]) / (TEMPL)(numM - 1.));
  //  stepH = (bb(3)-bb(1)) / (numM - 1);

  Mat in1(1, numN, temp_.type());
  for (int j=0; j<in1.cols; j++){
    in1.at<TEMPL>(0,j) = (TEMPL)((temp[0]+j*stepW));
  }
  Mat in2(1, numM, temp_.type());
  for (int j=0; j<in2.cols; j++){
    in2.at<TEMPL>(0,j) = (TEMPL)((temp[1]+j*stepH));
  }
  nTuples<TEMPL>( in1, in2, _pt );

  //  pt = nTuples(bb(0):stepW:bb(2),(bb(1):stepH:bb(3)));
  in1.release();
  in2.release();
  temp_.release();
  return;
}

// Generates numM x numN points on BBox.
template<class TEMPL>
void bboxPointsCustom(InputArray _bb, const int numM, const int numN, const int margin, OutputArray _pt)
{
  Mat bb_ = _bb.getMat();
  TEMPL* bb = (TEMPL*)bb_.data;

  Mat pt = _pt.getMat();

  TEMPL stepW = (TEMPL)(((bb[2] - margin)-(bb[0] + margin)) / (TEMPL)(numN - 1.));
  TEMPL stepH = (TEMPL)(((bb[3] - margin)-(bb[1] + margin)) / (TEMPL)(numM - 1.));

  int col = 0;
  for (int j=0; j<numN; j++){
    for (int i = 0; i<numM; i++){
      pt.at<TEMPL>(0,col) = (TEMPL)(((bb[0] + margin)+j*stepW));
      col++;
    }
  }
  col = 0;
  for (int i = 0; i<numN; i++){
    for (int j=0; j<numM; j++){
      pt.at<TEMPL>(1,col) = (TEMPL)(((bb[1] + margin)+j*stepH));
      col++;
    }
  }

}

template<class TEMPL>
void bboxPredict(InputArray _BB0, InputArray _pt0, InputArray _pt1, const int numMN, OutputArray _BB1)
{
  Mat BB0_ = _BB0.getMat();
  TEMPL* BB0 = (TEMPL*)BB0_.data;
  Mat pt0 = _pt0.getMat();
  Mat pt1 = _pt1.getMat();

//  CV_Assert( (BB0_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (pt0.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (pt1.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( pt0.rows == pt1.rows && pt0.cols == pt1.cols );

  Mat of(2, numMN, pt0.type());

//  of = pt1 - pt0;
  // of  = pt1 - pt0;
  for (int i=0; i<numMN; i++){
    for (int j=0; j<2; j++){
      of.at<TEMPL>(j,i) = pt1.at<TEMPL>(j,i)-pt0.at<TEMPL>(j,i);
    }
  }

  // rwocol = 0 -> row = 0
  float dx = (float)median1D<TEMPL>(of, 0, 0);
  // dx  = median(of(1,:));

  // rwocol = 0 -> row = 1
  float dy = (float)median1D<TEMPL>(of, 0, 1);
  // dy  = median(of(2,:));

  Mat d1( 1, (numMN)*(numMN-1)/2, pt0.type() );
//  distanceCol<TEMPL>(pt0, 2, d1); // euclidean
  // d1 = pdist(pt0','euclidean');
  float sum;
  int cnt = 0;
  for(int i=0; i<numMN-1; i++){
    for(int j=i+1; j<numMN; j++){
      sum = 0.0;
      for(int jj=0; jj<2; jj++){
        sum += pow(pt0.at<TEMPL>(jj,i)-pt0.at<TEMPL>(jj,j), 2);
      }
      d1.at<TEMPL>(0,cnt) = (TEMPL)sqrt(sum);
      cnt++;
    }
  }

  Mat d2( 1, (numMN)*(numMN-1)/2, pt1.type() );
//  distanceCol<TEMPL>(pt1, 2, d2); // euclidean
  // d2 = pdist(pt1','euclidean');
  cnt = 0;
  for(int i=0; i<numMN-1; i++){
    for(int j=i+1; j<numMN; j++){
      sum = 0.0;
      for(int jj=0; jj<2; jj++){
        sum += pow(pt1.at<TEMPL>(jj,i)-pt1.at<TEMPL>(jj,j), 2);
      }
      d2.at<TEMPL>(0,cnt) = (TEMPL)sqrt(sum);
      cnt++;
    }
  }

//  Mat d3(d1.rows, d1.cols, CV_32FC1);
//  d3 = d2/d1; // d2./d1  -> d3
  for(int i=0; i<(numMN)*(numMN-1)/2; i++){
    d2.at<TEMPL>(0,i) = d2.at<TEMPL>(0,i)/d1.at<TEMPL>(0,i);
  }

  // rwocol = 0 -> row = 0
  float s = (float)median1D<TEMPL>(d2, 0, 0);
  // s = median(d2./d1);

  TEMPL bbw = (BB0[2]-BB0[0]+1.); //bboxWidth1D(BB0);
  TEMPL bbh = (BB0[3]-BB0[1]+1.);//bboxHeight1D(BB0);

  float s1 = 0.5*(s-1)*bbw;
  //s1  = 0.5*(s-1)*bboxWidth(BB0);
  float s2 = 0.5*(s-1)*bbh;
  //s2  = 0.5*(s-1)*bboxHeight(BB0);

//  _BB1.create( 4, 1, BB0_.type() );
  Mat BB1_ = _BB1.getMat();
  TEMPL* BB1 = (TEMPL*)BB1_.data;

  BB1[0] = BB0[0]+(TEMPL)(dx-s1);
  BB1[1] = BB0[1]+(TEMPL)(dy-s2);
  BB1[2] = BB0[2]+(TEMPL)(s1+dx);
  BB1[3] = BB0[3]+(TEMPL)(s2+dy);
  //BB1  = [BB0(1)-s1; BB0(2)-s2; BB0(3)+s1; BB0(4)+s2] + [dx; dy; dx; dy];

  of.release();
  d1.release();
  d2.release();
//  d3.release();
}

#endif /* BOUNDING_BOX_H_ */
