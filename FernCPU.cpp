/***********************************************************************
  $FILENAME    : FernCPU.cpp

  $TITLE       : Fern class implementation

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Implements the Fern class for running on CPU

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef USE_OCL_

#include "tldTracker.hpp"
#include "BoundingBox.hpp"

using namespace std;
using namespace cv;

#define sub2idx(i) ((int) (floor((i)+0.0)))

void
tld::cpu::Fern::saveWEIGHT(void)
{
  ofstream fout("WEIGHT.txt");

  for (int idxFeature=0; idxFeature<num_features_bit; idxFeature++){
    for (int idxTree=0; idxTree<nTREES; idxTree++){
      fout << WEIGHT.at<float>(idxTree,idxFeature) << ",";
    }
    fout << "\n";
  }

  fout.close();

  ofstream fout2("nP.txt");

  for (int idxFeature=0; idxFeature<num_features_bit; idxFeature++){
    for (int idxTree=0; idxTree<nTREES; idxTree++){
      fout2 << nP.at<int>(idxTree,idxFeature) << ",";
    }
    fout2 << "\n";
  }

  fout2.close();

  ofstream fout3("nN.txt");

  for (int idxFeature=0; idxFeature<num_features_bit; idxFeature++){
    for (int idxTree=0; idxTree<nTREES; idxTree++){
      fout3 << nN.at<int>(idxTree,idxFeature) << ",";
    }
    fout3 << "\n";
  }

  fout3.close();

}

void
tld::cpu::Fern::saveOFF(const int BBOX_a_x, const int BBOX_a_y, const int BBOX_d_x,
    const int BBOX_d_y, const int OFF_0_x, const int OFF_0_y, const int OFF_1_x, const int OFF_1_y)
{

  ofstream fout("OFF.txt");

  fout << "BBOX a(" << BBOX_a_x << "," << BBOX_a_y << ")\n";
  fout << "BBOX d(" << BBOX_d_x << "," << BBOX_d_y << ")\n";
  fout << "OFF 0(" << OFF_0_x << "," << OFF_0_y << ")\n";
  fout << "OFF 1(" << OFF_1_x << "," << OFF_1_y << ")\n";

  fout.close();

}

tld::cpu::Fern::Fern ( )
{

  thrN = 0; nTREES = 0; nFEAT = 0; nSCALE = 0; iHEIGHT = 0; iWIDTH = 0;
  nBIT = 1; // number of bits per feature

}

tld::cpu::Fern::~Fern ()
{
  BBOX.release();
  OFF.release();
  integral_img.release();
  integral_img2.release();

  WEIGHT.release();
  nP.release();
  nN.release();
}

void
tld::cpu::Fern::init( InputArray _img, InputArray _grid, InputArray _features, InputArray _scales )
{
  Mat img = _img.getMat();
  Mat grid = _grid.getMat();
  Mat features = _features.getMat();
  Mat scales = _scales.getMat();

  // INIT: function(1, img, bb, features, scales)
  // =============================================================================

  if ( !BBOX.empty() ) {
    cerr << "\nERROR(fern): already initialized.\n"
        << endl;
    CV_Assert( !BBOX.empty() );
    return;
  }

  CV_Assert( (grid.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( (features.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
  CV_Assert( (scales.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  iHEIGHT    = img.rows;
  iWIDTH     = img.cols;
  nTREES     = features.cols; // opt.model.num_trees : 10
  nFEAT      = features.rows/4; // feature has 2 points: x1,y1,x2,y2 : 4*opt.model.num_features -> 4*13
  thrN       = 0.5 * nTREES;
  nSCALE     = scales.cols; // 11

  // width : cols
  // height : rows
  integral_img.create(iHEIGHT, iWIDTH, CV_64FC1);
  integral_img2.create(iHEIGHT, iWIDTH, CV_64FC1);

  // BBOX
  // create_offsets_bbox
  BBOX.create(11, grid.cols, CV_32SC1);
  //  a --------- c
  //    |       |
  //    |       |
  //    |       |
  //  b --------- d
  // (bb[0],bb[1]) : X1,Y1
  // (bb[2],bb[3]) : X2,Y2
  // bb[4] : area
  // bb[5] : pointer to features for this scale ((bb[4]-1)*2*nFEAT*nTREES)
  // bb[6] : number of left-right bboxes, will be used for searching neighbours
  for (int idxBBox = 0; idxBBox<grid.cols; idxBBox++){
    BBOX.at<int>(0,idxBBox) = sub2idx(grid.at<float>(0,idxBBox)); // corner a : x 1,4,6
    BBOX.at<int>(1,idxBBox) = sub2idx(grid.at<float>(1,idxBBox)); // corner a : y 1,1,1
    BBOX.at<int>(2,idxBBox) = sub2idx(grid.at<float>(0,idxBBox)); // corner b : x 1,4,6
    BBOX.at<int>(3,idxBBox) = sub2idx(grid.at<float>(3,idxBBox)); // corner b : y 42,42,42
    BBOX.at<int>(4,idxBBox) = sub2idx(grid.at<float>(2,idxBBox)); // corner c : x 26,29,31
    BBOX.at<int>(5,idxBBox) = sub2idx(grid.at<float>(1,idxBBox)); // corner c : y 1,1,1
    BBOX.at<int>(6,idxBBox) = sub2idx(grid.at<float>(2,idxBBox)); // corner d : x 26,29,31
    BBOX.at<int>(7,idxBBox) = sub2idx(grid.at<float>(3,idxBBox)); // corner d : y 42,42,42
    BBOX.at<int>(8,idxBBox) = (int)((grid.at<float>(2,idxBBox)-grid.at<float>(0,idxBBox))*(grid.at<float>(3,idxBBox)-grid.at<float>(1,idxBBox))); // 1025,1025,1025
    BBOX.at<int>(9,idxBBox) = (int)(grid.at<float>(4,idxBBox))*nTREES; // pointer to features for this scale : 0,0,0
    BBOX.at<int>(10,idxBBox) = (int)grid.at<float>(5,idxBBox); // number of left-right bboxes, will be used for searching neighbours: 170,170,170
  }
  //BBOX             = create_offsets_bbox(bbox);

  OFF.create(4*nFEAT, nSCALE*nTREES, CV_32SC1);
  int col = 0;
  for (int idxScale = 0; idxScale < nSCALE; idxScale++){     // scales : 2x11
    for (int idxTree = 0; idxTree < nTREES; idxTree++){   // features : 4*13x10= 4 x #features x #trees
      for (int idxFeature = 0; idxFeature < nFEAT; idxFeature++){  // features : 4*13x10
        CV_Assert( col < OFF.cols );
        OFF.at<int>(0+idxFeature*4,col) = sub2idx((scales.at<float>(1,idxScale))*features.at<float>(0+idxFeature*4,idxTree)); // x1: 15,20,23,10,3,
        OFF.at<int>(1+idxFeature*4,col) = sub2idx((scales.at<float>(0,idxScale))*features.at<float>(1+idxFeature*4,idxTree)); // y1: 33,25,4,8,12,
        OFF.at<int>(2+idxFeature*4,col) = sub2idx((scales.at<float>(1,idxScale))*features.at<float>(2+idxFeature*4,idxTree)); // x2: 23,25,3,10,25,
        OFF.at<int>(3+idxFeature*4,col) = sub2idx((scales.at<float>(0,idxScale))*features.at<float>(3+idxFeature*4,idxTree)); // y2: 33,25,4,0,12,
      }
      col++;
    }
  }
  //OFF          = create_offsets(scales, features);

  num_features_bit = (int)pow(2.0,nBIT*nFEAT); // 8192

  WEIGHT.create(nTREES, num_features_bit, CV_32FC1);
  WEIGHT.setTo(Scalar(0.));
  nP.create(nTREES, num_features_bit, CV_32SC1);
  nP.setTo(Scalar(0));
  nN.create(nTREES, num_features_bit, CV_32SC1);
  nN.setTo(Scalar(0));

}

void
tld::cpu::Fern::update_( InputArray _X, const int idxBBoxWarps, const int C, const int N )
{
  Mat X = _X.getMat();

  CV_Assert( idxBBoxWarps < X.cols );

  int idx = 0;
  for (int idxTree = 0; idxTree < nTREES; idxTree++){
    idx = X.at<int>(idxTree,idxBBoxWarps);
    CV_Assert( idx < num_features_bit );

    (C==1) ? nP.at<int>(idxTree,idx) += N : nN.at<int>(idxTree,idx) += N;

    if (nP.at<int>(idxTree,idx) == 0){
      WEIGHT.at<float>(idxTree,idx) = 0;
    } else {
      WEIGHT.at<float>(idxTree,idx) = ((float)(nP.at<int>(idxTree,idx))) / (float)(nP.at<int>(idxTree,idx) + nN.at<int>(idxTree,idx));
    }
  }
}

float
tld::cpu::Fern::measureForest( InputArray _X, const int idxBBoxWarps )
{
  Mat X = _X.getMat();

  CV_Assert( idxBBoxWarps < X.cols );

  float votes = 0;
  for (int idxTree = 0; idxTree < nTREES; idxTree++){
    CV_Assert( X.at<int>(idxTree,idxBBoxWarps) < num_features_bit );
    votes += WEIGHT.at<float>(idxTree,X.at<int>(idxTree,idxBBoxWarps));
    //    ACE_DEBUG(( LM_DEBUG, ACE_TEXT("WEIGHT[%d][%d]\n"), idxTree, Coord_INT(X, idxTree, idxBBoxWarps) ));
  }
  return votes;
}

int
tld::cpu::Fern::measureTreeOffset( InputArray _img_blur, const int idxBBox, const int idx_tree )
{
  Mat img_blur = _img_blur.getMat();

  int index = 0;

  CV_Assert( idx_tree < nTREES );
  CV_Assert( idxBBox < BBOX.cols );
  CV_Assert( BBOX.at<int>(9,idxBBox) < OFF.cols );
  CV_Assert( (BBOX.at<int>(9,idxBBox)+idx_tree) < OFF.cols );

  int fp0, fp1;
  int x1, y1, x2, y2;

  for (int idxFeature=0; idxFeature<nFEAT; idxFeature++){
    index <<= 1; // 0, 0, 0, 0, 2, 4, 8, 18, 38, 78, ... 315

    x1 = OFF.at<int>(0+idxFeature*4,BBOX.at<int>(9,idxBBox)+idx_tree)+BBOX.at<int>(0,idxBBox);
    CV_Assert( x1 < iWIDTH );
    y1 = OFF.at<int>(1+idxFeature*4,BBOX.at<int>(9,idxBBox)+idx_tree)+BBOX.at<int>(1,idxBBox);
    CV_Assert( y1 < iHEIGHT );
    x2 = OFF.at<int>(2+idxFeature*4,BBOX.at<int>(9,idxBBox)+idx_tree)+BBOX.at<int>(0,idxBBox);
    CV_Assert( x2 < iWIDTH );
    y2 = OFF.at<int>(3+idxFeature*4,BBOX.at<int>(9,idxBBox)+idx_tree)+BBOX.at<int>(1,idxBBox);
    CV_Assert( y2 < iHEIGHT );

    //    saveOFF(Coord_INT(BBOX, 0, idxBBox),
    //        Coord_INT(BBOX, 1, idxBBox),
    //        Coord_INT(BBOX, 6, idxBBox),
    //        Coord_INT(BBOX, 7, idxBBox),
    //        x1, y1, x2, y2);

    CV_Assert( x1 >= BBOX.at<int>(0,idxBBox)-1 && x1 <= BBOX.at<int>(6,idxBBox)+1 );
    CV_Assert( y1 >= BBOX.at<int>(1,idxBBox)-1 && y1 <= BBOX.at<int>(7,idxBBox)+1 );
    CV_Assert( x2 >= BBOX.at<int>(0,idxBBox)-1 && x2 <= BBOX.at<int>(6,idxBBox)+1 );
    CV_Assert( y2 >= BBOX.at<int>(1,idxBBox)-1 && y2 <= BBOX.at<int>(7,idxBBox)+1 );

    fp0 = (img_blur.at<uchar>(y1,x1) & 0x000000FF);
    //int fp0 = img_blur[off[0]+bbox[0]];

    fp1 = (img_blur.at<uchar>(y2,x2) & 0x000000FF);
    //int fp1 = img_blur[off[1]+bbox[0]];
    if (fp0 > fp1) {
      index |= 1;
    }
    //off += 2;
  }

  CV_Assert( index < num_features_bit );

  return index;
}

float
tld::cpu::Fern::measureBBoxOffset( InputArray _blurImg, const int idxBBox, const float minVar, OutputArray _tPatt, const int colPatt )
{
  Mat blurImg = _blurImg.getMat();
  Mat tPatt = _tPatt.getMat();

  CV_Assert( colPatt < tPatt.cols );

  float conf = 0.0;
  float bboxvar = bboxVarOffset(idxBBox);
  if (bboxvar < minVar) { return conf; }

  int idx = 0;
  for (int idxTree = 0; idxTree < nTREES; idxTree++){
    idx = measureTreeOffset(blurImg, idxBBox, idxTree);
    tPatt.at<int>(idxTree,colPatt) = idx;
    conf += WEIGHT.at<float>(idxTree,idx);
  }
  return conf;
}

float
tld::cpu::Fern::bboxVarOffset( const int idxBBox )
{
  CV_Assert( idxBBox < BBOX.cols );
  // off[0-3] corners of bbox, off[4] area

  double mX = (integral_img.at<double>(BBOX.at<int>(7,idxBBox),BBOX.at<int>(6,idxBBox)) -
      integral_img.at<double>(BBOX.at<int>(5,idxBBox),BBOX.at<int>(4,idxBBox)) -
      integral_img.at<double>(BBOX.at<int>(3,idxBBox),BBOX.at<int>(2,idxBBox)) +
      integral_img.at<double>(BBOX.at<int>(1,idxBBox),BBOX.at<int>(0,idxBBox))) / (double) BBOX.at<int>(8,idxBBox);
  //double mX  = (integral_img[off[3]] - integral_img[off[2]] - integral_img[off[1]] + integral_img[off[0]]) / (double) off[4];

  double mX2 = (integral_img2.at<double>(BBOX.at<int>(7,idxBBox),BBOX.at<int>(6,idxBBox)) -
      integral_img2.at<double>(BBOX.at<int>(5,idxBBox),BBOX.at<int>(4,idxBBox)) -
      integral_img2.at<double>(BBOX.at<int>(3,idxBBox),BBOX.at<int>(2,idxBBox)) +
      integral_img2.at<double>(BBOX.at<int>(1,idxBBox),BBOX.at<int>(0,idxBBox))) / (double) BBOX.at<int>(8,idxBBox);
  //double mX2 = (integral_img2[off[3]] - integral_img2[off[2]] - integral_img2[off[1]] + integral_img2[off[0]]) / (double) off[4];

  double output = mX2 - mX*mX;
  return (float)output;
}

float
tld::cpu::Fern::randdouble()
{
  return rand()/(float(RAND_MAX)+1);
}

// Computes the integral image of image and the integral image of
// the squares of the elements in image.
void
tld::cpu::Fern::integralImage( InputArray _image )
{
  Mat image = _image.getMat();

  // s(x,y)=s(x,y-1)+img(x,y)
  // iimg(x,y)=iimg(x-1,y)+s(x,y)
  // s(x,-1)=0
  // iimg(-1,y)=0

  // s(x,y)=s(x,y-1)+img(x,y)*img(x,y)
  // siimg(x,y)=siimg(x-1,y)+s(x,y)
  // s(x,-1)=0
  // siimg(-1,y)=0
  CV_Assert( iHEIGHT == image.rows );
  CV_Assert( iWIDTH == image.cols );

  unsigned char *i_m = (unsigned char *) image.data;
  double *ii_m = (double *) integral_img.data;
  double *sii_m = (double *) integral_img2.data;

  ii_m[0] = i_m[0];
  sii_m[0] = i_m[0] * i_m[0];

  // Create the first row of the integral image
  for (int x = 1; x < iWIDTH; x++) {
    ii_m[x] = ii_m[x-1] + i_m[x];
    sii_m[x] = sii_m[x-1] + i_m[x]*i_m[x];
  }

  // Compute each other row/column
  for (int y = 1, Y = iWIDTH, YY=0; y < iHEIGHT; y++, Y+=iWIDTH, YY+=iWIDTH) {
    // Keep track of the row sum
    double r = 0, rs = 0;

    for (int x = 0; x < iWIDTH; x++) {
      r += i_m[Y + x];
      rs += i_m[Y + x]*i_m[Y + x];
      ii_m[Y + x] = ii_m[YY + x] + r;
      sii_m[Y + x] = sii_m[YY + x] + rs;
    }
  }

  return;
}

void
tld::cpu::Fern::update( InputArray _X, InputArray _Y, const float Margin, const int bootstrap )
{
  Mat X = _X.getMat();
  Mat Y = _Y.getMat();

  // X : 10x2549
  // Y : 1x2549
  float thrP   = Margin * nTREES;

  //double test;
  for (int j = 0; j < bootstrap; j++){
    for (int idxBBoxWarps = 0; idxBBoxWarps < X.cols; idxBBoxWarps++){
      if (Y.at<int>(0,idxBBoxWarps) == 1){
        if (measureForest(X, idxBBoxWarps) <= thrP){
          update_(X, idxBBoxWarps, 1, 1);
        }
      } else {
        if (measureForest(X, idxBBoxWarps) >= thrN){
          update_(X, idxBBoxWarps, 0, 1);
        }
        //else if (measureForest(X, idxBBoxWarps) > 3){
        //  test = measureForest(X, idxBBoxWarps);
        //}
      }
    }
  }

  //  saveMatrix(Y);
  //  saveWEIGHT();
}

// EVALUATE PATTERNS
void
tld::cpu::Fern::evaluate( InputArray _X, OutputArray _sumWEIGHT )
{
  Mat X = _X.getMat();

  _sumWEIGHT.create( 1, X.cols, CV_32FC1 );
  Mat sumWEIGHT = _sumWEIGHT.getMat();

  for (int idxBBoxWarps = 0; idxBBoxWarps < X.cols; idxBBoxWarps++){
    sumWEIGHT.at<float>(0,idxBBoxWarps) = measureForest(X, idxBBoxWarps);
  }
}

// DETECT: TOTAL RECALL
void
tld::cpu::Fern::detect( InputArray _img_gray, InputArray _img_blur, const int maxBBox, const float minVar, OutputArray _conf, OutputArray _patt )
{
  Mat img_gray = _img_gray.getMat();
  Mat img_blur = _img_blur.getMat();
  Mat conf = _conf.getMat();
  Mat patt = _patt.getMat();

  // Pointer to preallocated output matrixes
  CV_Assert( conf.cols == BBOX.cols && patt.cols == BBOX.cols && patt.rows == nTREES);

  patt.setTo(Scalar(0));
  conf.setTo(Scalar(0));

  int nTest  = BBOX.cols * maxBBox; // 2
  if (nTest <= 0){
    CV_Assert( nTest > 0 );
    return;
  }
  if (nTest > BBOX.cols) nTest = BBOX.cols;
  float pStep  = (float) (BBOX.cols / nTest); // 1
  float pState = randdouble() * pStep; // 0.71

  // Integral images
  integralImage(img_gray);

  //////////////////////////////////////
//  vector<ocl::Info> info;
//   CV_Assert(ocl::getDevice(info));
//
//  cv::ocl::oclMat pyr_ocl;
//
//  CV_Assert(img_gray.depth() <= CV_32F && img_gray.channels() <= 4);
//  pyr_ocl.create((img_gray.rows + 1) / 2, (img_gray.cols + 1) / 2, img_gray.type());
////  pyr_ocl.create(10, 10, img_gray.type());
//
//  detectGPU( ocl::oclMat(img_gray), pyr_ocl );
  ///////////////////////////////////////

  // totalrecall
  int idxBBox = 0;

  while (1){
    // Get index of bbox
    idxBBox = (int) floor(pState); // 0 1 29297
    pState += pStep;
    if (idxBBox >= BBOX.cols) { break; }

    // measure bbox
    conf.at<float>(0,idxBBox) = measureBBoxOffset(img_blur, idxBBox, minVar, patt, idxBBox);
  }

}

// GET PATTERNS
void
tld::cpu::Fern::getPattern( InputArray _img_gray, InputArray _img_blur, InputArray _idx, const float minVar, OutputArray _pattern, OutputArray _status )
{
  Mat img_gray = _img_gray.getMat();
  Mat img_blur = _img_blur.getMat();
  Mat idx = _idx.getMat();

  CV_Assert( idx.rows == 1 );

  // bbox indexes
  // Example: idx->rows = 1 & idx->cols = 10
  int numIdx = idx.cols;

  // minimal variance
  if (minVar > 0){
    integralImage(img_gray);
  }

  // output patterns
  // Example: pattern->rows = 10 & pattern->cols = 10
  _pattern.create(nTREES, numIdx, CV_32SC1);
  Mat pattern = _pattern.getMat();
  pattern.setTo(Scalar(0));
  // Example: status->rows = 1 & status->cols = 10
  _status.create(1, numIdx, CV_32SC1);
  Mat status = _status.getMat();

  for (int idxBBox = 0; idxBBox < numIdx; idxBBox++){ // cols : index of BBoxes
    if (minVar > 0){
      float bboxvar = bboxVarOffset(idxBBox);
      //double bboxvar = bboxVarOffset(integral_img, integral_img2, BBOX+j*BBOX_STEP);
      if (bboxvar < minVar) {	status.at<int>(0,idxBBox) = 0; continue; }
    }
    status.at<int>(0,idxBBox) = 1;
    for (int idxTree = 0; idxTree < nTREES; idxTree++){ // rows
      pattern.at<int>(idxTree,idxBBox) = measureTreeOffset(img_blur, idx.at<int>(0,idxBBox), idxTree);
    }
  }
}

#endif
