/***********************************************************************
  $FILENAME    : LucasKanadeCPU.cpp

  $TITLE       : LucasKanade class implementation

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Implements the LucasKanade class for running on CPU

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef USE_OCL_

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "tldTracker.hpp"

using namespace std;
using namespace cv;

tld::cpu::LucasKanade::LucasKanade( InputArray _img )
{
  Mat img = _img.getMat();

  win_size = LUCASKANADE_WINSIZE; //4->61
  nPts = 0;
}

tld::cpu::LucasKanade::~LucasKanade ()
{
  // clean up
}

void
tld::cpu::LucasKanade::euclideanDistance (InputArray _point1, InputArray _point2, OutputArray _match)
{
  Mat point1_ = _point1.getMat();
  Mat point2_ = _point2.getMat();
  _match.create((int)nPts, 1, CV_32FC1, -1, true);
  Mat match_ = _match.getMat();
  float* match = (float*)match_.data;

  const Point2f* point1 = (const Point2f*)point1_.data;
  const Point2f* point2 = (const Point2f*)point2_.data;

  for (int i = 0; i < nPts; i++){
    match[i] = sqrt((point1[i].x - point2[i].x)*(point1[i].x - point2[i].x) +
        (point1[i].y - point2[i].y)*(point1[i].y - point2[i].y) );
  }
}

void
tld::cpu::LucasKanade::normCrossCorrelation(InputArray _points0, InputArray _points1, InputArray _status, OutputArray _match, const int winsize)
{
  Mat points0_ = _points0.getMat();
  Mat points1_ = _points1.getMat();
  Mat status_ = _status.getMat();
  _match.create((int)nPts, 1, CV_32FC1, -1, true);
  Mat match_ = _match.getMat();

  Size patchSize(win_size,win_size);

  Mat rec0( patchSize, IMG_I.type() );
  Mat rec1( patchSize, IMG_J.type() );
  Mat res( 1, 1, CV_32FC1 );

  const Point2f* points0 = (const Point2f*)points0_.data;
  const Point2f* points1 = (const Point2f*)points1_.data;
  const uchar* status = (const uchar*)status_.data;
  float* match = (float*)match_.data;

  for (int i = 0; i < nPts; i++){
    if (status[i] == 1){
      //Retrieves the pixel rectangle from an image with sub-pixel accuracy.
      getRectSubPix(IMG_I, patchSize, points0[i], rec0 );
      getRectSubPix(IMG_J, patchSize, points1[i], rec1 );
      //Compares a template against overlapped image regions.
      matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);
      match[i] = res.at<float>(0,0);
    } else {
      match[i] = 0.0;
    }
  }
  rec0.release();
  rec1.release();
  res.release();
}

void
tld::cpu::LucasKanade::lk( InputArray _imgI, InputArray _imgJ, InputArray _ptsI, InputArray _ptsJ, OutputArray _output )
{
  IMG_I = _imgI.getMat();
  IMG_J = _imgJ.getMat();
  Mat ptsI = _ptsI.getMat();
  Mat ptsJ = _ptsJ.getMat();

//  CV_Assert( (ptsI.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );
//  CV_Assert( (ptsJ.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  float nan = std::numeric_limits<float>::quiet_NaN();
//  double inf = std::numeric_limits<double>::infinity();

  nPts = ptsI.cols;

  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
  Size winSize(win_size,win_size);

  // Points
  Point2f pt;
  for (int i = 0; i < nPts; i++){
    pt = Point2f(ptsI.at<float>(0,i),ptsI.at<float>(1,i));
    points[0].push_back(pt); // template
//    vector<Point2f> tmp;
//    tmp.push_back(pt);
//    cornerSubPix( IMG_I, tmp, winSize, cvSize(-1,-1), termcrit);
//    points[0].push_back(tmp[0]); // template
    pt = Point2f(ptsJ.at<float>(0,i),ptsJ.at<float>(1,i));
    points[1].push_back(pt); // target
    pt = Point2f(ptsI.at<float>(0,i),ptsI.at<float>(1,i));
    points[2].push_back(pt); // forward-backward
  }

  vector<uchar> status;
  vector<float> err;
  int Level = 5;
//  calcOpticalFlowPyrLK(IMG_I, IMG_J, points[0], points[1], status, err, winSize, Level, termcrit, 0, CV_LKFLOW_INITIAL_GUESSES, 0.001);
  calcOpticalFlowPyrLK(IMG_I, IMG_J, points[0], points[1], status, err, winSize, Level, termcrit, 0, 0.001);
  status.clear();
  err.clear();
//  calcOpticalFlowPyrLK(IMG_J, IMG_I, points[1], points[2], status, err, winSize, Level, termcrit, 0, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY, 0.001);
  calcOpticalFlowPyrLK(IMG_J, IMG_I, points[1], points[2], status, err, winSize, Level, termcrit, 0, 0.001);

//  for( size_t i = 0; i < status.size(); i++ )
//   {
//     cout << (int)status[i]
//         << " "
//     << endl;
//   }

  vector<float> normCrosCorl;
  int Winsize = 10;
  normCrossCorrelation(points[0], points[1], status, normCrosCorl, Winsize);
  vector<float> fwdBack;
  euclideanDistance( points[0], points[2], fwdBack);

  // Output
//  _output.create(4, nPts, CV_32FC1);
  Mat outputLK = _output.getMat();
//  outputLK.setTo(Scalar(0.));
  for (int i = 0; i < nPts; i++){
    if (status[i] == 1){
      outputLK.at<float>(0,i) = (float) points[1][i].x;
      outputLK.at<float>(1,i) = (float) points[1][i].y;
      outputLK.at<float>(2,i) = (float) fwdBack[i];
      outputLK.at<float>(3,i) = (float) normCrosCorl[i];
    } else {
      outputLK.at<float>(0,i) = nan;
      outputLK.at<float>(1,i) = nan;
      outputLK.at<float>(2,i) = nan;
      outputLK.at<float>(3,i) = nan;
    }
  }

  points[0].clear();
  points[1].clear();
  points[2].clear();
  normCrosCorl.clear();
  fwdBack.clear();
  status.clear();
  err.clear();
}

#endif
