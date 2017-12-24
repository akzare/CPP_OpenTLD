/***********************************************************************
  $FILENAME    : tldTracker.hpp

  $TITLE       : TLD tracker class definition

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Defines the TLD tracker class

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef TLD_CPU_OCL_H_
#define TLD_CPU_OCL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_OCL_
#include "opencv2/ocl/ocl.hpp"
#endif

#include <fstream>
#include <list>

#include "Constants.h"
#include "main.hpp"

using namespace std;
using namespace cv;
#ifdef USE_OCL_
using namespace cv::ocl;
#endif

typedef std::list<Mat> PX;
typedef std::list<Mat> PEX;
typedef std::list<Mat> NEX;

class trackerApp;

namespace tld
{
#ifndef USE_OCL_
  // CPU Version
  namespace cpu
  {

    class LucasKanade
    {
    public:
      LucasKanade ( InputArray img );
      virtual ~LucasKanade ();
      void lk( InputArray imgI, InputArray imgJ, InputArray ptsI, InputArray ptsJ, OutputArray out );

    private:
      //  void loadImageFromMatlab(const mxArray *mxImage, IplImage *image);
      void euclideanDistance (InputArray point1, InputArray point2, OutputArray match);
      void normCrossCorrelation(InputArray points0, InputArray points1, InputArray status, OutputArray match, const int winsize);
      int win_size;
      int nPts;
      //CvPoint2D32f* points[3];
      vector<Point2f> points[3];
      Mat IMG_I;
      Mat IMG_J;
      Mat PYR_I;
      Mat PYR_J;
    };

    class Fern
    {
    public:
      Fern ( );
      virtual ~Fern ();
      void init( InputArray img, InputArray grid, InputArray features, InputArray scales );
      void update( InputArray X, InputArray Y, const float Margin, const int bootstrap );
      void update( InputArray X, InputArray Y, const float Margin, const int bootstrap, InputArray idx );
      void evaluate( InputArray X, OutputArray out );
      void detect( InputArray img_gray, InputArray img_blur, const int maxBBox, const float minVar, OutputArray conf, OutputArray patt );

      void getPattern( InputArray img_gray, InputArray img_blur, InputArray idx, const float minVar, OutputArray pattern, OutputArray status );
    private:
      void integralImage(InputArray image);
      void update_(InputArray x, const int col, const int C, const int N);
      float measureForest(InputArray idx, const int col);
      int measureTreeOffset (InputArray img, const int idx_bbox, const int idx_tree);
      float bboxVarOffset(const int col);
      float measureBBoxOffset (InputArray blur, const int idx_bbox, const float minVar, OutputArray tPatt, const int col);
      float randdouble ( );
      void saveWEIGHT(void);
      void saveOFF(const int BBOX_a_x, const int BBOX_a_y, const int BBOX_d_x, const int BBOX_d_y, const int OFF_0_x, const int OFF_0_y, const int OFF_1_x, const int OFF_1_y);
#ifdef PROFILING_ON_
      clock_t start, stop;
      float elapsedTime;
      time_t nowis;
      tm* localtm;
      char timeBuffer[256];
#endif
      float thrN;
      int nTREES;
      int nFEAT;
      int nSCALE;
      int iHEIGHT;
      int iWIDTH;
      Mat BBOX;
      Mat OFF;
      Mat integral_img;
      Mat integral_img2;
      Mat WEIGHT;
      Mat nP;
      Mat nN;
      int nBIT; // number of bits per feature
      int num_features_bit;
    };

    class tracker {
    public:
      tracker( trackerApp* );
      virtual ~tracker();
      void process();

    public:
      // Median frame by frame Tracker & Detector
      struct{
        Rect bbox_rect;
        Mat curBBox;
        Mat prvBBox;
        float curConf;
        float prvConf;
        int curSize;
        int prvSize;
        int prvValid;
        int curValid;
        Mat xFI;
        Mat xFII;
        Mat xFJ;
        Mat xFJJ;
        int numM; // numMxnumN grid of points within BBox
        int numN;
        int numMN;
        Mat idxF_;
        int* idxF;
        Mat tBB;
        int tValid;
        float tConf;
        Mat dBB;
        Mat dConf;
        int DTLen;
        int DT;
        int DTNum;
        int TR;
#ifdef FERN_OPENCV_ON_
        Mat object;
        bool DTFound;
        vector<int> pairs;
        vector<Point2f> dst_corners;
        vector<KeyPoint> objKeypoints, imgKeypoints;
#endif
      }TRDT;

    private:
      void init ( );
      void initFirstFrame ( );
      void display ( );
      void generateFeatures ( const int nTREES, const int nFEAT, OutputArray features );
      void generatePositiveData ( InputArray overlap, const int flag, OutputArray pX, OutputArray pEx, OutputArray bbP );
      void generateNegativeData ( InputArray overlap, OutputArray nX, OutputArray nEx );
      void splitNegativeData ( InputArray nX, InputArray nEx, OutputArray nX1, OutputArray nX2, OutputArray nEx1, OutputArray nEx2 );
      void trainNearestNeighbor ( InputArray pEx, InputArray nEx );
      void NearestNeighbor_1D (InputArray x, const int col, const int pex_isempty, const int nex_isempty, OutputArray isin, float &conf1, float &conf2);
      void NearestNeighbor ( InputArray x, OutputArray isin, OutputArray conf1, OutputArray conf2 );
      void getPattern ( InputArray bb, OutputArray pattern );
      void getPattern_1D ( InputArray bb, const int col, OutputArray pattern );
      void patch2Pattern ( InputArray patch, const int *patchsize, OutputArray pattern, const int col);
      void processFrame ( );
      void tracking ( );
      void detection ( );
      void learning ( );
#ifdef FERN_OPENCV_ON_
      void ObjectDetectorTrain( );
      void ObjectDetector( );
#endif
#ifdef PROFILING_ON_
      clock_t start, stop;
      float elapsedTime;
      time_t nowis;
      tm* localtm;
      char timeBuffer[256];
#endif
      trackerApp* m_pApp;
      Mat tmpBlurImg;
      bool initDone_;
      int count_;
      Size imgsize;

    public:
#ifdef PROFILING_ON_
      ofstream profilingLogFile;
#endif
      // Fern DETECTOR
      struct {
        Mat grid;
        int nGrid; // grid length
        Mat scales;
        struct {
          Mat x;
          int type;
        }features;
        // Temporal structures
        struct {
          Mat conf;
          Mat patt;
        }tmp; // temporary storage for confidence and pattern calculations
        struct{
          Mat bb;    // bounding boxes
          Mat patt;  // corresponding codes of the Ensemble Classifier
          Mat idx;   // indexes of detected bounding boxes within the scanning grid
          Mat conf1; // Relative Similarity (for final nearest neighbor classifier)
          Mat conf2; // Conservative Similarity (for integration with tracker)
          Mat isin;  // detected (isin=1) or rejected (isin=0) by nearest neighbor classifier
          Mat patch; // Corresponding patches
          int num_dt;
        }dt;
      }detector;

    private:
      // TRAINER
      struct {
        Mat X; // training data for fern
        Mat pEx; // training data for NearestNeighbor
        Mat Y;
        Mat nEx;
        float var; // Variance threshold
        Mat pex;
        Mat nex;
      }trainer;
      Mat patchPatt1D;
      Mat isin_NN1D;
      LucasKanade *lucaskanade;
      Fern *fern;
#ifdef FERN_OPENCV_ON_
      LDetector ldetector;
      PlanarObjectDetector detector_;
      int blurKSize;
      double sigma;
#endif

    };
  }
#endif


#ifdef USE_OCL_

  // OpenCL (OCL) Version
  namespace ocl
  {
    class tracker {
    public:
      tracker( trackerApp* );
      virtual ~tracker();
    };
  }
#endif


}

#endif /* TLD_CPU_OCL_H_ */
