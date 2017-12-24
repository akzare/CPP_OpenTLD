/***********************************************************************
  $FILENAME    : Distance.hpp

  $TITLE       : Distance template functions

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Includes utility template funtions for computing distance
                 between two matrixes

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <opencv2/core/core.hpp>      // Basic OpenCV structures

//#include <stdio.h>
//#include <stdint.h>
//#include "math.h"

#include "Constants.h"

using namespace std;
using namespace cv;

template<class TEMPL>
void distanceRow(InputArray _in1, InputArray _in2, const int flag, OutputArray _dist)
{
  Mat in1_ = _in1.getMat();
  Mat in2_ = _in2.getMat();

  _dist.create( in1_.rows, in2_.rows, CV_32FC1 );
  Mat dist_ = _dist.getMat();

  switch (flag)
  {
  case 1 : // normalized correlation
  {
    double corr = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for(int i=0; i<in1_.rows; i++){
      for(int j=0; j<in2_.rows; j++){
        corr = 0.0; norm1 = 0.0; norm2 = 0.0;
        for(int jj=0; jj<in1_.cols; jj++){
          corr += in1_.at<TEMPL>(i,jj)*in2_.at<TEMPL>(j,jj);
          norm1 += in1_.at<TEMPL>(i,jj)*in1_.at<TEMPL>(i,jj);
          norm2 += in2_.at<TEMPL>(j,jj)*in2_.at<TEMPL>(j,jj);
        }
        dist_.at<float>(i,j) = (float)((corr / sqrt(norm1*norm2) + 1) / 2.0);
      }
    }
  }
  break;
  case 2 : // euclidean distance
  {
    double sum;
    for(int i=0; i<in1_.rows; i++){
      for(int j=0; j<in2_.rows; j++){
        sum = 0;
        for(int jj=0; jj<in1_.cols; jj++){
          sum += pow(in1_.at<TEMPL>(i,jj)-in2_.at<TEMPL>(j,jj), 2);
        }
        dist_.at<float>(i,j) = (float)sqrt(sum);
      }
    }
  }
  break;
  }

  return;
}

template<class TEMPL>
void distanceCol(InputArray _in1, InputArray _in2, const int flag, OutputArray _dist)
{
  Mat in1_ = _in1.getMat();
  Mat in2_ = _in2.getMat();

  _dist.create( in1_.cols, in2_.cols, CV_32FC1 );
  Mat dist_ = _dist.getMat();

  switch (flag)
  {
  case 1 : // normalized correlation
  {
    double corr = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for(int i=0; i<in1_.cols; i++){
      for(int j=0; j<in2_.cols; j++){
        corr = 0.0; norm1 = 0.0; norm2 = 0.0;
        for(int jj=0; jj<in1_.rows; jj++){
          corr += in1_.at<TEMPL>(jj,i)*in2_.at<TEMPL>(jj,j);
          norm1 += in1_.at<TEMPL>(jj,i)*in1_.at<TEMPL>(jj,i);
          norm2 += in2_.at<TEMPL>(jj,j)*in2_.at<TEMPL>(jj,j);
        }
        dist_.at<float>(i,j) = (float)((corr / sqrt(norm1*norm2) + 1) / 2.0);
      }
    }
  }
  break;
  case 2 : // euclidean distance
  {
    double sum;
    for(int i=0; i<in1_.cols; i++){
      for(int j=0; j<in2_.cols; j++){
        sum = 0;
        for(int jj=0; jj<in1_.rows; jj++){
          sum += pow(in1_.at<TEMPL>(jj,i)-in2_.at<TEMPL>(jj,j), 2);
        }
        dist_.at<float>(i,j) = (float)sqrt(sum);
      }
    }
  }
  break;
  default:
    assert(0);
    CV_Error( CV_BadDepth, "" );
  }
  return;
}

template<class TEMPL>
void distanceCol(InputArray _in, const int flag, OutputArray _dist)
{
  Mat in_ = _in.getMat();

  _dist.create( 1, (in_.cols)*(in_.cols-1)/2, CV_32FC1 );
  Mat dist_ = _dist.getMat();

  switch (flag)
  {
  case 1 : // normalized correlation
  {
    double corr = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    int cnt = 0;
    for(int i=0; i<in_.cols-1; i++){
      for(int j=i+1; j<in_.cols; j++){
        corr = 0; norm1 = 0; norm2 = 0;
        for(int jj=0; jj<in_.rows; jj++){
          corr += in_.at<TEMPL>(jj,i)*in_.at<TEMPL>(jj,j);
          norm1 += in_.at<TEMPL>(jj,i)*in_.at<TEMPL>(jj,i);
          norm2 += in_.at<TEMPL>(jj,j)*in_.at<TEMPL>(jj,j);
        }
        CV_Assert( cnt < dist_.cols );
        dist_.at<float>(0,cnt) = (float)((corr / sqrt(norm1*norm2) + 1) / 2.0);
        cnt++;
      }
    }
  }
  break;
  case 2 : // euclidean distance
  {
    double sum;
    int cnt = 0;
    for(int i=0; i<in_.cols-1; i++){
      for(int j=i+1; j<in_.cols; j++){
        sum = 0.0;
        for(int jj=0; jj<in_.rows; jj++){
          sum += pow(in_.at<TEMPL>(jj,i)-in_.at<TEMPL>(jj,j), 2);
        }
        CV_Assert( cnt < dist_.cols );
        dist_.at<float>(0,cnt) = (float)sqrt(sum);
        cnt++;
      }
    }
  }
  break;
  default:
    assert(0);
    CV_Error( CV_BadDepth, "" );
  }
  return;
}

#endif /* DISTANCE_H_ */
