/***********************************************************************
  $FILENAME    : Statistics.hpp

  $TITLE       : Utility statistics computation template functions

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Implements some utility template functions for the 
                 statistics computation

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef STATISTICS_H_
#define STATISTICS_H_

#include "opencv2/gpu/gpu.hpp"

#include "Constants.h"

using namespace std;
using namespace cv;

template<class TEMPL>
void mean(InputArray _in, OutputArray _mean)
{
  Mat in_ = _in.getMat();

  if (in_.cols > 1 && in_.rows > 1){
    Mat sum(1, in_.cols, CV_64FC1);
    _mean.create( 1, in_.cols, CV_64FC1 );
    Mat mean = _mean.getMat();
    for(int j=0; j<in_.cols; j++){
      sum.at<double>(0,j) = 0;
      for(int i=0; i<in_.rows; i++){
        sum.at<double>(0,j) += in_.at<TEMPL>(i,j);
      }
    }
    for(int j=0; j<in_.cols; j++){
      mean.at<double>(0,j) = sum.at<double>(0,j)/in_.rows;
    }
  } else if (in_.rows == 1){
    double sum = 0;
    _mean.create( 1, 1, CV_64FC1 );
    Mat mean = _mean.getMat();
    for(int j=0; j<in_.cols; j++){
      sum += in_.at<TEMPL>(0,j);
    }
    mean.at<double>(0,0) = sum/in_.cols;
  } else{ // in_->cols = 1
    double sum = 0;
    _mean.create( 1, 1, CV_64FC1 );
    Mat mean = _mean.getMat();
    for(int j=0; j<in_.rows; j++){
      sum += in_.at<TEMPL>(j,0);
    }
    mean.at<double>(0,0) = sum/in_.rows;
  }
}

template<class TEMPL>
void median(InputArray _in, OutputArray _median)
{
  Mat in_ = _in.getMat();

  Mat temp = in_.clone();
  double temp_ = 0;

  if (in_.cols > 1 && in_.rows > 1){
    _median.create( 1, in_.cols, CV_64FC1 );
    Mat median = _median.getMat();
    for(int ii=0; ii<temp.cols; ii++){
      for(int i=0; i<temp.rows; i++){
        for(int j=i+1; j<temp.rows; j++){
          if(temp.at<TEMPL>(i,ii) > temp.at<TEMPL>(j,ii)){
            temp_ = temp.at<TEMPL>(j,ii);
            temp.at<TEMPL>(j,ii) = temp.at<TEMPL>(i,ii);
            temp.at<TEMPL>(i,ii) = temp_;
          }
        }
      }
    }
    for(int ii=0; ii<temp.cols; ii++){
      if (temp.rows%2 == 0){
        median.at<double>(0,ii) = (temp.at<TEMPL>(temp.rows/2,ii) + temp.at<TEMPL>(temp.rows/2-1,ii))/2;
      }else{
        median.at<double>(0,ii) = temp.at<TEMPL>(temp.rows/2,ii);
      }
    }
  } else if (in_.rows == 1){
    _median.create( 1, 1, CV_64FC1 );
    Mat median = _median.getMat();

    for(int i=0; i<temp.cols; i++){
      for(int j=i+1; j<temp.cols; j++){
        if(temp.at<TEMPL>(0,i) > temp.at<TEMPL>(0,j)){
          temp_ = temp.at<TEMPL>(0,j);
          temp.at<TEMPL>(0,j) = temp.at<TEMPL>(0,i);
          temp.at<TEMPL>(0,i) = temp_;
        }
      }
    }
    if (temp.cols%2 == 0){
      median.at<double>(0,0) = (temp.at<TEMPL>(0,temp.cols/2) + temp.at<TEMPL>(0,temp.cols/2-1))/2;
    }else{
      median.at<double>(0,0) = temp.at<TEMPL>(0,temp.cols/2);
    }
  } else{ // in_->cols = 1
    _median.create( 1, 1, CV_64FC1 );
    Mat median = _median.getMat();

    for(int i=0; i<temp.rows; i++){
      for(int j=i+1; j<temp.rows; j++){
        if(temp.at<TEMPL>(i,0) > temp.at<TEMPL>(j,0)){
          temp_ = temp.at<TEMPL>(j,0);
          temp.at<TEMPL>(j,0) = temp.at<TEMPL>(i,0);
          temp.at<TEMPL>(i,0) = temp_;
        }
      }
    }
    if (temp.rows%2 == 0){
      median.at<double>(0,0) = (temp.at<TEMPL>(temp.rows/2,0) + temp.at<TEMPL>(temp.rows/2-1,0))/2;
    }else{
      median.at<double>(0,0) = temp.at<TEMPL>(temp.rows/2,0);
    }
  }
  temp.release();
}

template<class TEMPL>
float median1D(InputArray _in, const int rowcol, const int num)
{
  Mat in_ = _in.getMat();

  Mat temp;
  float median = 0;
  if (rowcol == 0){ // rwocol = 0 -> row
    temp.create(1, in_.cols, CV_32SC1);
    for (int i=0; i<in_.cols; i++){
      temp.at<TEMPL>(0,i) = in_.at<TEMPL>(num,i);
    }
  }else{ // rwocol = 1 -> col
    temp.create(1, in_.rows, CV_32SC1);
    for (int i=0; i<in_.rows; i++){
      temp.at<TEMPL>(0,i) = in_.at<TEMPL>(i,num);
    }
  }

  for(int i=0; i<temp.cols; i++){
    for(int j=i+1; j<temp.cols; j++){
      if(temp.at<TEMPL>(0,i) > temp.at<TEMPL>(0,j)){
        float temp_ = temp.at<TEMPL>(0,j);
        temp.at<TEMPL>(0,j) = temp.at<TEMPL>(0,i);
        temp.at<TEMPL>(0,i) = temp_;
      }
    }
  }

  if (temp.cols%2 == 0){
    median = (temp.at<TEMPL>(0,temp.cols/2) + temp.at<TEMPL>(0,temp.cols/2-1))/2.;
  }else{
    median = temp.at<TEMPL>(0,temp.cols/2);
  }
  temp.release();
  return median;
}

template<class TEMPL>
void variance(InputArray _in, OutputArray _var )
{
  Mat in_ = _in.getMat();

  if (in_.cols > 1 && in_.rows > 1){
    _var.create( 1, in_.cols, CV_64FC1 );
    Mat var = _var.getMat();
    Mat sum(1, in_.cols, CV_64FC1);
    Mat mean(1, in_.cols, CV_64FC1);
    for(int j=0; j<in_.cols; j++){
      sum.at<double>(0,j) = 0;
      for(int i=0; i<in_.rows; i++){
        sum.at<double>(0,j) += in_.at<TEMPL>(i,j);
      }
    }
    for(int j=0; j<in_.cols; j++){
      mean.at<double>(0,j) = sum.at<double>(0,j)/in_.rows;
    }
    for(int j=0; j<in_.cols; j++){
      sum.at<double>(0,j) = 0;
      for(int i=0; i<in_.rows; i++){
        sum.at<double>(0,j) += pow((in_.at<TEMPL>(i,j)-mean.at<double>(0,j)),2);
      }
    }
    for(int j=0; j<in_.cols; j++){
      var.at<double>(0,j) = sum.at<double>(0,j)/(in_.rows-1);
    }
    sum.release();
    mean.release();
  } else if (in_.rows == 1){
    double sum = 0, mean = 0;
    _var.create( 1, 1, CV_64FC1 );
    Mat var = _var.getMat();
    for(int j=0; j<in_.cols; j++){
      sum += in_.at<TEMPL>(0,j);
    }
    mean = sum/in_.cols;
    sum = 0;
    for(int j=0; j<in_.cols; j++){
      sum += pow((in_.at<TEMPL>(0,j)-mean),2);
    }
    var.at<double>(0,0) = sum/(in_.cols-1);
  } else{ // in_->cols = 1
    double sum = 0, mean = 0;
    _var.create( 1, 1, CV_64FC1 );
    Mat var = _var.getMat();
    for(int j=0; j<in_.rows; j++){
      sum += in_.at<TEMPL>(j,0);
    }
    mean = sum/in_.rows;
    sum = 0;
    for(int j=0; j<in_.rows; j++){
      sum += pow((in_.at<TEMPL>(j,0)-mean),2);
    }
    var.at<double>(0,0) = sum/(in_.rows-1);
  }
}

template<class TEMPL>
double variance1D(InputArray  _in)
{
  Mat in_ = _in.getMat();

  double var = 0.;
  double sum = 0., mean = 0.;
  // in_->cols = 1
  if (in_.cols == 1){
    for(int j=0; j<in_.rows; j++){
      sum += in_.at<TEMPL>(j,0);
    }
    mean = sum/in_.rows;
    sum = 0;
    for(int j=0; j<in_.rows; j++){
      sum += pow((in_.at<TEMPL>(j,0)-mean),2);
    }
    var = sum/(in_.rows-1);
  }else{
    for(int j=0; j<in_.cols; j++){
      sum += in_.at<TEMPL>(0,j);
    }
    mean = sum/in_.cols;
    sum = 0;
    for(int j=0; j<in_.cols; j++){
      sum += pow((in_.at<TEMPL>(0,j)-mean),2);
    }
    var = sum/(in_.cols-1);
  }
  return var;
}

template<class TEMPL>
void stdDev( InputArray _in, OutputArray _var )
{
  Mat in_ = _in.getMat();

  variance<TEMPL>( in_, _var );
  Mat var = _var.getMat();
  if (in_.cols > 1 && in_.rows > 1){
    for(int j=0; j<in_.cols; j++){
      var.at<double>(0,j) = sqrt(var.at<double>(0,j));
    }
  } else{ // in_->cols = 1 or in_->rows = 1
    var.at<double>(0,0) = sqrt(var.at<double>(0,0));
  }
}

template<typename TEMPL>
void uniqueMat( InputArray _in, OutputArray _out )
{
  Mat in = _in.getMat();

//  CV_Assert( (in.type() & CV_MAT_DEPTH_MASK) == CV_32SC1 );

  Mat tmp1 = in.clone();
  Mat tmp_ = tmp1.reshape(0,in.rows*in.cols); // channel=unchanged, rows=in->rows*in->cols
  TEMPL* tmp = (TEMPL*)tmp_.data;

  int size = tmp_.rows;
  int temp;
  for (int i = 0; i < size-1; i++){
    for (int j = i+1; j < size; j++){
      if (tmp[j] < tmp[i]){
        temp = tmp[j];
        tmp[j] = tmp[i];
        tmp[i] = temp;
      }
    }
  }

  for (int i = 0; i < size-1; i++){
    if (tmp[i] == tmp[i+1]){
      for (int k = i+1; k < size-1; k++){
        tmp[k] = tmp[k+1];
      }
      size--;
      i--;
    }
//    for (int ii=0; ii<size; ii++){
//      ACE_DEBUG(( LM_DEBUG, ACE_TEXT("%d "), Coord_INT(tmp, ii, 0) ));
//    }
//    ACE_DEBUG(( LM_DEBUG, ACE_TEXT("\n") ));
  }

  _out.create( size, 1, in.type() );
  Mat out = _out.getMat();

  TEMPL* src = (TEMPL *) (tmp_.data);
  TEMPL* dest = (TEMPL *) (out.data);
  memcpy ( dest, src, size * sizeof(TEMPL) );

//  int L = 0; // X0
//  int T = 0; // Y0
//  TEMPL* src = (TEMPL *) (tmp.data+ L*sizeof(TEMPL) + T*tmp.cols*sizeof(TEMPL));
//  TEMPL* dest = (TEMPL *) (out.data);
//  int x = out.cols * sizeof(TEMPL);
//  for(int y=0; y<out.rows; y++) {// row : height
//    memcpy ( dest, src, x );
//    src += tmp.cols * sizeof(TEMPL);
//    dest += x;
//  }

//  for (int j = 0; j < out.rows; j++){
//    out.at<TEMPL>(j,0) = tmp.at<TEMPL>(j,0);
//  }
  tmp1.release();
}

template<typename TEMPL>
void uniqueMat( InputArray _in, OutputArray _b, OutputArray _m, OutputArray _n )
{
  Mat in = _in.getMat();

//  CV_Assert( (in.type() & CV_MAT_DEPTH_MASK) == CV_32SC1 );

  Mat tmp1 = in.clone();
  Mat tmp_ = tmp1.reshape(0,in.rows*in.cols); // channel=unchanged, rows=in->rows*in->cols
  TEMPL* tmp = (TEMPL*)tmp_.data;

  int size = tmp_.rows;
  int temp;
  for (int i = 0; i < size-1; i++){
    for (int j = i+1; j < size; j++){
      if (tmp[j] < tmp[i]){
        temp = tmp[j];
        tmp[j] = tmp[i];
        tmp[i] = temp;
      }
    }
  }

  for (int i = 0; i < size-1; i++){
    if (tmp[i] == tmp[i+1]){
      for (int k = i+1; k < size-1; k++){
        tmp[k] = tmp[k+1];
      }
      size--;
      i--;
    }
//    for (int ii=0; ii<size; ii++){
//      ACE_DEBUG(( LM_DEBUG, ACE_TEXT("%d "), Coord_INT(tmp, ii, 0) ));
//    }
//    ACE_DEBUG(( LM_DEBUG, ACE_TEXT("\n") ));
  }

  _b.create( size, 1, in.type() );
  Mat b = _b.getMat();
  TEMPL* src = (TEMPL *) (tmp_.data);
  TEMPL* dest = (TEMPL *) (b.data);
  memcpy ( dest, src, size * sizeof(TEMPL) );
//  for (int j = 0; j < b.rows; j++){
//    b.at<TEMPL>(j,0) = tmp.at<TEMPL>(j,0);
//  }

  in.copyTo(tmp1);
  tmp_ = tmp1.reshape(0,in.rows*in.cols); // channel=unchanged, rows=in->rows*in->cols

  _m.create( size, 1, in.type() );
  Mat m = _m.getMat();
  for (int j = 0; j < b.rows; j++){
    for (int i = tmp_.rows-1; i >= 0; i--){
      if (b.at<TEMPL>(j,0) == tmp[i]){
        m.at<TEMPL>(j,0) = i;
        i = -1;
      }
    }
  }

  _n.create( tmp_.rows, 1, in.type() );
  Mat n = _n.getMat();
  for (int j = 0; j < n.rows; j++){
    for (int i = 0; i < b.rows; i++){
      if (b.at<TEMPL>(i,0) == tmp[j]){
        n.at<TEMPL>(j,0) = i;
        i = b.rows;
      }
    }
  }
  tmp1.release();
}

template<class TEMPL>
void randomPerm( const int n, TEMPL* perm )
{
  int i, j, t;
  srand ( time(0) );

  for(i=0; i<n; i++)
    perm[i] = i;
  for(i=0; i<n; i++) {
    j = rand()%(n-i)+i;
    t = perm[j];
    perm[j] = perm[i];
    perm[i] = t;
  }
}

template<class TEMPL>
void randomValues( InputArray _in, const int _k, OutputArray _out )
{
  Mat in = _in.getMat();

  int k = _k;

  // Randomly selects 'k' values from vector 'in'.
//  CV_Assert( (in.type() & CV_MAT_DEPTH_MASK) == CV_32SC1 );
  CV_Assert( in.rows == 1 );
  CV_Assert( k < in.cols );

  int N = in.cols; // 12024

  if (k == 0){ // 100
    return;
  }

  if (k > N){
    k = N;
  }

  if ((double)(k*1.0/N) < 0.0001){
    Mat i2(1, k, in.type());
    for (int i=0; i<k; i++){
      double randomize = rand()/(double(RAND_MAX)+1);
      i2.at<TEMPL>(0,i) = (TEMPL)ceil(N*randomize);
    }
    uniqueMat<TEMPL>(i2, _out);
    //i1 = unique(ceil(N*rand(1,k)));
    Mat out_ = _out.getMat();

    // Changes shape of matrix/image without copying data
    out_ = out_.reshape(0,1); // channel=unchanged, rows=1
    //out = in(:,i1);
    _out.getMat() = out_;
    i2.release();
    return;
  }else{
    int* i2 = new int[N];
    randomPerm<int>(N, i2);
    //i2 = randperm(N);

    int temp;
    Mat i1_(1, k, CV_32SC1);
    int* i1 = (int*)i1_.data;

    for (int i=0; i<k; i++){
      i1[i] = i2[i];
    }
    for (int i=0; i<k-1; i++){
      for (int j = i+1; j < k; j++){
        if (i1[i] > i1[j]){ // sort
          temp = i1[i];
          i1[i] = i1[j];
          i1[j] = temp;
        }
      }
    }
    _out.create( 1, k, in.type() );
    Mat out_ = _out.getMat();

    for (int i=0; i<k; i++){
      out_.at<TEMPL>(0,i) = in.at<TEMPL>(0,i1[i]);
    }
    //out = in(:,sort(i2(1:k)));
    delete []i2;
    i1_.release();
    return;
  }
}

template<class TEMPL>
void nTuples ( InputArray _in1, InputArray _in2, OutputArray _out )
{
  Mat in1_ = _in1.getMat();
  TEMPL* in1 = (TEMPL*)in1_.data;
  Mat in2_ = _in2.getMat();
  TEMPL* in2 = (TEMPL*)in2_.data;

  CV_Assert( in1_.rows == 1 && in2_.rows == 1 );
  // Computes all possible ntupples.
  _out.create( 2, in1_.cols*in2_.cols, in1_.type() ); // CV_32FC1
  Mat out = _out.getMat();

  int col = 0;
  for (int j=0; j<in1_.cols; j++){
    for (int i = 0; i<in2_.cols; i++){
      out.at<TEMPL>(0,col) = in1[j];
      col++;
    }
  }
  col = 0;
  for (int i = 0; i<in1_.cols; i++){
    for (int j=0; j<in2_.cols; j++){
      out.at<TEMPL>(1,col) = in2[j];
      col++;
    }
  }
}

#endif /* STATISTICS_H_ */
