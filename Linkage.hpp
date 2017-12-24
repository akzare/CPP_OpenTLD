/***********************************************************************
  $FILENAME    : Linkage.hpp

  $TITLE       : Utility linkage template functions

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

************************************************************************/

#ifndef LINKAGE_H_
#define LINKAGE_H_

#include <iostream>

#include "Constants.h"
#include "Statistics.hpp"

using namespace std;
using namespace cv;

#define ISNAN(a) (a != a)
#define MAX_NUM_OF_INPUT_ARG_FOR_PDIST 50

template<class TEMPL>
void linkageTEMPLATE(
    int numInArg_,
    InputArray _in,
    const char *method,
    int numOutArg_,
    OutputArray _out,
    int call_pdist,
    TEMPL classDummy
)
{
  Mat in_ = _in.getMat();

  enum method_types
  {single,complete,average,weighted,centroid,median,ward} method_key;
  static TEMPL  inf;
  int        m,m2,m2m3,m2m1,n,i,j,bn,bc,bp,p1,p2,q,q1,q2,h,k,l,g;
  int        nk,nl,ng,nkpnl,sT,N;
  int        *obp,*scl,*K,*L;
  TEMPL      *y,*yi,*s,*b1,*b2,*T;
  TEMPL      t1,t2,t3,rnk,rnl;
  int        uses_scl = false,  no_squared_input = true;

  if       ( strcmp(method,"si") == 0 ) method_key = single;
  else if  ( strcmp(method,"co") == 0 ) method_key = complete;
  else if  ( strcmp(method,"av") == 0 ) method_key = average;
  else if  ( strcmp(method,"we") == 0 ) method_key = weighted;
  else if  ( strcmp(method,"ce") == 0 ) method_key = centroid;
  else if  ( strcmp(method,"me") == 0 ) method_key = median;
  else if  ( strcmp(method,"wa") == 0 ) method_key = ward;
  else
    cout << "\nError: stats:linkage:UnknownLinkageMethod.\n"
         << "\tUnknown linkage method.\n"
         << endl;

  if ((method_key==centroid) || (method_key==median) || (method_key==ward))
    no_squared_input = false;
  else
    no_squared_input = true;

  if (call_pdist) {
  } else {
    n  = in_.cols;
    m = (int) ceil(sqrt(2*(double)n));

    yi = (TEMPL*) in_.data;

    y = new TEMPL[n];

    if (no_squared_input) {
      memcpy(y,yi,n * sizeof(TEMPL));
    } else {
      for (i=0; i<n; i++) y[i] = yi[i] * yi[i];
    }
  }

  bn   = m-1;
  m2   = m * 2;
  m2m3 = m2 - 3;
  m2m1 = m2 - 1;

  inf = std::numeric_limits<TEMPL>::infinity();

  Mat out;
  Mat out_;
  if ((in_.type() & CV_MAT_DEPTH_MASK) == CV_32FC1){
    _out.create( bn, 3, CV_32FC1 );
    out_ = _out.getMat();
    out.create(bn, 3, CV_32FC1);
  }else if ((in_.type() & CV_MAT_DEPTH_MASK) == CV_32SC1){
    _out.create( bn, 3, CV_32SC1 );
    out_ = _out.getMat();
    out.create(bn, 3, CV_32SC1);
  }else{
    _out.create( bn, 3, CV_64FC1 );
    out_ = _out.getMat();
    out.create(bn, 3, CV_64FC1);
  }

  b1 = (TEMPL*)out.data;
  b2 = b1 + bn;
  s  = b2 + bn;

  if      (m>1023) N = 512;
  else if (m>511)  N = 256;
  else if (m>255)  N = 128;
  else if (m>127)  N = 64;
  else if (m>63)   N = 32;
  else             N = 16;
  if (method_key == single) N = N >> 2;

  T = new TEMPL[N];
  K = new int[N];
  L = new int[N];

  obp = new int[m];
  switch (method_key) {
  case average:
  case centroid:
  case ward:
    uses_scl = true;
    scl = new int[m];
    for (i=0; i<m; obp[i]=i, scl[i++]=1);
    break;
  default:
    for (i=0; i<m; i++) obp[i]=i;
  }


  sT = 0;  t3 = inf;

  for (bc=0,bp=m;bc<bn;bc++,bp++){

    for (h=0;((T[h]<t3) && (h<sT));h++);
    sT = h; t3 = inf;
    if (sT==0) {
      for (h=0; h<N; T[h++]=inf);
      p1 = ((m2m1 - bc) * bc) >> 1;
      for (j=bc; j<m; j++) {
        for (i=j+1; i<m; i++) {
          t2 = y[p1++];
          if (t2 <= T[N-1]) {
            for (h=N-1; ((t2 <= T[h-1]) && (h>0)); h--) {
              T[h]=T[h-1];
              K[h]=K[h-1];
              L[h]=L[h-1];
            }
            T[h] = t2;
            K[h] = j;
            L[h] = i;
            sT++;
          }
        }
      }
      if (sT>N) sT=N;
    }

    if (sT==0) break;


    k=K[0]; l=L[0]; t1=T[0];

    for (h=0,i=1;i<sT;i++) {
      if ( (k!=K[i]) && (l!=L[i]) && (l!=K[i]) && (k!=L[i]) ) {
        T[h]=T[i];
        K[h]=K[i];
        L[h]=L[i];
        if (bc==K[h]) {
          if (k>L[h]) {
            K[h] = L[h];
            L[h] = k;
          }
          else K[h] = k;
        }
        h++;
      }
    }
    sT=h;

    if (obp[k]<obp[l]) {
      *b1++ = (TEMPL) (obp[k]);
      *b2++ = (TEMPL) (obp[l]);
    } else {
      *b1++ = (TEMPL) (obp[l]);
      *b2++ = (TEMPL) (obp[k]);
    }
    *s++ = (no_squared_input) ? t1 : sqrt(t1);

    obp[k] = obp[bc];
    obp[l] = bp;

    q1 = bn - k - 1;
    q2 = bn - l - 1;

    p1 = (((m2m1 - bc) * bc) >> 1) + k - bc - 1;
    p2 = p1 - k + l;

    if (uses_scl) {
      nk     = scl[k];
      nl     = scl[l];
      nkpnl  = nk + nl;

      scl[k] = scl[bc];
      scl[l] = nkpnl;

    }

    switch (method_key) {
    case centroid:
      t1 = t1 * ((TEMPL) nk * (TEMPL) nl) / ((TEMPL) nkpnl * (TEMPL) nkpnl);
    case average:
      rnk = (TEMPL) nk / (TEMPL) nkpnl;
      rnl = (TEMPL) nl / (TEMPL) nkpnl;
      break;
    case median:
      t1 = t1/4;
    }

    switch (method_key) {
    case average:
      for (q=bn-bc-1; q>q1; q--) {
        t2 = y[p1] * rnk + y[p2] * rnl;
        if (t2 < t3) t3 = t2 ;
        y[p2] = t2;
        p1 = p1 + q;
        p2 = p2 + q;
      }
      p1++;
      p2 = p2 + q;
      for (q=q1-1;  q>q2; q--) {
        t2 = y[p1] * rnk + y[p2] * rnl;
        if (t2 < t3) t3 = t2 ;
        y[p2] = t2;
        p1++;
        p2 = p2 + q;
      }
      p1++;
      p2++;
      for (q=q2+1; q>0; q--) {
        t2 = y[p1] * rnk + y[p2] * rnl;
        if (t2 < t3) t3 = t2 ;
        y[p2] = t2;
        p1++;
        p2++;
      }
      break;

    case single:
      for (q=bn-bc-1; q>q1; q--) {
        if (y[p1] < y[p2]) y[p2] = y[p1];
        else if (ISNAN(y[p2])) y[p2] = y[p1];
        if (y[p2] < t3)    t3 = y[p2];
        p1 = p1 + q;
        p2 = p2 + q;
      }
      p1++;
      p2 = p2 + q;
      for (q=q1-1;  q>q2; q--) {
        if (y[p1] < y[p2]) y[p2] = y[p1];
        else if (ISNAN(y[p2])) y[p2] = y[p1];
        if (y[p2] < t3)    t3 = y[p2];
        p1++;
        p2 = p2 + q;
      }
      p1++;
      p2++;
      for (q=q2+1; q>0; q--) {
        if (y[p1] < y[p2]) y[p2] = y[p1];
        else if (ISNAN(y[p2])) y[p2] = y[p1];
        if (y[p2] < t3)    t3 = y[p2];
        p1++;
        p2++;
      }
      break;

      case complete:
        for (q=bn-bc-1; q>q1; q--) {
          if (y[p1] > y[p2]) y[p2] = y[p1];
          else if (ISNAN(y[p2])) y[p2] = y[p1];
          if (y[p2] < t3)    t3 = y[p2];
          p1 = p1 + q;
          p2 = p2 + q;
        }
        p1++;
        p2 = p2 + q;
        for (q=q1-1;  q>q2; q--) {
          if (y[p1] > y[p2]) y[p2] = y[p1];
          else if (ISNAN(y[p2])) y[p2] = y[p1];
          if (y[p2] < t3)    t3 = y[p2];
          p1++;
          p2 = p2 + q;
        }
        p1++;
        p2++;
        for (q=q2+1; q>0; q--) {
          if (y[p1] > y[p2]) y[p2] = y[p1];
          else if (ISNAN(y[p2])) y[p2] = y[p1];
          if (y[p2] < t3)    t3 = y[p2];
          p1++;
          p2++;
        }
        break;

      case weighted:
        for (q=bn-bc-1; q>q1; q--) {
          t2 = (y[p1] + y[p2])/2;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1 = p1 + q;
          p2 = p2 + q;
        }
        p1++;
        p2 = p2 + q;
        for (q=q1-1;  q>q2; q--) {
          t2 = (y[p1] + y[p2])/2;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2 = p2 + q;
        }
        p1++;
        p2++;
        for (q=q2+1; q>0; q--) {
          t2 = (y[p1] + y[p2])/2;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2++;
        }
        break;

      case centroid:
        for (q=bn-bc-1; q>q1; q--) {
          t2 = y[p1] * rnk + y[p2] * rnl - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1 = p1 + q;
          p2 = p2 + q;
        }
        p1++;
        p2 = p2 + q;
        for (q=q1-1;  q>q2; q--) {
          t2 = y[p1] * rnk + y[p2] * rnl - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2 = p2 + q;
        }
        p1++;
        p2++;
        for (q=q2+1; q>0; q--) {
          t2 = y[p1] * rnk + y[p2] * rnl - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2++;
        }
        break;

      case median:
        for (q=bn-bc-1; q>q1; q--) {
          t2 = (y[p1] + y[p2])/2 - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1 = p1 + q;
          p2 = p2 + q;
        }
        p1++;
        p2 = p2 + q;
        for (q=q1-1;  q>q2; q--) {
          t2 = (y[p1] + y[p2])/2 - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2 = p2 + q;
        }
        p1++;
        p2++;
        for (q=q2+1; q>0; q--) {
          t2 = (y[p1] + y[p2])/2 - t1;
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2++;
        }
        break;

      case ward:
        for (q=bn-bc-1,g=bc; q>q1; q--) {
          ng = scl[g++];
          t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1 = p1 + q;
          p2 = p2 + q;
        }
        g++;
        p1++;
        p2 = p2 + q;
        for (q=q1-1;  q>q2; q--) {
          ng = scl[g++];
          t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2 = p2 + q;
        }
        g++;
        p1++;
        p2++;
        for (q=q2+1; q>0; q--) {
          ng = scl[g++];
          t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
          if (t2<t3) t3=t2;
          y[p2] = t2;
          p1++;
          p2++;
        }
        break;

    }

    if (k!=bc) {
      q1 = bn - k;

      p1 = (((m2m3 - bc) * bc) >> 1) + k - 1;
      p2 = p1 - k + bc + 1;

      for (q=bn-bc-1; q>q1; q--) {
        p1 = p1 + q;
        y[p1] = y[p2++];
      }
      p1 = p1 + q + 1;
      p2++;
      for ( ; q>0; q--) {
        y[p1++] = y[p2++];
      }
    }
  }

  for (;bc<bn;bc++,bp++) {
    k=bc; l=bc+1;
    if (obp[k]<obp[l]) {
      *b1++ = (TEMPL) (obp[k]);
      *b2++ = (TEMPL) (obp[l]);
    } else {
      *b1++ = (TEMPL) (obp[l]);
      *b2++ = (TEMPL) (obp[k]);
    }
    obp[l] = bp;
    *s++ = std::numeric_limits<TEMPL>::quiet_NaN();
  }

  if (!call_pdist) delete []y;
  if (uses_scl) delete []scl;
  delete []obp;
  delete []L;
  delete []K;
  delete []T;

  b1 = (TEMPL *) out.data;
  for (int j=0; j<out_.cols; j++){
    for (int i=0; i<out_.rows; i++){
      out_.at<TEMPL>(i,j) = *b1;
      b1++;
    }
  }

}

template<class TEMPL>
void linkage(
    InputArray _in,
    const char *method,
    OutputArray _out
)
{
  Mat in = _in.getMat();

  CV_Assert( in.rows == 1 );
  int numInArg_ = 2;
  int numOutArg_ = 1;

  if((numInArg_!=2) && (numInArg_!=3)){
    cerr << "\nError(linkage): TwoOrThreeInputsRequired.\n"
         << "\tTwo or three inputs required for linkage.\n"
         << endl;
    return;
  }

  if(numOutArg_>1){
    cerr << "\nError(linkage): TooManyOutputArguments.\n"
         << "\tToo many output arguments for linkage.\n"
         << endl;
    return;
  }

  int depth = in.type() & CV_MAT_DEPTH_MASK;

  if (depth != CV_32SC1 && depth != CV_64FC1 && depth != CV_32FC1){
    cerr << "\nError(linkage): UndefinedFunctionOnlyDoubleOrInt.\n"
         << "\tFunction linkage is only defined for values of class 'double' or 'int'.\n"
         << endl;
    return;
  }

  linkageTEMPLATE<TEMPL>(numInArg_, in, method, numOutArg_, _out, (numInArg_ == 3), (TEMPL)(1.0));
}

template<class TEMPL>
void checkCut(InputArray _X, const TEMPL cutoff, InputArray _crit, OutputArray _out)
{
  Mat X = _X.getMat();
  Mat crit_ = _crit.getMat();
  TEMPL* crit = (TEMPL*)crit_.data;

  int n = X.rows + 1;
  _out.create(crit_.rows, 1, CV_32SC1);
  Mat conn_ = _out.getMat();
  int* conn = (int*)conn_.data;

  for (int j = 0; j < conn_.rows; j++){
    if (crit[j] <= cutoff)
      conn[j] = 1;
    else
      conn[j] = 0;
  }

  Mat to_do_(crit_.rows, 1, CV_32SC1);
  int* to_do = (int*)to_do_.data;

  for (int j = 0; j < to_do_.rows; j++){
    if (conn[j] == 1 && (X.at<TEMPL>(j,0) > (n-1) || X.at<TEMPL>(j,1) > (n-1)))
      to_do[j] = 1;
    else
      to_do[j] = 0;
  }

  int continueloop = 0;
  int rows_len = 0;
  Mat rows_(to_do_.rows, 1, CV_32SC1);
  int* rows = (int*)rows_.data;

  while(1){
    continueloop = 0;
    rows_len = 0;
    for (int j = 0; j < to_do_.rows; j++){
      if (to_do[j] == 1){
        continueloop = 1;
        rows[rows_len] = j;
        rows_len++;
      }
    }
    if (continueloop == 0)
      break;

    Mat cdone(rows_len, 2, CV_32SC1);
    cdone.setTo(Scalar(1));
    Mat crows_(rows_len, 1, CV_32SC1);
    int* crows = (int*)crows_.data;

    Mat t_(rows_len, 1, CV_32SC1);
    int* t = (int*)t_.data;

    for (int j=0; j<2; j++){
      for (int i=0; i<rows_len; i++){
        CV_Assert( rows[i] < X.rows );
        crows[i] = (int)X.at<TEMPL>(rows[i],j);
      }

      int t_len = 0;
      for (int i=0; i<rows_len; i++){
        if (crows[i] > (n-1)){
          t[t_len] = i;
          t_len++;
        }
      }

      if (t_len > 0){
        Mat child_(t_len, 1, CV_32SC1);
        int* child = (int*)child_.data;

        for (int i=0; i<t_len; i++){
          CV_Assert( t[i] < crows_.rows );
          child[i] = crows[t[i]] - n;
          if (to_do[child[i]] == 1){
            CV_Assert( t[i] < cdone.rows );
            cdone.at<int>(t[i],j) = 0;
          }else{
            CV_Assert( t[i] < cdone.rows );
            cdone.at<int>(t[i],j) = 1;
          }

          if (conn[child[i]] == 1 && conn[rows[t[i]]] == 1){
            CV_Assert( t[i] < conn_.rows );
            conn[t[i]] = 1;
          }else{
            CV_Assert( t[i] < conn_.rows );
            conn[t[i]] = 0;
          }
        }

        child_.release();
      }
    }

    for (int j = 0; j < cdone.rows; j++){
      if (cdone.at<int>(j,0) == 1 && cdone.at<int>(j,1) == 1){
        CV_Assert( rows[j] < to_do_.rows );
        to_do[rows[j]] = 0;
      }
    }

    cdone.release();
    crows_.release();
    t_.release();

  }
  to_do_.release();
  rows_.release();
}

template<class TEMPL>
void labelTree(InputArray _X, InputArray _conn, OutputArray _out)
{
  Mat X = _X.getMat();
  Mat conn_ = _conn.getMat();
  int* conn = (int*)conn_.data;

  int n = X.rows;
  int nleaves = n+1;
  Mat T_(n+1, 1, CV_32SC1);
  T_.setTo(Scalar(1));
  int* T = (int*)T_.data;

  Mat to_do_(n, 1, CV_32SC1);
  to_do_.setTo(Scalar(1));
  int* to_do = (int*)to_do_.data;

  Mat clustlist(n, 2, CV_32SC1);
  int cnt = 0;
  for (int i = 0; i < clustlist.cols; i++){
    for (int j = 0; j < clustlist.rows; j++){
      clustlist.at<int>(j,i) = cnt;
      cnt++;
    }
  }

  int continueloop = 0;
  int rows_len = 0;
  Mat rows_(to_do_.rows, 1, CV_32SC1);
  int* rows = (int*)rows_.data;

  while(1) {
    continueloop = 0;
    rows_len = 0;
    for (int j = 0; j < to_do_.rows; j++){
      if (to_do[j] == 1){
        continueloop = 1;
        if (conn[j] == 0){
          rows[rows_len] = j;
          rows_len++;
        }
      }
    }
    if (continueloop == 0)
      break;

    if (rows_len == 0){ break; }
    Mat children_(rows_len, 1, CV_32SC1);
    int* children = (int*)children_.data;
    Mat leaf_(rows_len, 1, CV_32SC1);
    int* leaf = (int*)leaf_.data;
    Mat joint_(rows_len, 1, CV_32SC1);
    int* joint = (int*)joint_.data;

    for (int j=0; j<2; j++){
      for (int i=0; i<rows_len; i++){
        CV_Assert( rows[i] < X.rows );
        children[i] = (int)X.at<TEMPL>(rows[i],j);
      }

      int leaf_len = 0;
      int joint_len = 0;

      for (int i=0; i<rows_len; i++){
        if (children[i] < nleaves){
          leaf[leaf_len] = i;
          leaf_len++;
        }else{
          joint[joint_len] = i;
          joint_len++;
        }
      }


      if (leaf_len > 0){
        for (int i=0; i<leaf_len; i++){
          CV_Assert( leaf[i] < children_.rows );
          CV_Assert( children[leaf[i]] < T_.rows );
          CV_Assert( leaf[i] < rows_.rows );
          CV_Assert( rows[leaf[i]] < clustlist.rows );
          T[children[leaf[i]]] = clustlist.at<int>(rows[leaf[i]],j);
        }
      }

      if (joint_len == 0){ continue; }
      int joint1_len = 0;
      Mat joint1_(joint_len, 1, CV_32SC1);
      int* joint1 = (int*)joint1_.data;

      for (int i=0; i<joint_len; i++){
        CV_Assert( joint[i] < children_.rows );
        int test1 = children[joint[i]];
        test1 = children[joint[i]]-nleaves;
        CV_Assert( (children[joint[i]]-nleaves) < conn_.rows && (children[joint[i]]-nleaves) >= 0 );
        if (conn[children[joint[i]]-nleaves] == 1){
          joint1[joint1_len] = joint[i];
          joint1_len++;
        }
      }

      if (joint1_len > 0){
        for (int i=0; i<joint1_len; i++){
          CV_Assert( joint1[i] < rows_.rows );
          int clustnum = clustlist.at<int>(rows[joint1[i]],j);
          int childnum = children[joint1[i]]-nleaves;
          CV_Assert( childnum >= 0 && childnum < clustlist.rows );
          clustlist.at<int>(childnum,0) = clustnum;


          clustlist.at<int>(childnum,1) = clustnum;

          conn[childnum] = 0;
        }
      }
      joint1_.release();
    }
    children_.release();
    leaf_.release();
    joint_.release();

    for (int i=0; i<rows_len; i++){
      to_do[rows[i]] = 0;
    }
  }

  Mat Ttmp1;
  Mat Ttmp2;
  uniqueMat<int>(T_, Ttmp1, Ttmp2, _out);

  T_.release();
  to_do_.release();
  rows_.release();
  clustlist.release();

  Ttmp1.release();
  Ttmp2.release();

}

template<class TEMPL>
void clusterCutoffCriterion(InputArray _Z, const TEMPL threshold, const char *flag, OutputArray _out)
{
  Mat Z = _Z.getMat();

  if ( Z.cols != 3 ){
    cerr << "\nError(clusterCutoffCriterion): BadTree.\n"
         << "\tZ must be a hierarchical tree as computed by the linkage function.\n"
         << endl;
    CV_Assert( Z.cols != 3 );
  }
  for (int i=0; i<Z.rows; i++){
    for (int j=0; j<2; j++){
      if ( (int)floor(Z.at<TEMPL>(i,j)) != (int)Z.at<TEMPL>(i,j) ){
        cerr << "\nError(clusterCutoffCriterion): BadTree.\n"
             << "\tZ must be a hierarchical tree as computed by the linkage function.\n"
             << endl;
        CV_Assert( floor(Z.at<TEMPL>(i,j)) != Z.at<TEMPL>(i,j) );
      }
    }
  }

  Mat crit(Z.rows, 1, Z.type()); // distance criterion
  for (int j = 0; j < crit.rows; j++){
    crit.at<TEMPL>(j,0) = Z.at<TEMPL>(j,2);
  }
  Mat conn;
  checkCut<TEMPL>(Z, threshold, crit, conn);
  labelTree<TEMPL>(Z, conn, _out);

  crit.release();

}

#endif /* LINKAGE_H_ */
