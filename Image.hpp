/***********************************************************************
  $FILENAME    : Image.hpp

  $TITLE       : Utility image template functions

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Implements some utility template functions for images

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef IMAGE_H_
#define IMAGE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Constants.h"
#include "BoundingBox.hpp"
#include "main.hpp"

using namespace std;
using namespace cv;

template<class TEMPL>
void imgPatch( InputArray _img_gray, OutputArray _patch, InputArray _bbox, int colBBox )
{
  Mat img_gray = _img_gray.getMat();
  Mat bbox = _bbox.getMat();

  int L = (int)fmax(0, bbox.at<TEMPL>(0,colBBox)); // X1 : L = max([1 bb(1)]);
  int T = (int)fmax(0, bbox.at<TEMPL>(1,colBBox)); // Y1 : T = max([1 bb(2)]);
  int R  = (int)fmin(img_gray.cols, bbox.at<TEMPL>(2,colBBox)); // X2 : R = min([size(img,2) bb(3)]);
  int B = (int)fmin(img_gray.rows, bbox.at<TEMPL>(3,colBBox)); // Y2 : B = min([size(img,1) bb(4)]);

  // bbox.at<Vec3b>(rows,cols)[0]
  // Output
  // width : R-L
  // height : B-T
  _patch.create( Size(R-L,B-T), img_gray.type() );
  Mat patch = _patch.getMat();

  uchar* src = (uchar *) (img_gray.data+ L*sizeof(uchar) + T*img_gray.cols*sizeof(uchar));
  uchar* dest = (uchar *) (patch.data);
  int x = patch.cols * sizeof(uchar);
  for(int y=0; y<patch.rows; y++) {// row : height
    memcpy ( dest, src, x );
    src += img_gray.cols * sizeof(uchar);
    dest += x;
  }
//  for(int y=T; y<B; y++) {// row : height
//    for(int x=L; x<R; x++) {// col : width
//      patch.at<uchar>(y-T,x-L) = img_gray.at<uchar>(y,x);
//    }
//  }
  // patch = img(T:B,L:R); Y1:Y2, X:X2

//  saveMatrix<uchar>(patch, true);
}

/* Warps image of size w x h, using affine transformation matrix (2x2 part)
   and offset (center of warping) ofsx, ofsy. Result is the region of size
   defined with roi. */
template<class TEMPL>
void warpImageROI(InputArray _image, InputArray _H, InputArray _B, TEMPL noiseVal,
    OutputArray _result)
{
  Mat image = _image.getMat();
  Mat H = _H.getMat();

  Mat B_ = _B.getMat();
  const TEMPL* B = (const TEMPL*)B_.data;

  TEMPL xmin = B[0];
  TEMPL xmax = B[1];
  TEMPL ymin = B[2];
  TEMPL ymax = B[3];

  _result.create((int)(ymax-ymin+1), (int)(xmax-xmin+1), image.type());
  Mat result_ = _result.getMat();
  uchar* output = (uchar*)result_.data;

  int w, h;
  w = image.cols;
  h = image.rows;

  TEMPL curx, cury, curz, wx, wy, wz, ox, oy, oz;
  int x, y;
  TEMPL i, j, xx, yy;
  TEMPL tempVal;
  /* pre-calculate necessary constant with respect to i,j offset
      translation, H is column oriented (transposed) */
  ox = H.at<TEMPL>(0,2);
  //ox = M(0,2);
  oy = H.at<TEMPL>(1,2);
  //oy = M(1,2);
  oz = H.at<TEMPL>(2,2);
  //oz = M(2,2);

  srand ( time(0) );

  yy = ymin;
  for (j=0; j<(int)(ymax-ymin+1); j++){ // row : height
    /* calculate x, y for current row */
    curx = H.at<TEMPL>(0,1)*yy + ox;
    //curx = M(0,1)*yy + ox;
    cury = H.at<TEMPL>(1,1)*yy + oy;
    //cury = M(1,1)*yy + oy;
    curz = H.at<TEMPL>(2,1)*yy + oz;
    //curz = M(2,1)*yy + oz;

    xx = xmin;
    yy = yy + 1;
    for (i=0; i<(int)(xmax-xmin+1); i++){ // col : width
      /* calculate x, y in current column */
      wx = H.at<TEMPL>(0,0)*xx + curx;
      //wx = M(0,0)*xx + curx;
      wy = H.at<TEMPL>(1,0)*xx + cury;
      //wy = M(1,0)*xx + cury;
      wz = H.at<TEMPL>(2,0)*xx + curz;
      //wz = M(2,0)*xx + curz;
      //       printf("%g %g, %g %g %g\n", xx, yy, wx, wy, wz);
      wx /= wz; wy /= wz;
      xx = xx + 1;

      x = (int)floor(wx);
      y = (int)floor(wy);

      if (x>=0 && y>=0){
        wx -= x; wy -= y;
        if (x+1==w && wx==1)
          x--;
        if (y+1==h && wy==1)
          y--;
        if ((x+1)<w && (y+1)<h){
          tempVal =
              (TEMPL)(image.at<uchar>(y,x)*(1-wx)*(1-wy) + image.at<uchar>(y,x+1)*wx*(1-wy) +
              image.at<uchar>(y+1,x)*(1-wx)*wy + image.at<uchar>(y+1,x+1)*wx*wy);
          tempVal += noiseVal*rand()/(TEMPL(RAND_MAX)+1);
          *output++ = (uchar)tempVal;
        } else
          *output++ = (uchar)0;
      } else
        *output++ = (uchar)0;
    }
  }
}

template<class TEMPL>
void imgPatch( InputArray _img_gray, OutputArray _patch, InputArray _bbox, tldOptPParam p_par )
{
  Mat img_gray = _img_gray.getMat();
  Mat bbox = _bbox.getMat();

  CV_Assert( bbox.cols == 1 && bbox.rows == 4 );
//  CV_Assert( (bbox.type() & CV_MAT_DEPTH_MASK) == CV_32FC1 );

  int dataType;
  if ((bbox.type() & CV_MAT_DEPTH_MASK) == CV_32FC1){
    dataType = CV_32FC1;
  }else if ((bbox.type() & CV_MAT_DEPTH_MASK) == CV_64FC1){
    dataType = CV_64FC1;
  }else{
    assert(0);
    CV_Error( CV_BadDepth, "" );
  }

  TEMPL NOISE = (TEMPL)p_par.noise;
  TEMPL ANGLE = (TEMPL)p_par.angle;
  TEMPL SCALE = (TEMPL)p_par.scale;
  TEMPL SHIFT = (TEMPL)p_par.shift;

  TEMPL cp[2];
  cp[0] = (bbox.at<TEMPL>(0,0)+bbox.at<TEMPL>(2,0))/2.0-1.0;
  cp[1] = (bbox.at<TEMPL>(1,0)+bbox.at<TEMPL>(3,0))/2.0-1.0;
  //cp  = bbox_center(bb)-1;

  TEMPL sh1[3][3] = {{1., 0., -1.*cp[0]}, {0., 1., -1.*cp[1]}, {0., 0., 1.}};
  Mat Sh1(3, 3, dataType, sh1);
  Sh1 = Sh1.clone();
  //Sh1 = [1 0 -cp(0); 0 1 -cp(1); 0 0 1];

  srand ( time(0) );
  TEMPL randomize = rand()/(TEMPL(RAND_MAX)+1);
  TEMPL sca_ = 1-SCALE*(randomize-0.5);
  //sca = 1-SCALE*(rand-0.5);

  TEMPL sca[3][3] = {{sca_, 0., 0.}, {0., sca_, 0.}, {0., 0., 1.}};
  Mat Sca(3, 3, dataType, sca);
  Sca = Sca.clone();
  //Sca = diag([sca sca 1]);

  srand ( time(0) );
  randomize = rand()/(TEMPL(RAND_MAX)+1);
  TEMPL ang_ = (2.0*(M_PI)/360.0)*ANGLE*(randomize-0.5);
  //ang = 2*pi/360*ANGLE*(rand-0.5);

  TEMPL ca = (TEMPL)cos(ang_);
  TEMPL sa = (TEMPL)sin(ang_);

  TEMPL ang[3][3] = {{ca, -1*sa, 0.}, {sa, ca, 0.}, {0., 0., 1.}};
  Mat Ang(3, 3, dataType, ang);
  Ang = Ang.clone();
  //Ang = [ca, -sa; sa, ca];
  //Ang(end+1,end+1) = 1;

  TEMPL bbH = bbox.at<TEMPL>(3,0)-bbox.at<TEMPL>(1,0)+1.;
  TEMPL bbW = bbox.at<TEMPL>(2,0)-bbox.at<TEMPL>(0,0)+1.;

  srand ( time(0) );
  randomize = rand()/(TEMPL(RAND_MAX)+1);
  TEMPL shR = SHIFT*bbH*(randomize-0.5);
  //shR  = SHIFT*bb_height(bb)*(rand-0.5);

  srand ( time(0) );
  randomize = rand()/(TEMPL(RAND_MAX)+1);
  TEMPL shC = SHIFT*bbW*(randomize-0.5);
  //shC  = SHIFT*bb_width(bb)*(rand-0.5);

  TEMPL sh2[3][3] = {{1., 0., shC}, {0., 1., shR}, {0., 0., 1.}};
  Mat Sh2(3, 3, dataType, sh2);
  Sh2 = Sh2.clone();
  //Sh2 = [1 0 shC; 0 1 shR; 0 0 1];

  bbW = bbW-1.;
  //bbW = bb_width(bb)-1;

  bbH = bbH-1.;
  //bbH = bb_height(bb)-1;

  Mat box_(4, 1, dataType);
  TEMPL* box = (TEMPL*)box_.data;
  box[0] = -1.0*bbW/2.0;
  box[1] = bbW/2.0;
  box[2] = -1.0*bbH/2.0;
  box[3] = bbH/2.0;
  //box = [-bbW/2 bbW/2 -bbH/2 bbH/2];

  Mat H = Sh2*Ang;   // Sh2*Ang -> H
  H = H*Sca;   // H*Sca -> H
  H = H*Sh1;   // H*Sh1 -> H
  //H = Sh2*Ang*Sca*Sh1;

  int bbsize[2];

  bbsize[0] = (int)(bbox.at<TEMPL>(3,0)-bbox.at<TEMPL>(1,0)+1.);
  bbsize[1] = (int)(bbox.at<TEMPL>(2,0)-bbox.at<TEMPL>(0,0)+1.);
  //bbsize = bb_size(bb);

  H = H.inv(); // inv(H) -> H

  // Output
  warpImageROI<TEMPL>(img_gray, H, box_, NOISE, _patch);

  //patch = uint8(warp(img_gray, inv(H), box) + NOISE*randn(bbsize(1),bbsize(2)));

  Sh1.release();
  Sca.release();
  Ang.release();
  Sh2.release();
  box_.release();
  H.release();
}

#endif /* IMAGE_H_ */
