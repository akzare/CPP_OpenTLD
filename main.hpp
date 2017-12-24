/***********************************************************************
  $FILENAME    : main.hpp

  $TITLE       : main application class definition

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Defines the main application class

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef MAIN_TLD_H_
#define MAIN_TLD_H_

#include <opencv2/core/core.hpp>      // Basic OpenCV structures

using namespace std;
using namespace cv;

class trackerArgs;
class trackerApp;

void PrintHelp();

typedef struct {
  string src;
  bool src_is_video;
  bool src_is_camera;
  bool src_is_image;
  int camera_id;
  bool write_video;
  string dst_video;
  float dst_video_fps;
  cv::Rect bbox;
}tldOptSrc;

typedef struct {
  int min_win;
  int patchsize[2];
  int fliplr;
  float ncc_thesame;
  float valid;
  int num_trees;
  int num_features;
  float thr_fern;
  float thr_nn;
  float thr_nn_valid;
  int num_init;
}tldOptModel;

// synthesis of positive examples during initialization
typedef struct {
  int num_closest;
  int num_warps;
  float noise;
  float angle;
  float shift;
  float scale;
}tldOptPParam;

typedef struct {
  float overlap;
  int num_patches;
}tldOptNParam;

typedef struct {
  int occlusion;
}tldOptTracker;

typedef struct {
  int maxbbox;
  int update_detector;
  int drop_img;
  int repeat;
}tldOptCtrl;

class trackerArgs
{
public:
  trackerArgs();
  static trackerArgs ReadArgs(int argc, char** argv);

  tldOptSrc source;

  tldOptModel model;

  // synthesis of positive examples during initialization
  tldOptPParam p_par_init;
  // synthesis of positive examples during update
  tldOptPParam p_par_update;

  // negative examples initialization/update
  tldOptNParam n_par;

  tldOptTracker tracker;

  tldOptCtrl control;
};

class trackerApp
{
public:
  trackerApp(const trackerArgs& s);
  virtual ~trackerApp();
  void RunTLD();
  void HandleKey(char key);
  void TLDWorkBegin();
  void TLDWorkEnd();
  string TLDWorkFps() const;
  void WorkBegin();
  void WorkEnd();
  string WorkFPS() const;
  string message() const;
  Mat curRGBImg;
  Mat curGrayImg;
  Mat curBlurImg;
  Mat prvGrayImg;
  Mat prvBlurImg;

#ifdef GRAYBLUR_SHOW_ON_
  Mat targetPatch;
#endif
#ifdef FERN_OPENCV_ON_
  Mat correspondImg;
#endif
  trackerArgs tldargs;

private:
  trackerApp operator=(trackerApp&);
  bool running;
  char image_name[256];
  int image_idx;
  int64 tld_work_begin;
  double tld_work_fps;
  int64 work_begin;
  double work_fps;

public:
#ifdef USE_OCL_
  int devNums;
#endif
};

#endif /* MAIN_TLD_H_ */
