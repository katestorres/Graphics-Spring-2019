
///////////////////////////////////////////////////////////////////////////////
// gfxrasterize.hpp
//
// Line segment rasterization.
//
// This file builds upon gfximage.hpp, so you may want to familiarize
// yourself with that header before diving into this one.
//
// Students: all of your work should go in this file, and the only files that
// you need to modify in project 1 are this file, and README.md.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include "gfximage.hpp"
#include "gfxpng.hpp"

namespace gfx {

// Draw a line segment from (x0, y0) to (x1, y1) inside image target, all
// with color.
//
// target must be non-empty.
// (x0, y0) and (x1, y1) must be valid coordinates in target.
// There is no restriction on how (x0, y0) and (x1, y1) must be oriented
// relative to each other.
//

float implicit_line(float x0, float y0, float x1, float y1, float x, float y) {
  return (y0-y1)*x + (x1-x0)*y + x0*y1 - x1*y0;
}

void rasterize_line_segment(hdr_image& target,
                            unsigned x0, unsigned y0,
                            unsigned x1, unsigned y1,
                            const hdr_rgb& color) {

  assert(!target.is_empty());
  assert(target.is_xy(x0, y0));
  assert(target.is_xy(x1, y1));

  //Swap
  if (x1 < x0) {
    unsigned int temp;
    temp = x1;
    x1 = x0;
    x0 = temp;
    temp = y1;
    y1 = y0;
    y0 = temp;
  }

  float m;
  if (x1 - x0 == 0) { //Same x-coordinate
    if (y1 - y0 == 0) { //Same y-coordinate
      target.pixel(x0,y0,color);
      return;
    }
    else if (y1 > y0) m = 10;
    else m = -10;
  }
  else m = (static_cast<float>(y1) - y0) / (x1 - x0); // slope

 //Shallow positive slope
  if (m > 0.0f && m <= 1.0f) {
    unsigned int y = y0;
    float d = implicit_line(x0,y0,x1,y1,x0+1.0f,y0+0.5f);
    for (unsigned int x = x0; x <= x1; ++x) {
      target.pixel(x,y,color);
      if (d < 0.0f) {
        ++y;
        d = implicit_line(x0,y0,x1,y1,x+2.0f,y+0.5f);
      }
      else {
        d = implicit_line(x0,y0,x1,y1,x+2.0f,y+0.5f);
      }
    }
  }
  
  //Shallow negative slope
  else if (m > -1.0f && m <= 0.0f) {
    unsigned int y = y0;
    float d = implicit_line(x0,y0,x1,y1,x0+1.0f,y0-0.5f);
    for (unsigned int x = x0; x <= x1; ++x) {
      target.pixel(x,y,color);
      if (d > 0.0f) {
        --y;
        d = implicit_line(x0,y0,x1,y1,x+2.0f,y-0.5f);
      }
      else {
        d = implicit_line(x0,y0,x1,y1,x+2.0f,y-0.5f);
      }
    }
  }
  
  //Steep negative slope
  else if (m <= -1.0f) { 
    unsigned int x = x0;
    float d = implicit_line(y0,x0,y1,x1,y0-1.0f,x0+0.5f);  
    for (int y = y0; y >= static_cast<int>(y1); --y) {
      target.pixel(x,y,color);
      if (d >= 0.0f) {
        ++x;
        d = implicit_line(y0,x0,y1,x1,y-2.0f,x+0.5f);
      }
      else {
        d = implicit_line(y0,x0,y1,x1,y-2.0f,x+0.5f);
      }
    }
  }
  
  //Steep positive slope
  else if (m > 1) { 
    unsigned int x = x0;
    float d = implicit_line(y0,x0,y1,x1,y0+1.0f,x0+0.5f);
    for (unsigned int y = y0; y <= y1; ++y) {
      target.pixel(x,y,color);
      if (d < 0.0f) {
        ++x;
        d = implicit_line(y0,x0,y1,x1,y+2.0f,x+0.5f);
      }
      else {
        d = implicit_line(y0,x0,y1,x1,y+2.0f,x+0.5f);
      }
    }
  }
}

// Convenience function to create many images, each containing one rasterized
// line segment, and write them to PNG files, for the purposes of unit testing.
bool write_line_segment_cases(const std::string& filename_prefix) {
  for (unsigned end_x = 0; end_x <= 10; ++end_x) {
    for (unsigned end_y = 0; end_y <= 10; ++end_y) {
      hdr_image img(11, 11, gfx::SILVER);
      rasterize_line_segment(img, 5, 5, end_x, end_y, gfx::RED);
      std::string filename = (filename_prefix
                              + "-" + std::to_string(end_x)
                              + "-" + std::to_string(end_y)
                              + ".png");
      if (!write_png(img, filename)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace gfx
