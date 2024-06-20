#pragma once

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


namespace py = pybind11;

class Converter {
public:
  explicit Converter(const cv::Size &resolution)
      : height_(resolution.height), 
        width_(resolution.width) {}

  std::vector<float_t> process(
    const py::array_t<int64_t> &array, 
    const py::array_t<int64_t> &box) {
    
    // obtain box range
    int64_t tl_x = box.at(0);
    int64_t tl_y = box.at(1);
    int64_t dr_x = box.at(0) + box.at(2);
    int64_t dr_y = box.at(1) + box.at(3);
    
    // convert to binary image
    const auto array_size = array.request().shape[0];
    cv::Mat image_1 = cv::Mat::zeros(width_, height_, CV_8UC1); // forward
    cv::Mat image_2 = cv::Mat::zeros(width_, height_, CV_8UC1); // backward
    
    size_t count = 0;
    for (size_t i = 0; i < array_size; i++) {
      // coordinate
      int64_t x = array.at(i, 1);
      int64_t y = array.at(i, 2);

      // whether out of range
      if (x < tl_x || x > dr_x && y < tl_y || y > dr_y) {
        continue;
      }

      if (i <= array_size / 2) {
        image_1.at<uint8_t>(array.at(i, 1), array.at(i, 2)) = 255;
      } else {
        image_2.at<uint8_t>(array.at(i, 1), array.at(i, 2)) = 255;
      }
      count++;
    }

    // combine images
    cv::Mat image = image_1 | image_2;

    // contour
    const std::vector<cv::Point> contour = findMainContour(image);
    const double contour_area = cv::contourArea(contour);
    const double arc_length = cv::arcLength(contour, true) + 1;

    // corners
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, 100, 0.01, 10);

    // backward
    const std::vector<cv::Point> contour_1 = findMainContour(image_1);
    const double area_1 = cv::contourArea(contour_1);
    cv::Moments M_1 = cv::moments(contour_1);
    cv::Point centroid_1(M_1.m10 / M_1.m00, M_1.m01 / M_1.m00);

    // forward
    const std::vector<cv::Point> contour_2 = findMainContour(image_2);
    const double area_2 = cv::contourArea(contour_2);
    cv::Moments M_2 = cv::moments(contour_2);
    cv::Point centroid_2(M_2.m10 / M_2.m00, M_2.m01 / M_2.m00);

    std::vector<float_t> results{
      (float_t) (count),
      (float_t) ((float_t) box.at(2) / (float_t) box.at(3)),
      (float_t) (contour_area / (box.at(2) * box.at(3))),
      (float_t) (4 * M_PI * contour_area / (arc_length * arc_length)),
      (float_t) (corners.size()),
    };

    return results;
  }

private:
  std::vector<cv::Point> findMainContour(cv::Mat &image) {
    // find contours
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // sort contours
    std::sort(contours.begin(), contours.end(), 
              [](std::vector<cv::Point> &contour_1, std::vector<cv::Point> &contour_2){
                double area1 = cv::contourArea(contour_1);
                double area2 = cv::contourArea(contour_2);
                return (area1 > area2);
              });

    return contours[0];
  }

  size_t height_;
  size_t width_;
};