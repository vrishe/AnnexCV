#pragma once

#ifndef __ANNEX_SQI_H__
#define __ANNEX_SQI_H__


namespace cv
{
	cv::Ptr<cv::BaseFilter>		getSQIFilter	(int srcType, int dstType, unsigned radius = 1, double sigma = -1.0);
	cv::Ptr<cv::FilterEngine>	createSQIFilter	(int srcType, int dstType, unsigned radius = 1, double sigma = -1.0, 
		int rowBorderType = cv::BORDER_DEFAULT, int columnBorderType = -1, const cv::Scalar &borderValue = cv::Scalar());

	void filterSQI(const cv::Mat &src, cv::Mat &dst, unsigned radius = 1, double sigma = -1.0, int borderType = cv::BORDER_DEFAULT);
}

#endif // __ANNEX_SQI_H__