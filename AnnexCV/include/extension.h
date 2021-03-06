#pragma once

#ifndef __ANNEX_EXTENSION_H__
#define __ANNEX_EXTENSION_H__


#define BEGIN_TIMER_SECTION(tick_var)		\
	int64  tick_var = cv::getTickCount();

#if defined ANNEX_MACROS_EXTENDED

#if defined WIN32 || defined _WIN32 || defined WINCE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define END_TIMER_SECTION(tick_var, logtext)												\
{																							\
	SIZE_T _text_amount = strlen(logtext) + 23;												\
	TCHAR* _debugstr	= new TCHAR[_text_amount];											\
	sprintf_s(_debugstr, _text_amount, "%s: %fs.\n", logtext,								\
					(cv::getTickCount() - (tick_var)) / cv::getTickFrequency());			\
	OutputDebugString(_debugstr);															\
	delete[] _debugstr;																		\
}

#else

#include <iostream>

#define END_TIMER_SECTION(tick_var, logtext)												\
{																							\
	double elapsed_time = (double(cv::getTickCount() - tick_var) / cv::getTickFrequency());	\
	std::cout << logtext << ": " << elapsed_time << "s." std::endl;							\
}

#endif

#else

#define END_TIMER_SECTION(tick_var, time_var)												\
	time_var = (double(cv::getTickCount() - tick_var) / cv::getTickFrequency());

#endif


#define FOR_MATRIX2D(matrix, elemtype, elemname, code)							\
/*	_ASSERT(matrix.channels() == 1 && matrix.dims == 2)*/						\
	for (int __r = 0; __r < matrix.rows; ++__r)									\
	{																			\
		elemtype *__row = const_cast<elemtype *>(matrix.ptr<elemtype>(__r));	\
		for (int __c = 0; __c < matrix.cols; ++__c)								\
		{																		\
			elemtype &elemname = __row[__c];									\
			code																\
		}																		\
	}


namespace cv
{

	class PyramidAdapterHack : public cv::PyramidAdaptedFeatureDetector
	{
	public:
		PyramidAdapterHack(const cv::Ptr<cv::FeatureDetector>& detector, int maxLevel = 2)
			: PyramidAdaptedFeatureDetector(detector, maxLevel) { }

		using cv::PyramidAdaptedFeatureDetector::maxLevel;
		using cv::PyramidAdaptedFeatureDetector::detector;
	};

	// Produces a square matrix (2*radius + 1)x(2*radius + 1) filled by 1's in a disk form
	cv::Mat diskMask(double radius);
	// Gives a local maxima mask (map) for a grayscale image given
	size_t nonMaxSuppression3x3(const cv::Mat &src, cv::Mat &dst, bool preserveMaximaValues = false);
	// Gives graymap intensity occurence histogram for a grayscale image
	void discreteGraymapHistogram(const cv::Mat &src, cv::OutputArray dst);
	void truncateHist(cv::Mat &src, double plow, double phigh);

	template<typename _Vt>
	inline cv::Point_<_Vt> operator/(const cv::Point_<_Vt> &pt, _Vt scalar) { return cv::Point_<_Vt>(pt.x / scalar, pt.y / scalar); }

	template<typename _Vt>
	inline cv::Point_<_Vt> operator*(const cv::Point_<_Vt> &pt, _Vt scalar) { return cv::Point_<_Vt>(pt.x * scalar, pt.y * scalar); }

	template<typename _Vt, int _cn>
	inline cv::Vec<_Vt, _cn> operator/(const cv::Vec<_Vt, _cn> &v, _Vt scalar) 
	{ 
		cv::Vec<_Vt, _cn> vector;
		for (int i = 0; i < _cn; ++i)
			vector[i] = v[i] / scalar;

		return vector; 
	}

	template<typename _Vt, int _cn>
	inline cv::Vec<_Vt, _cn> operator*(const cv::Vec<_Vt, _cn> &v, _Vt scalar) 
	{ 
		cv::Vec<_Vt, _cn> vector;
		for (int i = 0; i < _cn; ++i)
			vector[i] = v[i] * scalar;

		return vector; 
	}
	
	template<typename _Vt> 
	inline int sign(_Vt val) { return val != 0 ? (val > 0 ? 1 : -1) : 0; }

	template<> inline int sign<unsigned>			(unsigned			val) { return static_cast<int>(val != 0); }
	template<> inline int sign<uchar>				(uchar				val) { return static_cast<int>(val != 0); }
	template<> inline int sign<ushort>				(ushort				val) { return static_cast<int>(val != 0); }
	template<> inline int sign<unsigned long>		(unsigned long		val) { return static_cast<int>(val != 0); }
	template<> inline int sign<unsigned long long>	(unsigned long long val) { return static_cast<int>(val != 0); }

	// TODO: will be used no std::funcs in future
	inline double round (double x) { return x > 0.0  ? std::floor(x + 0.5)  : std::ceil(x - 0.5);  }
	inline float  round (float  x) { return x > 0.0F ? std::floor(x + 0.5F) : std::ceil(x - 0.5F); }

	inline bool isBitmap (const cv::Mat &mat) 
	{ 
		int depth = mat.depth();
		return mat.dims == 2 &&
			(depth == CV_8U || depth == CV_16U || depth == CV_32F);
	}
	inline bool isGraymap(const cv::Mat &mat) { return isBitmap(mat) && mat.channels() == 1; }

} // namespace cv


#endif // #ifndef __ANNEX_EXTENSION_H