#include "precomp.h"


#include "sqi.h"
#include "sqi_template.cpp"


namespace cv
{
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cv::Ptr<cv::BaseFilter> getSQIFilter(int srcType, int dstType, unsigned radius, double sigma)
	{
		int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);

		CV_Assert( CV_MAT_CN(srcType) == CV_MAT_CN(dstType) && sdepth <= ddepth );

		if( sdepth == CV_8U && ddepth == CV_32F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<uchar, float, float>(radius, sigma));
		if( sdepth == CV_8U && ddepth == CV_64F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<uchar, double, double>(radius, sigma));

		if( sdepth == CV_16U && ddepth == CV_32F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<ushort, float, float>(radius, sigma));
		if( sdepth == CV_16U && ddepth == CV_64F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<ushort, double, double>(radius, sigma));

		if( sdepth == CV_16S && ddepth == CV_32F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<short, float, float>(radius, sigma));
		if( sdepth == CV_16S && ddepth == CV_64F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<short, double, double>(radius, sigma));

		if( sdepth == CV_32F && ddepth == CV_32F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<float, float, float>(radius, sigma));
		if( sdepth == CV_64F && ddepth == CV_64F )
			return cv::Ptr<cv::BaseFilter>(
				new SelfQuotientImageFilter<double, double, double>(radius, sigma));

		CV_Error_( CV_StsNotImplemented, 
			("Unsupported combination of source format (=%d), and destination format (=%d)", srcType, dstType) );

		return cv::Ptr<cv::BaseFilter>(NULL);
	}

	cv::Ptr<cv::FilterEngine> createSQIFilter(int srcType, int dstType, unsigned radius, double sigma, 
		int rowBorderType, int columnBorderType, const cv::Scalar &borderValue)
	{
		srcType = CV_MAT_TYPE(srcType); dstType = CV_MAT_TYPE(dstType);
		
		cv::Ptr<cv::BaseFilter> _filter2D = getSQIFilter(srcType, dstType, radius, sigma);
		return cv::Ptr<cv::FilterEngine>(new cv::FilterEngine(
			_filter2D, cv::Ptr<cv::BaseRowFilter>(0), cv::Ptr<cv::BaseColumnFilter>(0), 
			srcType, dstType, srcType, rowBorderType, columnBorderType, borderValue)
			);
	}

	void filterSQI(const cv::Mat &src, cv::Mat &dst, unsigned radius, double sigma, int borderType)
	{
		CV_Assert( isBitmap(src) );
		if ( dst.dims != 2 || src.depth() > dst.depth() || src.channels() != dst.channels() || src.size() != dst.size() )
		{
			CV_Assert( dst.empty() || dst.refcount != NULL );
			dst.create(src.size(), CV_MAKETYPE(CV_MAT_DEPTH(src.type()) <= CV_32F ? CV_32F : CV_64F, src.channels()));
		}

		cv::Ptr<cv::FilterEngine> filter 
			= createSQIFilter(src.type(), dst.type(), radius, sigma, borderType); 
		filter->apply(src, dst);
	}
}