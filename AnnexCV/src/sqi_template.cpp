#include "extension.h"

namespace cv
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename _St, typename _Kt, typename _Rt>
	class SelfQuotientImageFilter : public cv::BaseFilter
	{
		// Filter direct parameters
		unsigned	paramRadius;
		double		paramSigma;

		// Internal intermediate data structures
		std::vector<cv::Point>		_pt;		// Kernel top-left based relative coordinates.
		std::vector<void *>			_psrc;		// Source image pixel data element pointer set.
		std::vector<_Kt>			_kval;		// Kernel values (Gaussian).
		std::vector<_Kt>			_wval;

	public:
		SelfQuotientImageFilter(unsigned radius = 1U, double sigma = -1.0)
			: paramRadius(0), paramSigma(std::numeric_limits<double>::signaling_NaN()) { init(radius, sigma); }

		void init		(unsigned radius = 1U, double sigma = -1.0);
		void operator()	(const uchar** src, uchar* dst, int dststep, int dstcount, int width, int cn);
	};

	template <typename _St, typename _Kt, typename _Rt>
	void SelfQuotientImageFilter<_St, _Kt, _Rt>::init (unsigned radius, double sigma)
	{
		CV_Assert(radius > 0);

		if (sigma == -1.0) sigma = paramSigma = radius / 3.0;
		CV_Assert(sigma > .0);

		if (radius != paramRadius)
		{
			paramRadius = radius;
			ksize		= cv::Size	(2*radius + 1, 2*radius + 1);
			anchor		= cv::Point	(radius, radius);

			for (int i = 0; i < ksize.height; ++i)
			{
				for (int j = 0; j < ksize.height; ++j)
					_pt.push_back(cv::Point(j,i) - anchor);
			}
			_psrc.resize(_pt.size());
			_kval.resize(_pt.size());
			_wval.resize(_pt.size());
		}

		double double_sigma				= 2.0 * sigma;
		double double_pi_sigma_square	= double_sigma * sigma * CV_PI;
		for (size_t i = 0, imax = _kval.size(); i < imax; ++i)
		{
			const cv::Point &p = _pt[i];
			_kval[i] = cv::saturate_cast<_Kt>(std::exp(
				-((p.x * p.x + p.y * p.y) / double_sigma)) * double_pi_sigma_square);
		}
	}

	template <typename _St, typename _Kt, typename _Rt>
	void SelfQuotientImageFilter<_St, _Kt, _Rt>::operator()(const uchar** src, uchar* dst, int dststep, int dstcount, int width, int cn)
	{
		const _St**	values = (const _St**)&_psrc[0]; // C-style conversion. for the sake of shortness
		_Kt*		kernel = &_kval[0];
		_Kt*		weight = &_wval[0];

		double kernelDivisor = ksize.area();

		const cv::Point* coords = &_pt[0];
		for (int i, j, nWidth = width * cn, nElem = _pt.size(); dstcount > 0; --dstcount, ++src, dst += dststep)
		{
			const _St* anchorValue = reinterpret_cast<const _St*>(src[anchor.y]) + anchor.x * cn;
			_Rt* output	= reinterpret_cast<_Rt*>(dst);

			for (j = 0; j < nElem; ++j)
				values[j] = reinterpret_cast<const _St*>(src[coords[j].y + anchor.y]) + (coords[j].x + anchor.x) * cn;

			for (i = 0; i < nWidth; ++i)
			{
				_Kt mean = _Kt(0);
				for (j = 0; j < nElem; ++j)
					mean += values[j][i];
				mean /= nElem;

				_Kt kernelSum = _Kt(0);
				for (j = 0; j < nElem; ++j)
				{
					if (values[j][i] < mean) 
					{
						weight[j] = .0;
						continue;
					}
					kernelSum += weight[j] = kernel[j];
				}

				double convolutionValue = .0;
				for (j = 0; j < nElem; ++j)
					convolutionValue += values[j][i] * (weight[j] /= kernelSum);
				output[i] = cv::saturate_cast<_Rt>(
					anchorValue[i] / (convolutionValue / kernelDivisor + .1));

				if (output[i] != output[i] || output[i] == std::numeric_limits<_Rt>::infinity())
					output[i] = _Rt(0);
			}
		}
	}
}