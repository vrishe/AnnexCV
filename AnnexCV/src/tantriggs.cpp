
#include "precomp.h"

#include "extension.h"
#include "tantriggs.h"


#define ZERO_ONE_SPAN	.0, 1.
#define TRUNC_DEFAULT	.2, 99.8


namespace cv
{
	// Algortim setters
	void TanTriggsNormalization::set_gamma(float gamma)
	{
		m_gamma = std::max(gamma, .0f);
	}

	void TanTriggsNormalization::set_sigma0(float sigma0)
	{
		m_sigma[0] = std::max(sigma0, 1.f);

		if (m_sigma[0] >= m_sigma[1])
			set_sigma1(m_sigma[0]);
	}

	void TanTriggsNormalization::set_sigma1(float sigma1)
	{
		m_sigma[1] = std::min(m_sigma[0] + 1.f, sigma1);
	}

	void TanTriggsNormalization::set_tau(float tau)
	{
		m_tau = std::max(tau, 1.f);
	}

	void TanTriggsNormalization::set_alpha(float alpha)
	{
		m_alpha = std::max(alpha, .1f);
	}


	// Algorithm sequence
	inline void TanTriggsNormalization::_gammaCorrection(TanTriggsNormalization *alg, cv::Mat &img)
	{
		FOR_MATRIX2D(img, float, intensity,
			intensity = std::pow(intensity, alg->m_gamma); );
		cv::normalize(img, img, .0, 1., cv::NORM_MINMAX);
	}

	inline void TanTriggsNormalization::_differenceOfGaussian(TanTriggsNormalization *alg, cv::Mat &img)
	{
		cv::Mat inner, outer;

		cv::GaussianBlur(img, inner, cv::Size(-1, -1), alg->m_sigma[0]);
		cv::GaussianBlur(img, outer, cv::Size(-1, -1), alg->m_sigma[1]);

		img = inner - outer;

		cv::truncateHist(img, .2f, 99.8f);
		cv::normalize(img, img, .0, 1., cv::NORM_MINMAX);
	}

	inline void TanTriggsNormalization::_contrastEqualization(TanTriggsNormalization *alg, cv::Mat &img)
	{
		int img_area = img.size().area();
		
		double meanSum = .0;
		FOR_MATRIX2D(img, float, value, 
				meanSum += std::pow(std::abs(value), alg->m_alpha); );
		img /= std::pow(meanSum / img_area, 1. / alg->m_alpha);

		meanSum = .0;
		FOR_MATRIX2D(img, float, value,
			meanSum += std::pow(std::min(alg->m_tau, std::abs(value)), alg->m_alpha); );
		img /= std::pow(meanSum / img_area, 1. / alg->m_alpha);

		FOR_MATRIX2D(img, float, intensity,
			intensity = alg->m_tau * std::tanh(intensity / alg->m_tau); );
		cv::normalize(img, img, .0, 1., cv::NORM_MINMAX);
	}


	// Normalization chain
	void TanTriggsNormalization::apply(cv::Mat &img)
	{
		CV_Assert(isGraymap(img));

		if (img.empty()) return;

		int	img_depth = img.depth();
		double	elemMaxSize;
		if (img.type() != CV_32FC1)
			img.convertTo(img, CV_32FC1, 1.0 / (elemMaxSize = ((1 << (img.elemSize1() << 3)) - 1)));

		_gammaCorrection		(this, img);
		_differenceOfGaussian	(this, img);
		_contrastEqualization	(this, img);

		if (img.depth() != img_depth)
			img.convertTo(img, CV_MAKETYPE(img_depth, 1), elemMaxSize);
	}


	cv::Ptr<TanTriggsNormalization> TanTriggsNormalization::create()
	{
		return cv::Algorithm::create<TanTriggsNormalization>("Imgproc.TanTriggsNormalization");
	}

} // namespace cv
