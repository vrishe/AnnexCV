#pragma once

#ifndef __ANNEX_TANTRIGGS_H__
#define __ANNEX_TANTRIGGS_H__


namespace cv
{

	class TanTriggsNormalization : public cv::Algorithm
	{
	private:

		// Algoritm parameters
		float	m_gamma;
		float	m_sigma[2];
		float	m_tau;
		float	m_alpha;


		// Algorithm sequence
		static void _gammaCorrection		(TanTriggsNormalization *alg, cv::Mat &img);
		static void _differenceOfGaussian	(TanTriggsNormalization *alg, cv::Mat &img);
		static void _contrastEqualization	(TanTriggsNormalization *alg, cv::Mat &img);


	public:

		TanTriggsNormalization(float gamma = .2f, float sigma0 = 1.0f, float sigma1 = 2.0f, float tau = 10.f, float alpha = .1f) 
		{
			set_gamma	(gamma);
			set_sigma0	(sigma0);
			set_sigma1	(sigma1);
			set_tau		(tau);
			set_alpha	(alpha);
		}

		// Algortim setters
		void set_gamma	(float gamma);
		void set_sigma0	(float sigma0);
		void set_sigma1	(float sigma1);
		void set_tau	(float tau);
		void set_alpha	(float alpha);

		// Normalization chain
		void apply(cv::Mat &img);

		// Algorithm info
		cv::AlgorithmInfo* info() const;

		static cv::Ptr<TanTriggsNormalization> create();
	};

} // namespace cv


#endif // #ifndef __ANNEX_TANTRIGGS_H__
