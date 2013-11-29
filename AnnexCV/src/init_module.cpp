#include "precomp.h"

#include "init_module.h"
#include "internal/private.h"

#include "susan.h"
#include "tantriggs.h"


namespace cv
{

	typedef void (cv::Algorithm::*uint_setter) (unsigned);
	typedef void (cv::Algorithm::*dbl_setter)  (double);

	PROXY_CLASS(SUSAN, cv::Algorithm,
			void set_radius(unsigned r) { MyProxy::ref(this).set_radius(r); }
			void set_tparam(double   t) { MyProxy::ref(this).set_tparam(t); }
			void set_gparam(double   g) { MyProxy::ref(this).set_gparam(g); }
		);

	CV_INIT_ALGORITHM(SUSAN, "Feature2D.SUSAN",
		obj.info()->addParam(obj, "radius",		obj._radius, false, NULL, (uint_setter)&SUSANManager::set_radius);
		obj.info()->addParam(obj, "tparam",		obj._tparam, false, NULL,  (dbl_setter)&SUSANManager::set_tparam);
		obj.info()->addParam(obj, "gparam",		obj._gparam, false, NULL,  (dbl_setter)&SUSANManager::set_gparam);
		obj.info()->addParam(obj, "prefilter",	obj._prefilter);
		obj.info()->addParam(obj, "subpixel",	obj._subpixel)
	);

	typedef void (cv::Algorithm::*float_setter)  (float);

	PROXY_CLASS(TanTriggsNormalization, cv::Algorithm,
		void set_gamma	(float gamma)	{ MyProxy::ref(this).set_gamma(gamma); }
		void set_sigma0	(float sigma0)	{ MyProxy::ref(this).set_sigma0(sigma0); }
		void set_sigma1	(float sigma1)	{ MyProxy::ref(this).set_sigma1(sigma1); }
		void set_tau	(float tau)		{ MyProxy::ref(this).set_tau(tau); }
		void set_alpha	(float alpha)	{ MyProxy::ref(this).set_alpha(alpha); } );

	CV_INIT_ALGORITHM(TanTriggsNormalization, "Imgproc.TanTriggsNormalization",
		obj.info()->addParam(obj, "gamma",	obj.m_gamma,	false, NULL, (float_setter)&TanTriggsNormalizationManager::set_gamma);
		obj.info()->addParam(obj, "sigma0", obj.m_sigma[0],	false, NULL, (float_setter)&TanTriggsNormalizationManager::set_sigma0);
		obj.info()->addParam(obj, "sigma1", obj.m_sigma[1],	false, NULL, (float_setter)&TanTriggsNormalizationManager::set_sigma1);
		obj.info()->addParam(obj, "tau",	obj.m_tau,		false, NULL, (float_setter)&TanTriggsNormalizationManager::set_tau);
		obj.info()->addParam(obj, "alpha",	obj.m_alpha,	false, NULL, (float_setter)&TanTriggsNormalizationManager::set_alpha) );


	bool initModule_annex()
	{
		bool bAll = true;
		bAll &= createSUSAN_hidden()->info()					!= NULL;
		bAll &= createTanTriggsNormalization_hidden()->info()	!= NULL;
		return bAll;
	}

} // namespace cv