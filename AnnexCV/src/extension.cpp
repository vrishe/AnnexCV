
#include "precomp.h"

#include "extension.h"


#define __2P_LINTERP(type, x1, y1, x2, y2, x) \
	(type(y2) - type(y1)) / (type(x2) - type(x1)) * (type(x) - type(x1)) + type(y1)


namespace cv
{

	cv::Mat diskMask(double radius)
	{
		assert(radius > 0);

		unsigned r  = static_cast<unsigned>(round(radius)), d = 2 * r + 1;
		double rpow = radius + 0.5; rpow *= rpow;
		
		cv::Mat res = cv::Mat(d, d, CV_8UC1);
		for(unsigned imax = d*d, i=0; i < imax ; ++i)
		{
			int x = i % d - r, y = i / d - r;
			res.at<uchar>(y + r,x + r) = (uchar)(static_cast<unsigned>(x*x + y*y) <= rpow);			
		}
		return res;
	}

	template<typename _ElemT>
	size_t nms_perform(const cv::Mat &src, cv::Mat &dst, _ElemT maxValue)
	{
		cv::Mat skip(cv::Size(src.cols, 2), CV_8UC1, cv::Scalar(0));
		uchar *skip_cur = skip.ptr<uchar>(0), 
			  *skip_nxt = skip.ptr<uchar>(1); 
		
		size_t maxima_count = 0;
		for (int r = 0, rmax = src.rows - 2; r < rmax; )
		{
			const _ElemT *buf_row  = src.ptr<_ElemT>(++r);

			for(int c = 0, cmax = src.cols - 2; c < cmax; )
			{
				if (skip_cur[++c]) continue;

				if (buf_row[c] <= buf_row[c + 1])
				{
					do { ++c; } while (c < cmax && buf_row[c] <= buf_row[c + 1]); 
					if (c == cmax) break;	
				}
				else if (buf_row[c] <= buf_row[c - 1]) continue;

				skip_cur[c + 1] = true;

				const _ElemT *buf_row_temp = src.ptr<_ElemT>(r + 1);
				if (buf_row[c] <= buf_row_temp[c - 1]) continue; /**/skip_nxt[c - 1] = true;
				if (buf_row[c] <= buf_row_temp[c	]) continue; /**/skip_nxt[c    ] = true;
				if (buf_row[c] <= buf_row_temp[c + 1]) continue; /**/skip_nxt[c + 1] = true;
				
				buf_row_temp = src.ptr<_ElemT>(r - 1);
				if (buf_row[c] <= buf_row_temp[c - 1]) continue;
				if (buf_row[c] <= buf_row_temp[c    ]) continue;
				if (buf_row[c] <= buf_row_temp[c + 1]) continue;

				dst.at<_ElemT>(r, c) = maxValue != _ElemT(0) ? maxValue : buf_row[c];
				++maxima_count;
			}

			std::swap(skip_cur, skip_nxt);
			memset(skip_nxt, 0, skip.step);
		}
		return maxima_count;
	}
	size_t nonMaxSuppression3x3(const cv::Mat &src, cv::Mat &dst, bool preserveMaximaValues)
	{
		CV_Assert(!src.empty() && isGraymap(src));

		cv::Mat buf;
		if (src.data == dst.data)
		{
			buf = src.clone();
		}
		else
		{
			CV_Assert( dst.empty() || dst.refcount != NULL );

			buf = src;
			dst.create(buf.size(), buf.type());	
		}
		dst.setTo(cv::Scalar::all(0));
		
		int depth = src.depth();
		switch(depth)
		{
		case CV_8U:
			return nms_perform(buf, dst, preserveMaximaValues ? 
				std::numeric_limits<uchar>::max() : std::numeric_limits<uchar>::min());
			break;
		case CV_16U:
			return nms_perform(buf, dst, preserveMaximaValues ? 
				std::numeric_limits<ushort>::max() : std::numeric_limits<ushort>::min());
			break;
		case CV_32F:
			return nms_perform(buf, dst, float(0));
			break;
		}
		return 0;
	}

	template<typename _ElemT>
	inline void dgh_estimate(const cv::Mat &src, cv::Mat &dst)
	{
		unsigned *hist = reinterpret_cast<unsigned*>(dst.data);
		for (int r = 0, rmax = src.rows; r < rmax; ++r)
		{
			const _ElemT *src_row = src.ptr<_ElemT>(r);
			for (int c = 0, cmax = src.cols; c < cmax; ++c)	
			{
				if (hist[src_row[c]] != std::numeric_limits<unsigned>::max()) ++(hist[src_row[c]]);
			}
		}
	}
	void discreteGraymapHistogram(const cv::Mat &src, cv::OutputArray dst)
	{
		int sdepth = src.depth();
		CV_Assert( isGraymap(src) && sdepth != CV_32F );

		switch (sdepth)
		{
		case CV_8U:
			dst.create(1, std::numeric_limits<uchar>::max() + 1, cv::DataType<unsigned>::type);
			dgh_estimate<uchar>(src, dst.getMat());
			break;
		case CV_16U:
			dst.create(1, std::numeric_limits<ushort>::max() + 1, cv::DataType<unsigned>::type);
			dgh_estimate<ushort>(src, dst.getMat());
			break;
		}
	}


	template<typename _ElemT>
	inline bool bin_search_perform(const std::vector<_ElemT> &sorted_vector, const _ElemT &val, std::pair<size_t, size_t> &closest_values)
	{
		size_t left = 0, right = sorted_vector.size() - 1, center;

		bool found = false;
		while (!found)
		{
			center = ((center = left + right) / 2) | (center % 2);
			if (left >= center || center >= right) break;


			const _ElemT &ref_val = sorted_vector[center];
			if (val < ref_val)
				right = center;
			else if (val > ref_val)
				left = center;
			else found = true;	
		}

		closest_values.first	= left;
		closest_values.second	= right;

		return found;
	}
	template<typename _ElemT>
	inline void thist_perform(cv::Mat &src, double plow, double phigh)
	{
		std::vector<_ElemT> img_val(src.size().area());
		std::vector<double> img_idx(img_val.size());

		for (int r = 0, rmax = src.rows; r < rmax; ++r)
			std::memcpy(&img_val[r * src.cols], src.ptr<_ElemT>(r), src.cols);
		std::sort(img_val.begin(), img_val.end());

		for (size_t i = 0, imax = img_val.size(); i < imax; ++i)
			img_idx[i] = (100. * i) / imax;

		// Binary search and truncation
		double plow_val, phigh_val;
		{
			std::pair<size_t, size_t> plow_idx, phigh_idx;
			if (!bin_search_perform(img_idx, plow, plow_idx))
				plow_val = __2P_LINTERP(double, img_idx[plow_idx.first], img_val[plow_idx.first],
					img_idx[plow_idx.second], img_val[plow_idx.second], plow);
			else plow_val = double(img_val[plow_idx.first]);

			if (!bin_search_perform(img_idx, phigh, phigh_idx))
				phigh_val = __2P_LINTERP(double, img_idx[phigh_idx.first], img_val[phigh_idx.first],
					img_idx[phigh_idx.second], img_val[phigh_idx.second], phigh);
			else phigh_val = double(img_val[phigh_idx.first]);
		}

		for (int r = 0, rmax = src.rows; r < rmax; ++r)
		{
			_ElemT *row = src.ptr<_ElemT>(r);
			for (int c = 0, cmax =src.cols; c < cmax; ++c)
				row[c] = cv::saturate_cast<_ElemT>(std::max(plow_val, std::min(double(row[c]), phigh_val)));
		}
	}
	void truncateHist(cv::Mat &src, double plow, double phigh)
	{
		CV_Assert(isGraymap(src) && plow < phigh);

		if (src.empty()) return;

		int sdepth = src.depth();
		switch (sdepth)
		{
		case CV_8U:
			thist_perform<uchar>(src, plow, phigh);
			break;

		case CV_16U:
			thist_perform<ushort>(src, plow, phigh);
			break;

		case CV_32F:
			thist_perform<float>(src, plow, phigh);
			break;
		}
	}

} // namespace cv