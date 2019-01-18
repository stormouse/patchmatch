#include "pch.h"
#include "metric.h"
#include "util.h"

using namespace cv;
using namespace std;

#define CHANNEL_TYPE Vec3b

// double(*sim)(Mat&, Mat&, int) = metric::sim_abs_diff;  //metric::ssim;
double(*sim)(Mat&, Mat&, int) = metric::sim_diff_dxy<CHANNEL_TYPE>;  //metric::ssim;

typedef vector < vector < vector<int> > >	Vector3i;
typedef vector < vector <int> >				Vector2i;


void reconstruct(Vector3i& f, Mat& dst, Mat& ref, int patch_size);

Mat pick_patch(Mat& mat, int r, int c, int r_offset, int c_offset, int patch_size) {
	int rr = r + r_offset, rc = c + c_offset;
	return mat(Range(rr - patch_size / 2, rr + patch_size / 2), Range(rc - patch_size / 2, rc + patch_size / 2));
}

// initialize offset store
void initialize(Vector3i& f, int n_rows_dst, int n_cols_dst, int n_rows_ref, int n_cols_ref, int patch_size) {

	f.resize(n_rows_dst);
	for (int i = 0; i < n_rows_dst; i++) {
		f[i].resize(n_cols_dst);
		for (int j = 0; j < n_cols_dst; j++) 
			f[i][j].resize(2);
	}

	for (int i = 0; i < n_rows_dst; i++) {
		for (int j = 0; j < n_cols_dst; j++) {
			f[i][j][0] = int(util::random_range(0, n_rows_ref - patch_size)) - i + patch_size / 2;
			f[i][j][1] = int(util::random_range(0, n_cols_ref - patch_size)) - j + patch_size / 2;
		}
	}

}


// iterate
void patchmatch(Vector3i& f, Mat& img_dst, Mat& img_ref, int patch_size = 3, int n_iterations = 5) {

	const int n_rows_dst = img_dst.rows, n_cols_dst = img_dst.cols;
	const int n_rows_ref = img_ref.rows, n_cols_ref = img_ref.cols;
	
	/* initialize */
	cout << "initializing..." << endl;
	initialize(f, n_rows_dst, n_cols_dst, n_rows_ref, n_cols_ref, patch_size);

	/* iterate */
	int row_start, row_end, col_start, col_end, step;

	Vector2i v;  // current similarity compared with current patch offset
	v.resize(n_rows_dst); for (int i = 0; i < n_rows_dst; i++) { v[i].resize(n_cols_dst); }

	for (int i = patch_size / 2; i < n_rows_dst - patch_size / 2; i++)
		for (int j = patch_size / 2; j < n_rows_dst - patch_size / 2; j++)
		{
			auto p1 = pick_patch(img_dst, i, j, 0, 0, patch_size);
			auto p2 = pick_patch(img_ref, i, j, f[i][j][0], f[i][j][1], patch_size);
			v[i][j] = sim(p1, p2, 1);
		}


	bool reverse = false;

	for (int t = 0; t < n_iterations; t++) {
		
		//Mat progress(img_dst.rows, img_dst.cols, img_dst.type());
		//reconstruct(f, progress, img_ref, patch_size);
		//imshow("progress", progress);
		//cvWaitKey(0);

		/* propagate */
		cout << "iteration " << t + 1<< endl;

		if (reverse) {
			row_start = n_rows_dst - patch_size / 2;
			row_end = patch_size / 2;
			col_start = n_cols_dst - patch_size / 2;
			col_end = patch_size / 2;
			step = -1;
		}
		else {
			row_start = patch_size / 2;
			row_end = n_rows_dst - patch_size / 2;
			col_start = patch_size / 2;
			col_end = n_cols_dst - patch_size / 2;
			step = 1;
		}


		auto checkvalid_ref = [patch_size, n_rows_ref, n_cols_ref](int r, int c, int ro, int co)
		{
			if (r + ro < patch_size / 2) return false;
			if (c + co < patch_size / 2) return false;
			if (r + ro + patch_size / 2 >= n_rows_ref) return false;
			if (c + co + patch_size / 2 >= n_cols_ref) return false;
			return true;
		};

		for (int i = row_start; i != row_end; i += step) {
			for (int j = col_start; j != col_end; j += step) {
				double sm[3];
				Mat patch = pick_patch(img_dst, i, j, 0, 0, patch_size); 

				sm[0] = v[i][j];
				if (checkvalid_ref(i, j, f[i + step][j][0], f[i + step][j][1])) {
					Mat xpatch = pick_patch(img_ref, i, j, f[i + step][j][0], f[i + step][j][1], patch_size);
					sm[1] = sim(patch, xpatch, 1);
				}
				else sm[1] = -1e16f;
				
				if (checkvalid_ref(i, j, f[i][j + step][0], f[i][j + step][1])) {
					Mat ypatch = pick_patch(img_ref, i, j, f[i][j + step][0], f[i][j + step][1], patch_size);
					sm[2] = sim(patch, ypatch, 1);
				}
				else sm[2] = -1e16f;
				
				int k = util::argmax(sm, 3);
				v[i][j] = sm[k];

				switch (k) {
				case 1: f[i][j][0] = f[i + step][j][0]; f[i][j][1] = f[i + step][j][1]; break;
				case 2: f[i][j][0] = f[i][j + step][0]; f[i][j][1] = f[i][j + step][1]; break;
				}
			}
		}
		reverse = !reverse;
	
		/* random search */
	
		for (int i = row_start; i != row_end; i += step){
			for (int j = col_start; j != col_end; j += step) {
				int r_ws = n_rows_ref, c_ws = n_cols_ref;
				float alpha = 0.5f, exp = 0.5f;
				while (r_ws*alpha > 1 && c_ws*alpha > 1) {
					int rmin = util::max(0, int(i + f[i][j][0] - r_ws*alpha)) + patch_size / 2;
					int rmax = util::min(int(i + f[i][j][0] + r_ws*alpha), n_rows_ref - patch_size) + patch_size / 2;
					int cmin = util::max(0, int(j + f[i][j][1] - c_ws*alpha)) + patch_size / 2;
					int cmax = util::min(int(j + f[i][j][1] + c_ws*alpha), n_cols_ref - patch_size) + patch_size / 2;

					if (rmin > rmax) rmin = rmax = f[i][j][0] + i;
					if (cmin > cmax) cmin = cmax = f[i][j][1] + j;

					int r_offset = int(util::random_range(rmin, rmax)) - i;
					int c_offset = int(util::random_range(cmin, cmax)) - j;
					
					Mat patch = pick_patch(img_dst, i, j, 0, 0, patch_size);
					Mat cand = pick_patch(img_ref, i, j, r_offset, c_offset, patch_size);

					float similarity = sim(patch, cand, 1);
					if (similarity > v[i][j]) {
						v[i][j] = similarity;
						f[i][j][0] = r_offset; f[i][j][1] = c_offset;
					}
					
					alpha *= exp;
				}
			}
		}

	}
}


void reconstruct(Vector3i& f, Mat& dst, Mat& ref, int patch_size) {
	int n_rows = dst.rows, n_cols = dst.cols;
	for (int i = 0; i < n_rows; i++) {
		for (int j = 0; j < n_cols; j++) {
			int r = f[i][j][0], c = f[i][j][1];
			dst.at<CHANNEL_TYPE>(i, j) = ref.at<CHANNEL_TYPE>(i + r, j + c);
		}
	}
}


void testImage(const char *img_dst_path, const char *img_ref_path) {
	srand((unsigned int)time(NULL));
	
	Mat dst = imread(img_dst_path);
	Mat ref = imread(img_ref_path);
	Mat res = dst.clone();

	assert(dst.type() == CV_8UC3);
	
	Vector3i mapping;
	patchmatch(mapping, res, ref, 3, 5);
	reconstruct(mapping, res, ref, 5);

	cv::imwrite("data\\output.jpg", res);

	imshow("result", res);
	cv::waitKey(0);

}


int main() {
	const char *img_dst = "data\\multiple_oreo.jpg";
	const char *img_ref = "data\\single_oreo.jpg";

	testImage(img_dst, img_ref);

	return 0;
}
