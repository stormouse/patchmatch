#include "everything.h"
#include "metric.h"
#include "util.h"

using namespace cv;
using namespace std;

double(*sim)(Mat&, Mat&, int) = metric::sim_abs_diff;  //metric::ssim;

typedef vector < vector < vector<int> > >	Vector3i;
typedef vector < vector <int> >				Vector2i;

#define CHANNEL_TYPE Vec3b

void reconstruct(Vector3i& f, Mat& dst, Mat& ref, int patch_size);

Mat pick_patch(Mat& mat, int r, int c, int r_offset, int c_offset, int patch_size) {
	int rr = r + r_offset, rc = c + c_offset;
	return mat(Range(rr, rr + patch_size), Range(rc, rc + patch_size));
}

// initialize offset store
void initialize(Vector3i& f, int n_rows, int n_cols, int patch_size) {

	f.resize(n_rows);
	for (int i = 0; i < n_rows; i++) {
		f[i].resize(n_cols);
		for (int j = 0; j < n_cols; j++) 
			f[i][j].resize(2);
	}

	for (int i = 0; i < n_rows; i++) {
		for (int j = 0; j < n_cols; j++) {
			f[i][j][0] = int(util::random_range(0, n_rows - patch_size)) - i;
			f[i][j][1] = int(util::random_range(0, n_cols - patch_size)) - j;

			if (i == 0 || j == 0 || i == n_rows - 1 || j == n_cols - 1)
				f[i][j][0] = f[i][j][1] = 0;
		}
	}

}


// iterate
void patchmatch(Vector3i& f, Mat& img_dst, Mat& img_ref, int patch_size = 3, int n_iterations = 5) {

	const int n_rows = img_dst.rows, n_cols = img_dst.cols;
	
	/* initialize */
	cout << "initializing..." << endl;
	initialize(f, n_rows, n_cols, patch_size);

	
	/* iterate */
	int row_start, row_end, col_start, col_end, step;

	Vector2i v;  // current similarity compared with current patch offset
	v.resize(n_rows); for (int i = 0; i < n_rows;i++) { v[i].resize(n_cols); }

	for (int i = 0; i < n_rows - patch_size; i++)
		for (int j = 0; j < n_cols - patch_size; j++)
			v[i][j] = sim(pick_patch(img_dst, i, j, 0, 0, patch_size), 
						  pick_patch(img_ref, i, j, f[i][j][0], f[i][j][1], patch_size), 1);

	bool reverse = false;

	for (int t = 0; t < n_iterations; t++) {
		
		Mat progress(img_dst.rows, img_dst.cols, img_dst.type());
		reconstruct(f, progress, img_ref, patch_size);
		imshow("progress", progress);
		//cvWaitKey(0);

		/* propagate */
		cout << "iteration " << t + 1<< endl;

		if (reverse) {
			row_start = n_rows - patch_size - 2;
			row_end = -1;
			col_start = n_cols - patch_size - 2;
			col_end = -1;
			step = -1;
		}
		else {
			row_start = 1;
			row_end = n_rows - patch_size;
			col_start = 1;
			col_end = n_cols - patch_size;
			step = 1;
		}


		auto checkvalid = [patch_size, n_rows, n_cols](int r, int c, int ro, int co)
		{
			if (r + ro < 0) return false;
			if (c + co < 0) return false;
			if (r + ro + patch_size >= n_rows) return false;
			if (c + co + patch_size >= n_cols) return false;
			return true;
		};

		for (int i = row_start; i != row_end; i += step) {
			for (int j = col_start; j != col_end; j += step) {

				float sm[3];
				Mat patch = pick_patch(img_dst, i, j, 0, 0, patch_size); 
				// Mat ipatch = pick_patch(img_ref, i, j, f[i][j][0], f[i][j][1], patch_size);		// sim(patch, ipatch) == v[i][j]
				sm[0] = v[i][j];
				if (checkvalid(i, j, f[i - step][j][0], f[i - step][j][1])) {
					Mat xpatch = pick_patch(img_ref, i, j, f[i - step][j][0], f[i - step][j][1], patch_size);
					sm[1] = sim(patch, xpatch, 1);
				}
				else sm[1] = -1e6f;
				
				if (checkvalid(i, j, f[i][j - step][0], f[i][j - step][1])) {
					Mat ypatch = pick_patch(img_ref, i, j, f[i][j - step][0], f[i][j - step][1], patch_size);
					sm[2] = sim(patch, ypatch, 1);
				}
				else sm[2] = -1e6f;
				
				int k = util::argmax(sm, 3);
				v[i][j] = sm[k];

				switch (k) {
				case 0: break;
				case 1: f[i][j][0] = f[i - step][j][0]; f[i][j][1] = f[i - step][j][1]; break;
				case 2: f[i][j][0] = f[i][j - step][0]; f[i][j][1] = f[i][j - step][1]; break;
				default: break; // error
				}
			}
		}
		reverse = !reverse;
	
		/* random search */
	
		for (int i = row_start; i != row_end; i += step){
			for (int j = col_start; j != col_end; j += step) {
				int r_ws = n_rows, c_ws = n_cols;
				float alpha = 0.5f, exp = 0.5f;
				while (r_ws*alpha > 1 && c_ws*alpha > 1) {
					int rmin = util::max(0, int(i + f[i][j][0] - r_ws*alpha));
					int rmax = util::min(int(i + f[i][j][0] + r_ws*alpha), n_rows - patch_size);
					int cmin = util::max(0, int(j + f[i][j][1] - c_ws*alpha));
					int cmax = util::min(int(j + f[i][j][1] + c_ws*alpha), n_cols - patch_size);

					
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

	int r = n_rows - patch_size - 1;
	int c;
	for (c = 0; c < n_cols - patch_size - 1; c++) {
		Mat last_patch = pick_patch(ref, r, c, f[r][c][0], f[r][c][1], patch_size);
		for (int i = 0; i < patch_size; i++) {
			for (int j = 0; j < patch_size; j++) {
				dst.at<CHANNEL_TYPE>(r + i, c + j) = last_patch.at<CHANNEL_TYPE>(i, j);
			}
		}
	}

	c = n_cols - patch_size - 1;
	for (r = 0; r < n_rows - patch_size - 1; r++) {
		Mat last_patch = pick_patch(ref, r, c, f[r][c][0], f[r][c][1], patch_size);
		for (int i = 0; i < patch_size; i++) {
			for (int j = 0; j < patch_size; j++) {
				dst.at<CHANNEL_TYPE>(r + i, c + j) = last_patch.at<CHANNEL_TYPE>(i, j);
			}
		}
	}
}


void testImage(const char *img_dst_path, const char *img_ref_path) {
	srand((unsigned int)time(NULL));
	
	Mat dst = imread(img_dst_path);
	Mat ref = imread(img_ref_path);
	
	assert(dst.cols == ref.cols && dst.rows == ref.rows);
	assert(dst.type() == CV_8UC3);
	
	Vector3i mapping;
	patchmatch(mapping, dst, ref, 5, 5);
	reconstruct(mapping, dst, ref, 5);
	
	imshow("result", dst);
	cvWaitKey(0);

	// cvSaveImage("data\\output.jpg", &dst);
}


int main() {
	char *img_dst = "data\\dancing_a.png";
	char *img_ref = "data\\dancing_b.png";

	testImage(img_dst, img_ref);

	return 0;
}
