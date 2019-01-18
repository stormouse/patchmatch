#pragma once

namespace util {
	using namespace std;

	float random_range(float min, float max) {
		if (min > max) { int t = min; min = max; max = t; }
		float f = (rand() % 65537) * 1.0f / 65537.0f;
		return f * (max - min) + min;
	}

	int argmax(float v[], int n) {
		int max_i = 0;
		for (int i = 1; i < n; i++)
			if (v[i] > v[max_i]) max_i = i;

		return max_i;
	}

	int argmin(float v[], int n) {
		int min_i = 0;
		for (int i = 1; i < n; i++)
			if (v[i] < v[min_i])min_i = i;
		return min_i;
	}

	int max(int a, int b) {
		return a > b ? a : b;
	}

	int min(int a, int b) { return a < b ? a : b; }
}