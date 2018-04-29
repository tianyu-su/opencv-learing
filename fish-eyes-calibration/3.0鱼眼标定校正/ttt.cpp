#include <opencv2\opencv.hpp>
#include <fstream>
#include <cstdlib>
using namespace std;
using namespace cv;
int main() {



	cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
	cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
	intrinsic_matrix << 360.7406034419511, 0, 311.4822234334455, 0, 362.1742734247002, 219.5196371955846, 0, 0, 1;
	distortion_coeffs << -0.487037, 1.12335, -1.18219, -0.00255861;


	//intrinsic_matrix << "[360.7406034419511, 0, 311.4822234334455;0, 362.1742734247002, 219.5196371955846;0, 0, 1]";
	cout << distortion_coeffs << endl;
	//cout << intrinsic_matrix << endl;
	system("pause");
	return 0;
}