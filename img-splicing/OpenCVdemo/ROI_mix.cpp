#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
using namespace cv;
using namespace std;
int main() {
	Mat inserted_book = imread("1.jpg",0);
	Mat insert_book = imread("2.jpg",0);
	if (!inserted_book.data || !insert_book.data) {
		printf("Error!\n");
		return 0;
	}
	Mat RoiInserted = inserted_book(Rect(100, 100, 100, 100));
	Mat RoiInsert = insert_book(Rect(100, 100, 100, 100));
	RoiInsert.copyTo(RoiInserted);
	imshow("example",inserted_book);
	waitKey();
	return 0;
}