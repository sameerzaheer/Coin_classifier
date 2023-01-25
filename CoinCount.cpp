// CoinCount.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h> 
#include <windows.h>

using namespace cv;
using namespace std;

Mat imgGry, imgCol;
vector<Mat> imgCol_planes;

bool isDebug = true;

void LoadImages(int num =1) {
	string fileName = "G:\\Projects\\CoinCount\\Sample" + to_string(num) + ".jpg";
	imgGry = imread(fileName, IMREAD_GRAYSCALE);
	imgCol = imread(fileName, IMREAD_COLOR);

	//downsample to less than 1200x1200
	Mat temp1 = imgGry, temp2 = imgCol;
	int iters = 0;
	while (iters < 10 && (temp1.rows > 1200 || temp1.cols > 1200)) {
		pyrDown(temp1, imgGry, Size(temp1.cols / 2, temp1.rows / 2));
		pyrDown(temp2, imgCol, Size(temp2.cols / 2, temp2.rows / 2));
		temp1 = imgGry, temp2 = imgCol;
	}

	// Separate the image in 3 planes ( B, G and R )
	split(imgCol, imgCol_planes);
	
	//namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	//imshow("Hough Circle Transform Demo", imgCol);
	//waitKey(0);
	
}

void GetHoughCircles() {
	vector<Vec3f> circles;
	HoughCircles(imgGry, circles, HOUGH_GRADIENT, 1, imgGry.rows / 20, 150, 40, 0, 0);
	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(imgGry, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(imgGry, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	/// Show your results
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", imgGry);
	waitKey(0);
}

std::vector<KeyPoint> GetBlobs() {
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	//Note most of the default parameters work just fine. However, it is bad software practice to rely 
	//on defaults as they can change from version to version, hence I've set some reasonable parameters below.

	params.filterByColor = true;
	params.blobColor = 255;	
	params.minThreshold = 100;
	params.maxThreshold = 256;	
	params.filterByArea = true;
	params.minArea = 300;	
	params.filterByCircularity = true;
	params.minCircularity = 0.85;	
	params.filterByConvexity = false;	
	params.filterByInertia = false;
 
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector->detect(imgCol, keypoints);
		
	if (isDebug) {
		// Draw detected blobs as blue circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		Mat im_with_keypoints;
		drawKeypoints(imgCol, keypoints, im_with_keypoints, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//also write sizes
		for (int i = 0; i < keypoints.size(); i++) {
			string s = to_string(i) + " : " + to_string(keypoints[i].size);
			putText(im_with_keypoints,  s , cvPoint(keypoints[i].pt.x, keypoints[i].pt.y - keypoints[i].size /2),
				FONT_HERSHEY_SIMPLEX, 0.7, cvScalar(200, 200, 250), 1, CV_AA);
		}

		imshow("keypoints", im_with_keypoints);
		waitKey(0);
	}

	return keypoints;
}

vector<double> GetHistogram(KeyPoint kp, Mat msk) {
	int histSize = 8;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	vector<Mat> histsBgr(3);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	vector<Scalar> drawCols{ Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };
	vector<double> histV;
	for (int i = 0; i < 3; i++) {//B,G,R
		calcHist(&imgCol_planes[i], 1, 0, msk, histsBgr[i], 1, &histSize, &histRange, true, false);
		normalize(histsBgr[i], histsBgr[i], 0, histImage.rows, NORM_MINMAX, -1, Mat());

		histV.push_back(histsBgr[i].at<float>(0));
		for (int j = 1; j < histSize; j++) {
			histV.push_back(histsBgr[i].at<float>(j));
			if (isDebug) {
				// Draw for each channel
				line(histImage, Point(bin_w*(j - 1), hist_h - cvRound(histsBgr[i].at<float>(j - 1))),
					Point(bin_w*(j), hist_h - cvRound(histsBgr[i].at<float>(j))),
					drawCols[i], 2, 8, 0);
			}
		}
	}

	// Display
	if (isDebug) {
		namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		imshow("calcHist Demo", histImage);
		waitKey(0);
	}

	return histV;
}

bool ConcentricCircle(KeyPoint kp, Mat msk) {
	Mat subregion;
	imgCol_planes[0].copyTo(subregion, msk);
	//imshow("subregion", subregion);
	//waitKey(0);
	int radius = kp.size / 2;
	subregion = subregion(Range(kp.pt.y - radius, kp.pt.y + radius + 1), 
						  Range(kp.pt.x - radius, kp.pt.x + radius + 1));
	if (isDebug) {
		imshow("subregion", subregion);
		waitKey(0);
	}
	vector<Vec3f> circles;
	HoughCircles(subregion, circles, HOUGH_GRADIENT, 1, subregion.rows / 2, 60, 30, radius / 3, radius-1);
	return circles.size() > 0;
}

vector<double> GetFeature(KeyPoint kp) {
	Mat msk(imgGry.rows, imgGry.cols, CV_8UC1, Scalar(0));
	//create mask
	int radius = kp.size * 0.5;
	int radius2 = radius * radius;
	int margin = 3 * 3; // to obtain only "in" pixels
	//not the most efficient way of traversing a circle...
	for (int i = kp.pt.x - radius; i <= kp.pt.x + radius; i++) {
		for (int j = kp.pt.y - radius; j <= kp.pt.y + radius; j++) {
			if (pow(abs(i - kp.pt.x), 2) + pow(abs(j - kp.pt.y), 2) + margin < radius2) {
				msk.at<uchar>(j, i) = 255;
			}
		}
	}
	if (isDebug) {
		imshow("mask", msk);
		waitKey(0);
	}
	
	vector<double> featureVector;
		
	bool hasCircles = ConcentricCircle(kp, msk);
	auto histV = GetHistogram(kp, msk);

	featureVector.push_back(kp.size);
	featureVector.push_back(hasCircles ? 1.0 : 0.0);
	featureVector.insert(featureVector.end(), histV.begin(), histV.end());
	return featureVector;
}

void WriteFeaturesToFile(string filename, vector<vector<double>> featureVs) {
	ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < featureVs.size(); i++) {
		auto featureV = featureVs[i];
		myfile << featureV[0];
		for (int j = 1; j < featureV.size(); j++) {
			myfile << "," << featureV[j];
		}
		myfile << endl;
	}
	myfile.close();

}

int main()
{
	isDebug = true;
	LoadImages(1);
	//GetHoughCircles(); //not as accurate as GetBlobs
	auto keyPoints = GetBlobs();

	//LoadImages(2);
	//keyPoints = GetBlobs();

	//LoadImages(3);
	//keyPoints = GetBlobs();

	int size = -1;
	vector<vector<double>> featureList;
	for (int i = 0; i < keyPoints.size(); i++) {
		featureList.push_back(GetFeature(keyPoints[i]));
	}

	WriteFeaturesToFile("G:\\Projects\\CoinCount\\features1.txt", featureList);
		
    return 0;
}
