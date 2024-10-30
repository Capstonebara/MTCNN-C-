#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../include/Camera.h"
#include "../include/Facenet.h"
using namespace std;

int main()
{
	string facenetModel = "/Users/daoxuanbac/Personal/FaceRecognition/model/facenet/facenet.pb";
	string facenetModeltxt = "model/facenet/facenet.pbtxt";
	string mtcnnmodel = "/Users/daoxuanbac/Personal/FaceRecognition/model/mtcnn";
	string dataset = "/Users/daoxuanbac/Personal/FaceRecognition/test/dataset";
	string aligned = "/Users/daoxuanbac/Personal/FaceRecognition/test/aligned";

	MTCNNDetector* detector = new MTCNNDetector(mtcnnmodel);
	cout << "Mtcnn loaded successfully... Detecting database" << endl;
	detector->datasetAlign(dataset, aligned);
	cout << "chua loi gi ca" << endl;
	Facenet* facenet = new Facenet(facenetModel, facenetModeltxt);
	cout << "Facenet loaded successfully... Detecting database" << endl;
	facenet->datasetExtract(dataset, aligned);
	
	//Video or picture path can be imported
	//Mode: 0 camera, 1 video, 2 pictures
	int mode = 1;
	string path = "test/test.mp4";
	cout << "Please enter the video or picture path and mode (mode: 0 camera, 1 video, 2 pictures) separated by a space:" << endl;
	cin >> path >> mode;
	cout << "Press q to exit" << endl;
	Camera *camera = new Camera(path, mode);
	camera->videoShow(detector, facenet);
	

	delete detector;
	delete camera;
	delete facenet;
	return 0;
}