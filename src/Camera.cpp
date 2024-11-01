#include "../include/Camera.h"

Camera::Camera(string videopath, int mode)
{
	videoPath_ = videopath;
	
	mode_ = mode;
	if (videopath.find(".mp4") != -1 && mode==1)
		cap_ = cv::VideoCapture(videopath);
	else if(mode == 0)
		cap_ = cv::VideoCapture(0);

}

//Draw face rectangle and display face label
void Camera::DramRect(cv::Mat& img, vector<Face>& faces, vector<string>& label)
{
	for (int i = 0; i < faces.size(); i++)
	{	
		//Get Rectangular Box
		cv::Rect rect = faces[i].bbox.getSquare().getRect();
		cv::Rect screct(int(rect.tl().x), int(rect.tl().y ), 
			int(rect.width), int(rect.height));
		
		cv::Scalar color;
		
		if (label[i] != "none")color = cv::Scalar(0, 255, 0);
		else color = cv::Scalar(0, 0, 255);
		
		cv::rectangle(img, screct, color);
		double frntsize = screct.width / 100.0;
		//Show Label
		cv::putText(img, label[i], cv::Point(screct.tl().x, screct.tl().y),
			cv::FONT_HERSHEY_TRIPLEX, frntsize, color);
	}
	// std::cout << "Draw xong roi" << std::endl;
}

// void Camera::faceRecognition(cv::Mat& img, vector<Face>& faces, vector<string>& label, MTCNNDetector* detector, Facenet* facenet)
void Camera::faceRecognition(cv::Mat& img, vector<Face>& faces, vector<string>& label, MTCNNDetector* detector)
{
    double threshold = 1.1;
    for (int i = 0; i < faces.size(); i++)
    {
        try {
            // Check if face is valid
            if (faces[i].bbox.getSquare().getRect().width <= 0 || 
                faces[i].bbox.getSquare().getRect().height <= 0) {
                std::cout << "Warning: Invalid face dimensions detected" << std::endl;
                continue;
            }

            // Align face
            vector<Face> def{ faces[i] };
            detector->faceAlign(img, def, "/Users/daoxuanbac/Personal/FaceRecognition/test/detected/dt");

            // Check if aligned image exists and is valid
            cv::Mat alignedImg = cv::imread("/Users/daoxuanbac/Personal/FaceRecognition/test/detected/dt_align0.jpg");
            if (alignedImg.empty()) {
                std::cout << "Warning: Failed to load aligned face image" << std::endl;
                continue;
            }

            // Convert to the exact expected input dimensions for the neural network
            cv::Size expectedSize(160, 160);  // Standard size for many face recognition models
            cv::Mat processedImg;
            cv::resize(alignedImg, processedImg, expectedSize);

            // Ensure proper channel order and normalization
            processedImg.convertTo(processedImg, CV_32F);
            processedImg = (processedImg - 127.5) / 128.0;  // Normalize to [-1, 1] range

            // Check processed image dimensions
            if (processedImg.dims != 3) {
                cv::Mat temp;
                std::vector<cv::Mat> channels = {processedImg, processedImg, processedImg};
                cv::merge(channels, temp);
                processedImg = temp;
            }

            // Extract features with dimension checking
            // cv::Mat feat;
            // try {
            //     feat = facenet->featureExtract(processedImg);
            //     if (feat.empty()) {
            //         std::cout << "Warning: Feature extraction produced empty result" << std::endl;
            //         continue;
            //     }
            // }
            // catch (const cv::Exception& e) {
            //     std::cerr << "Feature extraction error: " << e.what() << std::endl;
            //     continue;
            // }

            // Face recognition
            // label[i] = facenet->faceRecognition(feat, threshold);
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in face recognition: " << e.what() << std::endl;
            continue;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in face recognition: " << e.what() << std::endl;
            continue;
        }
    }
}

//Read videos or pictures
// void Camera::videoShow(MTCNNDetector* detector, Facenet* facenet)
void Camera::videoShow(MTCNNDetector* detector)
{
    if (mode_ != 2 && !cap_.isOpened()) {
        std::cerr << "Error: Camera or video file not opened" << std::endl;
        return;
    }

    cv::Mat img;
    if (mode_ == 0 || mode_ == 1) {
        int frame = 0;
        vector<Face> faces;
        vector<string> label;
        int lastface_num = 0;
        
        while (cap_.read(img)) {
            if (img.empty()) {
                std::cerr << "Error: Empty frame" << std::endl;
                continue;
            }

            try {
                // Process image scaling
                int h = img.rows;
                int w = img.cols;
                float scale = 1.0;
                cv::Mat scimg;
                
                if (h > 108) {
                    scale = h / 108.0;
                    h = static_cast<int>(h / scale);
                    w = static_cast<int>(w / scale);
                    cv::resize(img, scimg, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
                } else {
                    scimg = img.clone();
                }

                frame++;
                
                // Detect faces every 5 frames
                if (frame % 5 == 0) {
                    try {
                        faces = detector->detect(scimg, 20.f, 0.5f, 0.709f);
                        
                        // Scale faces back to original size
                        for (auto& face : faces) {
                            face.faceScale(scale);
                        }

                        // Resize label vector if needed
                        if (label.size() < faces.size()) {
                            label.resize(faces.size(), "none");
                        }

                        // Perform face recognition every 2 frames if faces are detected
                        // if (frame % 2 == 0 && !faces.empty()) {
                        //     faceRecognition(img, faces, label, detector, facenet);
                        //     lastface_num = faces.size();
                        // }
                    }
                    catch (const cv::Exception& e) {
                        std::cerr << "Face detection error: " << e.what() << std::endl;
                        continue;
                    }
                }

                // Draw rectangles around detected faces
                DramRect(img, faces, label);
                
                cv::imshow("video", img);
                int key = cv::waitKey(25);
                if (key == 'q')
                    break;
            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV error in video processing: " << e.what() << std::endl;
                continue;
            }
        }
        cap_.release();
	}
	// else if (mode_ == 2)
	// {
	// 	img = cv::imread(videoPath_);
	// 	int h = img.rows;
	// 	int w = img.cols;
	// 	if (h > 480)
	// 	{
	// 		float scale = h / 480.0;
	// 		h = h / scale;
	// 		w = w / scale;
	// 		cv::resize(img, img, cv::Size(w, h));
	// 	}
		
	// 	vector<Face> faces = detector->detect(img, 20.f, 1.f,0.709f);
	// 	vector<string> label(faces.size());
	// 	std::cout << "Loi cho nay chang" << std::endl;

	// 	label.resize(faces.size(),"none");
	// 	std::cout << "Loi cho nay chang" << std::endl;
	// 	faceRecognition(img, faces, label, detector, facenet);
	
	// 	DramRect(img, faces, label);

	// 	cv::imshow("img", img);
	// 	int key = cv::waitKey(10000);
	// 	if (key == 'q')
	// 		return;
	// }
	
}

Camera::~Camera() 
{
	if (cap_.isOpened())
		cap_.release();
}