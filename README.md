# Qt_OpencvNeeded

you can use this library for simple use of opencv with these function:


    QImage MatToQImage(Mat *Image);// convert a Mat frame to QImage
    cv::Mat QImageToCvMat( const QImage &inImage, bool inCloneImageData = true ); // convert a QImage to Mat Frame
    void colorReduce(cv::Mat &Image, int dev=64);// Reduce Frame Color by division
    bool AddLogo(cv::Mat &Image,cv::Mat &logo);// add a logo to your Mat Frame
    void salt(cv::Mat &Image,int n);// Add salt effect
    int countCamera(void);// count how many Camera/s are detected

    std::vector<cv::Vec3f> Simple_FindeCircles(Mat *Frame, Scalar _Contours_Color, bool _Show = false);
    std::vector<Vec4i> Simple_FindeLines(Mat *Frame, Scalar _Contours_Color, bool _Show = false);
    cv::vector<vector<cv::Point>> Simple_FindContours(Mat *Frame, double _threshold
                                                      ,cv::Scalar _Contours_Color = cv::Scalar(255,255,255,255), bool Show = false);
