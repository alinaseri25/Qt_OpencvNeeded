#ifndef QOPENCV_H
#define QOPENCV_H

#include <QObject>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <QDebug>


using namespace cv;

class Qopencv : public QObject
{
    Q_OBJECT
public:
    explicit Qopencv(QObject *parent = nullptr);

    QImage MatToQImage(Mat *Image);
    cv::Mat QImageToCvMat( const QImage &inImage, bool inCloneImageData = true );
    void colorReduce(cv::Mat &Image, int dev=64);
    bool AddLogo(cv::Mat &Image,cv::Mat &logo);
    void salt(cv::Mat &Image,int n);
    int countCamera(void);

    std::vector<cv::Vec3f> Simple_FindeCircles(Mat *Frame, Scalar _Contours_Color, bool _Show = false);
    std::vector<Vec4i> Simple_FindeLines(Mat *Frame, Scalar _Contours_Color, bool _Show = false);
    cv::vector<vector<cv::Point>> Simple_FindContours(Mat *Frame, double _threshold
                                                      ,cv::Scalar _Contours_Color = cv::Scalar(255,255,255,255), bool Show = false);

signals:

public slots:
};

#endif // QOPENCV_H
