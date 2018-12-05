#include "qopencv.h"

Qopencv::Qopencv(QObject *parent) : QObject(parent)
{

}

QImage Qopencv::MatToQImage(Mat *Image)
{
    QImage image;
    switch (Image->type())
    {
        // 8-bit, 4 channel
        case CV_8UC4:
        {
           image = QImage( Image->data,
                         Image->cols, Image->rows,
                         static_cast<int>(Image->step),
                         QImage::Format_ARGB32 );

           return image;
        }
        // 8-bit, 3 channel
        case CV_8UC3:
        {
           image = QImage( Image->data,
                         Image->cols, Image->rows,
                         static_cast<int>(Image->step),
                         QImage::Format_RGB888 );

           return image.rgbSwapped();
        }

        // 8-bit, 1 channel
        case CV_8UC1:
        {
  #if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
           image = QImage( Image->data,
                         Image->cols, Image->rows,
                         static_cast<int>(Image->step),
                         QImage::Format_Grayscale8 );
  #else
           static QVector<QRgb>  sColorTable;

           // only create our color table the first time
           if ( sColorTable.isEmpty() )
           {
              sColorTable.resize( 256 );

              for ( int i = 0; i < 256; ++i )
              {
                 sColorTable[i] = qRgb( i, i, i );
              }
           }

           image = QImage( Image->data,
                         Image->cols, Image->rows,
                         static_cast<int>(Image->step),
                         QImage::Format_Indexed8 );

           image.setColorTable( sColorTable );
  #endif

           return image;
        }
    }
    return image;
}

Mat Qopencv::QImageToCvMat(const QImage &inImage, bool inCloneImageData)
{
    switch ( inImage.format() )
    {
       // 8-bit, 4 channel
       case QImage::Format_ARGB32:
       case QImage::Format_ARGB32_Premultiplied:
       {
          cv::Mat  mat( inImage.height(), inImage.width(),
                        CV_8UC4,
                        const_cast<uchar*>(inImage.bits()),
                        static_cast<size_t>(inImage.bytesPerLine())
                        );

          return (inCloneImageData ? mat.clone() : mat);
       }

       // 8-bit, 3 channel
       case QImage::Format_RGB32:
       {
          if ( !inCloneImageData )
          {
             qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
          }

          cv::Mat  mat( inImage.height(), inImage.width(),
                        CV_8UC4,
                        const_cast<uchar*>(inImage.bits()),
                        static_cast<size_t>(inImage.bytesPerLine())
                        );

          cv::Mat  matNoAlpha;

          cv::cvtColor( mat, matNoAlpha, cv::COLOR_BGRA2BGR );   // drop the all-white alpha channel

          return matNoAlpha;
       }

       // 8-bit, 3 channel
       case QImage::Format_RGB888:
       {
          if ( !inCloneImageData )
          {
             qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
          }

          QImage   swapped = inImage.rgbSwapped();

          return cv::Mat( swapped.height(), swapped.width(),
                          CV_8UC3,
                          const_cast<uchar*>(swapped.bits()),
                          static_cast<size_t>(swapped.bytesPerLine())
                          ).clone();
       }

       // 8-bit, 1 channel
       case QImage::Format_Indexed8:
       {
          cv::Mat  mat( inImage.height(), inImage.width(),
                        CV_8UC1,
                        const_cast<uchar*>(inImage.bits()),
                        static_cast<size_t>(inImage.bytesPerLine())
                        );

          return (inCloneImageData ? mat.clone() : mat);
       }

       default:
          qWarning() << "ASM::QImageToCvMat() - QImage format not handled in switch:" << inImage.format();
          break;
    }

    return cv::Mat();
}

void Qopencv::colorReduce(Mat &Image, int dev)
{
    cv::Mat_<cv::Vec3b>::iterator it = Image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator itend = Image.end<cv::Vec3b>();

    for( ; it != itend ; ++it)
    {
        (*it)[0] = (*it)[0]/dev*dev + dev/2;
        (*it)[1] = (*it)[1]/dev*dev + dev/2;
        (*it)[2] = (*it)[2]/dev*dev + dev/2;
    }
}

bool Qopencv::AddLogo(Mat &Image, Mat &logo)
{
    if(!logo.empty())
    {
        cv::Mat imageROI = Image(cv::Rect(20,20,logo.cols,logo.rows));
        cv::addWeighted(imageROI,1,logo,0.9,0.9,imageROI);
        //cv::Mat Mask = cv::imread("D:/1.png");
        //logo.copyTo(imageROI,Mask);
        return true;
    }
    else
    {
        return false;
    }
}

void Qopencv::salt(Mat &Image, int n)
{
#pragma omp parallel for
    for(int k=0 ; k<n ; k++)
    {
        int i = rand()%Image.cols;
        int j = rand()%Image.rows;
        unsigned char color = rand()%255;

        if(Image.channels() == 1)
        {
            Image.at<uchar>(j,i) = color;
        }
        else if (Image.channels() == 3)
        {
            Image.at<cv::Vec3b>(j,i)[0] = color;
            Image.at<cv::Vec3b>(j,i)[1] = color;
            Image.at<cv::Vec3b>(j,i)[2] = color;
        }
    }
}

int Qopencv::countCamera()
{
    int numberOfDevices = 0;
    bool noError = true;

    while (noError)
    {
        try
        {
            // Check if camera is available.
            VideoCapture videoCapture(numberOfDevices); // maybe crash if not available, hence try/catch.

            if(!videoCapture.isOpened())// if camera not exist then return number.
            {
                noError = false;
                return numberOfDevices;
            }
        }
        catch (...)
        {
            noError = false;
            return numberOfDevices;
        }

        // If above call worked, we have found another camera.
        numberOfDevices++;
    }
    return numberOfDevices;
}

std::vector<Vec3f> Qopencv::Simple_FindeCircles(Mat *Frame, Scalar _Contours_Color, bool _Show)
{
    cv::Mat Tmp;
    cv::cvtColor(*Frame,Tmp,CV_BGR2GRAY);
    cv::GaussianBlur(Tmp,Tmp,cv::Size(3,3),1.5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(Tmp,circles,CV_HOUGH_GRADIENT,
                     2,//accumulator resolution (size of the image/2)
                     30,//minimum distance between two circle
                     150,//canny high threshold
                     100,//minimum number of votes
                     5,800);//min and max radius
    //std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
    if(_Show)
    {
#pragma omp parallel for
        for(uint i=0;i<circles.size();i++)
        {
            cv::circle(*Frame,
                    cv::Point((circles[i])[0],(circles[i])[1]),
                    (circles[i])[2],
                    _Contours_Color,
                    2);
        }
    }
    return  circles;
}

std::vector<Vec4i> Qopencv::Simple_FindeLines(Mat *Frame, Scalar _Contours_Color, bool _Show)
{
    cv::Mat Tmp;
    cv::cvtColor(*Frame,Tmp,CV_RGB2GRAY);
    cv::GaussianBlur(Tmp,Tmp,cv::Size(5,5),1,5);
    std::vector<Vec4i> Lines;
    cv::HoughLinesP(Tmp,Lines,1,(3.14/180),500,600,100);
    //cv::HoughLines(Tmp,Lines,1,3.14/180,200);
    //std::vector<Vec4i>::const_iterator itl = Lines.begin();
    if(_Show)
    {
#pragma omp parallel for
        for(uint i=0;i<Lines.size();i++)
        {
            cv::Point pt1((Lines[i])[0] , (Lines[i])[1]);
            cv::Point pt2((Lines[i])[2] , (Lines[i])[3]);
            cv::line(*Frame,pt1,pt2,_Contours_Color);
        }
    }
    return Lines;
}

cv::vector<vector<Point> > Qopencv::Simple_FindContours(Mat *Frame, double _threshold, Scalar _Contours_Color, bool _Show)
{
    cv::Mat TmpImage(Frame->rows,Frame->cols,CV_8U);
    Mat canny_output;
    cv::vector<vector<cv::Point>> contours;
    cv::vector<Vec4i> hierarchy;
    cv::cvtColor(*Frame,TmpImage,cv::COLOR_BGR2GRAY);
    cv::Canny(TmpImage
              ,canny_output
              ,_threshold
              ,_threshold*2
              ,3);
    cv::findContours(canny_output
                     ,contours
                     ,hierarchy
                     ,CV_RETR_TREE
                     ,CV_CHAIN_APPROX_SIMPLE
                     ,cv::Point(0,0));
    if(_Show)
    {
#pragma omp parallel for
        for(uint i = 0; i < contours.size() ; i++)
        {
            cv::drawContours(*Frame
                             ,contours
                             ,i
                             ,_Contours_Color
                             ,2
                             ,8
                             ,hierarchy
                             ,0
                             ,cv::Point());
        }
    }
    return contours;
}

