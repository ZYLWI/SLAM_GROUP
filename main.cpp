//
// Created by yu on 18-12-18.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.h>
#include <chrono>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

const string img1_path = "../1.png";
const string img2_path = "../2.png";
const Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

const double fx = 520.9;
const double cx = 325.1;
const double fy = 521.0;
const double cy = 249.7;

typedef Eigen::Matrix<double, 6, 1> Vector6d;

void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches);

void pose_estmation_2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                         std::vector<DMatch> matches, Mat& R_1, Mat& t_1, Mat& R_2, Mat& t_2);

double compute_error(Mat& R, Mat& t, vector<DMatch>& matches, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2);


void bundleAdjustment(const vector<Point2d>& pc1, const vector<Point2d>& pc2, Sophus::SE3& T);
int main(int argc, char** argv){
    // read picture
    Mat img1 = imread(img1_path, CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(img2_path, CV_LOAD_IMAGE_COLOR);

    if(img1.empty() || img2.empty()){
        cerr << "can't find img" << endl;
        return 1;
    }

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints_1, keypoints_2, matches);

    Mat R_1, t_1;
    //Fundamental
    Mat R_2, t_2;
    pose_estmation_2d2d(keypoints_1, keypoints_2,matches, R_1, t_1, R_2, t_2);

    double cost_1, cost_2;
    cost_1 = compute_error(R_1, t_1, matches, keypoints_1, keypoints_2);
    cost_2 = compute_error(R_2, t_2, matches, keypoints_1, keypoints_2);

    Mat R, t;
    if(cost_1 <= cost_2){
        R = R_1, t = t_1;
    }else{
        R = R_2, t = t_2;
    }

    Eigen::Matrix3d R_;
    R_ << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(0, 0);

    Eigen::Vector3d t_;
    t_ << t.at<double>(0, 0),t.at<double>(0, 1),t.at<double>(0, 2);

    Sophus::SE3 T(R_, t_);
    vector<Point2d> pc1, pc2;
    for(int i = 0; i < matches.size(); i++){
        Point2d p1(keypoints_1[matches[i].queryIdx].pt);
        Point2d p2(keypoints_2[matches[i].trainIdx].pt);
        pc1.push_back(p1);
        pc2.push_back(p2);
    }
    bundleAdjustment(pc1, pc2, T);
    return 0;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,std::vector<DMatch>& matches){
    Mat descriptors_1, descriptors_2;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> mathcer = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    vector<DMatch> match;
    mathcer->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;

    for(int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    for(int i = 0; i < descriptors_1.rows; i++){
        if(match[i].distance <= max(2 * min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

void pose_estmation_2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                         std::vector<DMatch> matches, Mat& R_1, Mat& t_1, Mat& R_2, Mat& t_2){
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2d> points1;
    vector<Point2d> points2;

    for(int i = 0; i < (int) matches.size(); i++){
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //compute Fundamental Matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);

    Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);

    recoverPose(essential_matrix, points1, points2, R_1, t_1, focal_length, principal_point);
    recoverPose(homography_matrix, points1, points2, R_2, t_2, focal_length, principal_point);
}

double compute_error(Mat& R, Mat& t, vector<DMatch>& matches, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2){
    double cost = 0.0;
    Eigen::Vector2d error = Eigen::Vector2d::Zero();

    for(int i = 0; i < matches.size(); i++) {
        Point2d Pc1(keypoints_1[matches[i].queryIdx].pt.x, keypoints_1[matches[i].queryIdx].pt.y);
        Point2d Pc2(keypoints_2[matches[i].trainIdx].pt.x, keypoints_2[matches[i].trainIdx].pt.y);

        Point3d Pw;
        Pw.z = 1;
        Pw.x = (Pc1.x - cx) * 1 / fx;
        Pw.y = (Pc1.y - cy) * 1 / fy;

        Mat Pcw = R * (Mat_<double>(3, 1) << Pw.x, Pw.y, Pw.z) + t;
        Point3d Pw2(Pcw.at<double>(0, 0), Pcw.at<double>(1, 0), Pcw.at<double>(2, 0));
        Point2d Pc2_;
        Pc2_.x = (fx * Pw2.x) / Pw2.z + cx;
        Pc2_.y = (fy * Pw2.y) / Pw2.z + cy;

        error = Eigen::Vector2d(Pc2.x, Pc2.y) - Eigen::Vector2d(Pc2_.x, Pc2_.y);
        cost += error.transpose() * error;
    }
    return cost;
}

void bundleAdjustment(const vector<Point2d>& pc1, const vector<Point2d>& pc2, Sophus::SE3& T1){
    Sophus::SE3 T;
    int iterators = 100;
    Mat img;
    Mat img1 = imread(img1_path, CV_LOAD_IMAGE_COLOR);
    double lastCost = 0.0, cost = 0.0;
    for(int iter = 0; iter < iterators; iter++){
        Mat image_clone = img.clone();
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d g = Vector6d::Zero();
        cost = 0.0;
        for(int i = 0; i < pc1.size(); i++){
            cv::circle(image_clone, pc1[i], 5, cv::Scalar(0, 0, 250), 2);
            Point3d Pw;
            Pw.z = 1;
            Pw.x = (pc1[i].x - cx) * 1 / fx;
            Pw.y = (pc1[i].y - cy) * 1 / fy;

            Eigen::Vector3d P(Pw.x, Pw.y, Pw.z);
            Eigen::Vector3d Pc = T * P;

            double X = Pc[0], Y = Pc[1], Z = Pc[2];
            Eigen::Vector2d e;
            e(0, 0) = pc2[i].x - fx * X / Z - cx;
            e(0, 1) = pc2[i].y - fy * Y / Z - cy;

            cost += e.matrix().transpose() * e;

            string text;
            stringstream stringstream1;
            stringstream1 << cost;
            stringstream1 >> text;

           // cv::putText(image_clone, text, pc1[i], FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
           // cv::imshow("test", image_clone);

            Eigen::Matrix<double, 2, 6> J;
            double Z_2 = pow(Z, 2), X_2 = pow(X, 2), Y_2 = pow(Y, 2);

            J(0, 0) = -1 * fx/Z;
            J(0, 1) = 0;
            J(0, 2) = fx*X/Z_2;
            J(0, 3) = fx*X*Y/Z_2;
            J(0, 4) = -fx - fx*X_2/Z_2;
            J(0, 5) = fx*Y/Z;

            J(1, 0) = 0;
            J(1, 1) = -fy/Z;
            J(1, 2) = fy*Y/Z_2;
            J(1, 3) = fy + fy*Y_2/Z_2;
            J(1, 4) = -fy*Y*X/Z_2;
            J(1, 5) = -fy*X/Z;

            H += J.transpose() * J;
            g += -J.transpose() * e;
        }
        //sleep(5000);
        //cv::waitKey(0);
        Vector6d dx;
        dx = H.ldlt().solve(g);
        T = Sophus::SE3::exp(dx) * T;

        if(isnan(dx[0])){
            cerr << "result is nan!" << endl;
            break;
        }

        if(iter > 0 && cost >= lastCost){
            cerr << "cost: " << cost << ", lastcost: " << lastCost << endl;
            break;
        }

        lastCost = cost;
        cout << "iteration " << iter << " cost = " << cout.precision(12) << cost ;
        //cout << " estimated pose: \n" << T.matrix() << endl;
    }
    cout << "estimated pose: \n" << T.matrix() << endl;
}

