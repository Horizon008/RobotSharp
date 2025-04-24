using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using System;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace RobotSharp
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture _videoCapture;
        private bool _isStreaming;
        private Mat _frame;

        private BackgroundSubtractorMOG2 _bgSubtractor;
        private Scalar _lowerSkin = new Scalar(0, 48, 80);
        private Scalar _upperSkin = new Scalar(20, 255, 255);
        private Mat _morphKernel;
        private int _fingerCount;
        private DateTime _lastGestureTime = DateTime.Now;
        public MainWindow()
        {
            InitializeComponent();
            Closing += MainWindow_Closing;
        }

        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            StopCamera();
        }

        private async void StartStopButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_isStreaming)
                {
                    StopCamera();
                    return;
                }

                _isStreaming = true;
                StartStopButton.Content = "ОСТАНОВИТЬ";

                await Task.Run(() =>
                {
                    using (_videoCapture = new VideoCapture(0))
                    {
                        _videoCapture.Set(VideoCaptureProperties.FrameWidth, 640);
                        _videoCapture.Set(VideoCaptureProperties.FrameHeight, 480);

                        while (_isStreaming && _videoCapture.IsOpened())
                        {
                            using (var frame = new Mat())
                            {
                                if (!_videoCapture.Read(frame) || frame.Empty()) break;

                                ProcessFrame(frame);

                                Dispatcher.Invoke(() =>
                                {
                                    CameraImage.Source = BitmapSourceConverter.ToBitmapSource(frame);
                                });
                            }
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}");
            }
            finally
            {
                StopCamera();
            }
        }

        private void ProcessFrame(Mat frame)
        {
            try
            {
                Cv2.Resize(frame, frame, new OpenCvSharp.Size(320, 240));
                Cv2.GaussianBlur(frame, frame, new OpenCvSharp.Size(3, 3), 0);
                using var skinMask = GetSkinMask(frame);
                using var fgMask = new Mat();
                _bgSubtractor ??= BackgroundSubtractorMOG2.Create(500, 16, false);
                _bgSubtractor.Apply(frame, fgMask);
                Cv2.BitwiseAnd(skinMask, fgMask, skinMask);
                FindHandContours(frame, skinMask);
                AnalyzeGestures();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка обработки: {ex.Message}");
            }
        }

        private Mat GetSkinMask(Mat frame)
        {
            using var hsv = new Mat();
            Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);
            Cv2.InRange(hsv, _lowerSkin, _upperSkin, hsv);

            Cv2.MorphologyEx(hsv, hsv, MorphTypes.Open, _morphKernel);
            Cv2.Dilate(hsv, hsv, _morphKernel, iterations: 2);

            return hsv.Clone();
        }

        private void FindHandContours(Mat frame, Mat mask)
        {
            var contours = Cv2.FindContoursAsArray(mask,
                RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);

            foreach (var contourSet in contours.OrderByDescending(c => Cv2.ContourArea(c)))
            {
                if (Cv2.ContourArea(contourSet) < 2000) continue;

                var hull = Cv2.ConvexHull(contourSet);
                Cv2.DrawContours(frame, new[] { hull }, -1, Scalar.Green, 2);

                _fingerCount = CountFingers(new[] { contourSet });
                break;
            }
        }

        private int CountFingers(OpenCvSharp.Point[][] contours)
        {
            try
            {
                if (contours == null || contours.Length == 0) return 0;

                var contour = contours[0];
                var hullIndices = Cv2.ConvexHullIndices(contour);
                if (hullIndices.Length < 3) return 0;

                var defects = Cv2.ConvexityDefects(contour, hullIndices);
                if (defects == null || defects.Length == 0) return 0;

                int validDefects = 0;
                foreach (var defect in defects)
                {
                    var start = contour[defect[0]];   
                    var end = contour[defect[1]];
                    var far = contour[defect[2]];


                    double a = Math.Sqrt(Math.Pow(start.X - far.X, 2) + Math.Pow(start.Y - far.Y, 2));
                    double b = Math.Sqrt(Math.Pow(end.X - far.X, 2) + Math.Pow(end.Y - far.Y, 2));
                    double angle = Math.Acos((a * a + b * b - Math.Pow(start.X - end.X, 2) - Math.Pow(start.Y - end.Y, 2)) / (2 * a * b));

                    if (angle < 1.0) validDefects++;
                }
                return validDefects > 0 ? validDefects + 1 : 0;
            }
            catch(Exception ex) {

            }
            return 0;
        }

        private void AnalyzeGestures()
        {
            if ((DateTime.Now - _lastGestureTime).TotalSeconds < 2) return;

            Dispatcher.Invoke(() =>
            {
                switch (_fingerCount)
                {
                    case 1:
                        StartStopButton_Click(null, null);
                        break;
                    case 5:
                        Close();
                        break;
                }
            });
            UpdateUI();
            _lastGestureTime = DateTime.Now;
        }

        private void UpdateUI()
        {
            GestureStatus.Text = $"Обнаружено пальцев: {_fingerCount}";
            FrameRate.Text = $"FPS: {CalculateFPS()}";
        }

        private double CalculateFPS()
        {
            var elapsed = DateTime.Now - _lastGestureTime;
            return elapsed.TotalSeconds > 0 ? 1 / elapsed.TotalSeconds : 0;
        }

        private void StopCamera()
        {
            _isStreaming = false;
            StartStopButton.Content = "СТАРТ";
            _videoCapture?.Dispose();
            _frame?.Dispose();
        }
    }
}
