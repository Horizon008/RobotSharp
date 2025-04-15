using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace RobotSharpV2
{
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private bool _isCapturing;
        private Mat _currentFrame = new Mat();
        private int _binarizationLevel = 100; 

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr handle);

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void StartStopButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isCapturing)
            {
                StopCapture();
            }
            else
            {
                await StartCaptureAsync();
            }
        }

        private async Task StartCaptureAsync()
        {
            try
            {
                _capture = new VideoCapture(0);
                _isCapturing = true;
                StartStopButton.Content = "СТОП";

                await Task.Run(async () =>
                {
                    while (_isCapturing)
                    {
                        using (Mat frame = new Mat())
                        {
                            _capture.Read(frame);
                            if (!frame.IsEmpty)
                            {
                                var processedFrame = ProcessFrame(frame, out Mat maskFrame, out int fingerCount);
                                Dispatcher.Invoke(() =>
                                {
                                    UpdateUI(processedFrame);
                                    UpdateMaskUI(maskFrame);
                                    FrameRate.Text = $"Количество пальцев: {fingerCount}";
                                });
                            }
                        }
                        await Task.Delay(30);
                    }
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}");
                StopCapture();
            }
        }

        private Mat ProcessFrame(Mat inputFrame, out Mat maskFrame, out int fingerCount)
        {
            var outputFrame = inputFrame.Clone();
            maskFrame = new Mat();
            fingerCount = 0;

            
            Mat grayFrame = new Mat();
            Mat thresholdFrame = new Mat();
            Mat blurredFrame = new Mat();
            Mat medianFilteredFrame = new Mat();
            Mat dilatedFrame = new Mat();
            Mat erodedFrame = new Mat();

            try
            {
                
                CvInvoke.CvtColor(inputFrame, grayFrame, ColorConversion.Bgr2Gray);

                
                CvInvoke.Threshold(grayFrame, thresholdFrame, _binarizationLevel, 255, ThresholdType.Binary);

                //CvInvoke.AdaptiveThreshold(grayFrame, thresholdFrame, 500, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, 3);
                CvInvoke.MedianBlur(thresholdFrame, medianFilteredFrame, 5);
                thresholdFrame = medianFilteredFrame.Clone();
                var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(3, 3), new System.Drawing.Point(-1, -1));
                CvInvoke.MorphologyEx(thresholdFrame, dilatedFrame, MorphOp.Open, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Constant, new MCvScalar(0));

                CvInvoke.Dilate(thresholdFrame, dilatedFrame, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Constant, new MCvScalar(1));
                CvInvoke.Erode(dilatedFrame, erodedFrame, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Constant, new MCvScalar(1));
                thresholdFrame = erodedFrame.Clone(); 

                
                maskFrame = thresholdFrame.Clone();

                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    Mat hierarchy = new Mat();
                    CvInvoke.FindContours(
                        thresholdFrame,
                        contours,
                        hierarchy,
                        RetrType.External,
                        ChainApproxMethod.ChainApproxSimple);

           
                    for (int i = 0; i < contours.Size; i++)
                    {
                        var contour = contours[i];

                        if (CvInvoke.ContourArea(contour) > 500)
                        {
                            
                            var hull = new VectorOfPoint();
                            CvInvoke.ConvexHull(contour, hull, false);

                            
                            fingerCount += CountFingersUsingAngles(hull);
                        }
                    }

                    for (int i = 0; i < contours.Size; i++)
                    {
                        CvInvoke.DrawContours(
                            outputFrame,
                            contours,
                            i,
                            new MCvScalar(0, 255, 0),
                            2);
                    }
                }
            }
            finally
            {
                
                grayFrame.Dispose();
                blurredFrame.Dispose();
                medianFilteredFrame.Dispose();
                dilatedFrame.Dispose();
                erodedFrame.Dispose();
            }

            return outputFrame;
        }

        private int CountFingersUsingAngles(VectorOfPoint contour)
        {
            int fingerCount = 0;

            var hull = new VectorOfPoint();
            CvInvoke.ConvexHull(contour, hull, false);

            System.Drawing.Point[] hullPoints = hull.ToArray();

            
            if (hullPoints.Length < 5)
                return fingerCount;

            System.Windows.Point[] wpfHullPoints = new System.Windows.Point[hullPoints.Length];
            for (int i = 0; i < hullPoints.Length; i++)
            {
                wpfHullPoints[i] = new System.Windows.Point(hullPoints[i].X, hullPoints[i].Y);
            }

           
            for (int i = 0; i < wpfHullPoints.Length; i++)
            {
                System.Windows.Point pt1 = wpfHullPoints[i];
                System.Windows.Point pt2 = wpfHullPoints[(i + 1) % wpfHullPoints.Length];
                System.Windows.Point pt3 = wpfHullPoints[(i + 2) % wpfHullPoints.Length];

                double angle = GetAngle(pt1, pt2, pt3);

                
                if (angle > 45) 
                {
                   
                    System.Windows.Point pt4 = wpfHullPoints[(i + 3) % wpfHullPoints.Length];

                    double angleAfter = GetAngle(pt2, pt3, pt4);

                   
                    if (Math.Abs(angleAfter - 180) < 20)
                    {
                        
                        double distance = GetDistance(pt1, pt3);

         
                        if (distance < 50) 
                        {
                            fingerCount++;
                        }
                    }
                }
            }

           
            double contourPerimeter = CvInvoke.ArcLength(contour, true);
            double contourArea = CvInvoke.ContourArea(contour);

         
            if (contourArea > 500 && contourPerimeter / contourArea > 5)
            {
                
                fingerCount++;
            }

            return fingerCount;
        }

        private double GetDistance(System.Windows.Point pt1, System.Windows.Point pt2)
        {
            return Math.Sqrt(Math.Pow(pt2.X - pt1.X, 2) + Math.Pow(pt2.Y - pt1.Y, 2));
        }

        private double GetAngle(System.Windows.Point pt1, System.Windows.Point pt2, System.Windows.Point pt3)
        {
            double angle = Math.Abs(Math.Atan2(pt3.Y - pt2.Y, pt3.X - pt2.X) - Math.Atan2(pt1.Y - pt2.Y, pt1.X - pt2.X));
            if (angle > Math.PI)
            {
                angle -= Math.PI;
            }
            return angle * (180.0 / Math.PI);
        }

        private void UpdateUI(Mat frame)
        {
            CameraImage.Source = ToBitmapSource(frame);
        }

        private void UpdateMaskUI(Mat mask)
        {
            MaskImage.Source = ToBitmapSource(mask);
        }

        private BitmapSource ToBitmapSource(Mat frame)
        {
            using (var bitmap = frame.ToBitmap())
            {
                IntPtr hBitmap = bitmap.GetHbitmap();
                try
                {
                    return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                        hBitmap,
                        IntPtr.Zero,
                        Int32Rect.Empty,
                        BitmapSizeOptions.FromEmptyOptions());
                }
                finally
                {
                    DeleteObject(hBitmap);
                }
            }
        }

      
        private void BinarizationSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _binarizationLevel = (int)e.NewValue;
        }

        private void StopCapture()
        {
            _isCapturing = false;
            StartStopButton.Content = "СТАРТ";
            _capture?.Dispose();
            _currentFrame?.Dispose();
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            StopCapture();
        }
    }
}
