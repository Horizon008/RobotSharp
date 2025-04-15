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

            using (Mat grayFrame = new Mat())
            using (Mat thresholdFrame = new Mat())
            {
                CvInvoke.CvtColor(inputFrame, grayFrame, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(grayFrame, thresholdFrame, 100, 255, ThresholdType.Binary);

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
            return outputFrame;
        }

        private int CountFingersUsingAngles(VectorOfPoint contour)
        {
            int fingerCount = 0;

            var hull = new VectorOfPoint();
            CvInvoke.ConvexHull(contour, hull, false);


            System.Drawing.Point[] hullPoints = hull.ToArray();

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

                
                if (angle < 45) 
                {
                    fingerCount++;
                }
            }

            return fingerCount;
        }



        private double GetAngle(Point pt1, Point pt2, Point pt3)
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
