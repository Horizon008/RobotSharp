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
                                var processedFrame = ProcessFrame(frame);
                                Dispatcher.Invoke(() => UpdateUI(processedFrame));
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

        private Mat ProcessFrame(Mat inputFrame)
        {
            var outputFrame = inputFrame.Clone();

            using (Mat grayFrame = new Mat())
            using (Mat thresholdFrame = new Mat())
            {
                CvInvoke.CvtColor(inputFrame, grayFrame, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(grayFrame, thresholdFrame, 100, 255, ThresholdType.Binary);

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

        private void UpdateUI(Mat frame)
        {
            CameraImage.Source = ToBitmapSource(frame);
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