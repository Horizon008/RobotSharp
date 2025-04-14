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
            //Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2GRAY);
            //Cv2.CvtColor(frame, frame, ColorConversionCodes.GRAY2BGR);
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
