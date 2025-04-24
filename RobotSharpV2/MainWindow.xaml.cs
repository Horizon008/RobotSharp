using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace RobotSharpV2
{
    public partial class MainWindow : Window
    {
        private string movementDirection = "";
        private VideoCapture _capture;
        private bool _isCapturing;
        private Mat _currentFrame = new Mat();
        private System.Drawing.Point? _previousHandCenter = null;
        private ScalarArray _lowerSkinColor = new ScalarArray(new MCvScalar(0, 48, 80));
        private ScalarArray _upperSkinColor = new ScalarArray(new MCvScalar(20, 255, 255));
        private bool _calibrationMode = false;
        private int _binarizationLevel = 100;
        private DispatcherTimer _gameTimer;
        private List<Rectangle> _snakeParts = new();
        private System.Windows.Point _snakeDirection = new(1, 0);
        private int _cellSize = 20;
        private System.Windows.Point _foodPosition;
        private Rectangle _food;


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
                _capture.Set(CapProp.FrameWidth, 640);
                _capture.Set(CapProp.FrameHeight, 480);
                _capture.Set(CapProp.Fps, 30);

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
                                    MovementLabel.Text = $"Движение: {movementDirection}";
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

        private Mat FilterSkin(Mat inputFrame)
        {
            Mat hsvFrame = new Mat();
            CvInvoke.CvtColor(inputFrame, hsvFrame, ColorConversion.Bgr2Hsv);
            Mat hsvMask = new Mat();
            CvInvoke.InRange(hsvFrame, _lowerSkinColor, _upperSkinColor, hsvMask);
            Mat ycrcb = new Mat();
            CvInvoke.CvtColor(inputFrame, ycrcb, ColorConversion.Bgr2YCrCb);
            Mat ycrcbMask = new Mat();
            ScalarArray lowerYCrCb = new ScalarArray(new MCvScalar(0, 133, 77));
            ScalarArray upperYCrCb = new ScalarArray(new MCvScalar(255, 173, 127));
            CvInvoke.InRange(ycrcb, lowerYCrCb, upperYCrCb, ycrcbMask);
            Mat skinMask = new Mat();
            CvInvoke.BitwiseAnd(hsvMask, ycrcbMask, skinMask);
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(5, 5), new System.Drawing.Point(-1, -1));
            CvInvoke.MorphologyEx(skinMask, skinMask, MorphOp.Close, kernel, new System.Drawing.Point(-1, -1), 2, BorderType.Reflect, new MCvScalar(0));
            CvInvoke.MorphologyEx(skinMask, skinMask, MorphOp.Open, kernel, new System.Drawing.Point(-1, -1), 2, BorderType.Reflect, new MCvScalar(0));

            return skinMask;
        }
        private void BinarizationSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _binarizationLevel = (int)e.NewValue;
        }

        private Mat ProcessFrame(Mat inputFrame, out Mat maskFrame, out int fingerCount)
        {
            var outputFrame = inputFrame.Clone();
            maskFrame = new Mat();
            fingerCount = 0;

            Mat lab = new Mat();
            CvInvoke.CvtColor(inputFrame, lab, ColorConversion.Bgr2Lab);
            VectorOfMat labChannels = new VectorOfMat();
            CvInvoke.Split(lab, labChannels);
            CvInvoke.EqualizeHist(labChannels[0], labChannels[0]);
            CvInvoke.Merge(labChannels, lab);
            CvInvoke.CvtColor(lab, inputFrame, ColorConversion.Lab2Bgr);

            //if (_calibrationMode)
            //{
            //    System.Drawing.Rectangle roi = new System.Drawing.Rectangle(
            //        inputFrame.Width / 2 - 50,
            //        inputFrame.Height / 2 - 50,
            //        100, 100);

            //    using (Mat calibrationRoi = new Mat(inputFrame, roi))
            //    {
            //        CalibrateSkinColor(calibrationRoi);
            //    }
            //    _calibrationMode = false;
            //    MessageBox.Show("Калибровка завершена!");
            //}

            Mat skinMask = FilterSkin(inputFrame);
            CvInvoke.BitwiseAnd(inputFrame, inputFrame, outputFrame, skinMask);

            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                Mat hierarchy = new Mat();
                CvInvoke.FindContours(
                    skinMask,
                    contours,
                    hierarchy,
                    RetrType.External,
                    ChainApproxMethod.ChainApproxSimple);

                double maxArea = 0;
                VectorOfPoint largestContour = null;
                for (int i = 0; i < contours.Size; i++)
                {
                    double area = CvInvoke.ContourArea(contours[i]);
                    if (area > maxArea && area > 5000) 
                    {
                        maxArea = area;
                        largestContour = contours[i];
                    }
                }

                if (largestContour != null)
                {                   VectorOfPoint approxContour = new VectorOfPoint();
                    double epsilon = 0.01 * CvInvoke.ArcLength(largestContour, true);
                    CvInvoke.ApproxPolyDP(largestContour, approxContour, epsilon, true);


                    var moments = CvInvoke.Moments(largestContour);
                    if (moments.M00 != 0)
                    {
                        int centerX = (int)(moments.M10 / moments.M00);
                        int centerY = (int)(moments.M01 / moments.M00);
                        var currentCenter = new System.Drawing.Point(centerX, centerY);

                        CvInvoke.Circle(outputFrame, currentCenter, 5, new MCvScalar(0, 0, 255), -1);
                        UpdateMovementDirection(currentCenter);

                        fingerCount = CountFingersUsingConvexityDefects(approxContour, out List<System.Windows.Point> fingertips);

                        CvInvoke.DrawContours(outputFrame, new VectorOfVectorOfPoint(largestContour), -1, new MCvScalar(0, 255, 0), 2);
                        foreach (var pt in fingertips)
                        {
                            CvInvoke.Circle(outputFrame, new System.Drawing.Point((int)pt.X, (int)pt.Y), 6, new MCvScalar(255, 0, 0), 2);
                        }
                    }
                }

                maskFrame = skinMask.Clone();
            }

            return outputFrame;
        }

        private void CalibrateSkinColor(Mat skinRegion)
        {
            Mat hsv = new Mat();
            CvInvoke.CvtColor(skinRegion, hsv, ColorConversion.Bgr2Hsv);

            MCvScalar mean = CvInvoke.Mean(hsv);
            _lowerSkinColor = new ScalarArray(new MCvScalar(
                Math.Max(0, mean.V0 - 10),
                Math.Max(30, mean.V1 - 40),
                Math.Max(60, mean.V2 - 40)));
            _upperSkinColor = new ScalarArray(new MCvScalar(
                Math.Min(25, mean.V0 + 10),
                Math.Min(255, mean.V1 + 40),
                Math.Min(255, mean.V2 + 40)));
        }

        private int CountFingersUsingConvexityDefects(VectorOfPoint contour, out List<System.Windows.Point> fingertipPoints)
        {
            fingertipPoints = new List<System.Windows.Point>();
            if (contour.Size < 30) return 0;

 
            var hull = new VectorOfPoint();
            CvInvoke.ConvexHull(contour, hull, false);

            var defects = new VectorOfInt();
            CvInvoke.ConvexityDefects(contour, hull, defects);

            int fingerCount = 0;
            var defectArray = defects.ToArray();

            for (int i = 0; i < defectArray.Length; i += 4)
            {
                int startIdx = defectArray[i];
                int endIdx = defectArray[i + 1];
                int defectPtIdx = defectArray[i + 2];
                float depth = defectArray[i + 3] / 256f;

                if (depth > 20)
                {
                    var startPt = new System.Drawing.Point(contour[startIdx].X, contour[startIdx].Y);
                    var endPt = new System.Drawing.Point(contour[endIdx].X, contour[endIdx].Y);
                    var defectPt = new System.Drawing.Point(contour[defectPtIdx].X, contour[defectPtIdx].Y);

                    double angle = CalculateAngle(startPt, defectPt, endPt);
                    if (angle < 90) 
                    {
                        fingertipPoints.Add(new System.Windows.Point(startPt.X, startPt.Y));
                        fingertipPoints.Add(new System.Windows.Point(endPt.X, endPt.Y));
                        fingerCount++;
                    }
                }
            }


            fingerCount = fingerCount > 0 ? fingerCount + 1 : 0;

            return fingerCount;
        }

        private double CalculateAngle(System.Drawing.Point a, System.Drawing.Point b, System.Drawing.Point c)
        {
            double ab = Math.Sqrt(Math.Pow(b.X - a.X, 2) + Math.Pow(b.Y - a.Y, 2));
            double bc = Math.Sqrt(Math.Pow(b.X - c.X, 2) + Math.Pow(b.Y - c.Y, 2));
            double ac = Math.Sqrt(Math.Pow(c.X - a.X, 2) + Math.Pow(c.Y - a.Y, 2));

            return Math.Acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * (180 / Math.PI);
        }

        private void UpdateMovementDirection(System.Drawing.Point currentCenter)
        {
            if (_previousHandCenter == null)
            {
                _previousHandCenter = currentCenter;
                return;
            }

            int dx = currentCenter.X - _previousHandCenter.Value.X;
            int dy = currentCenter.Y - _previousHandCenter.Value.Y;

            if (Math.Sqrt(dx * dx + dy * dy) > 15) 
            {
                if (Math.Abs(dx) > Math.Abs(dy))
                    movementDirection = dx > 0 ? "Влево" : "Вправо";
                else
                    movementDirection = dy > 0 ? "Вниз" : "Вверх";
            }

            _previousHandCenter = currentCenter;
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
        private void SnakeGameButton_Click(object sender, RoutedEventArgs e)
        {
            StartSnakeGame();
        }

        private void StartSnakeGame()
        {
            GameCanvas.Visibility = Visibility.Visible;
            GameCanvas.Children.Clear();
            _snakeParts.Clear();

            var start = new Rectangle
            {
                Width = _cellSize,
                Height = _cellSize,
                Fill = Brushes.Lime
            };

            Canvas.SetLeft(start, 100);
            Canvas.SetTop(start, 100);
            GameCanvas.Children.Add(start);
            _snakeParts.Add(start);

            _snakeDirection = new System.Windows.Point(1, 0);
            SpawnFood();

            _gameTimer = new DispatcherTimer();
            _gameTimer.Interval = TimeSpan.FromMilliseconds(150);
            _gameTimer.Tick += GameLoop;
            _gameTimer.Start();
        }

        private void GameLoop(object sender, EventArgs e)
        {
            UpdateSnakeDirectionFromCamera();

            var head = _snakeParts.First();
            double x = Canvas.GetLeft(head) + _snakeDirection.X * _cellSize;
            double y = Canvas.GetTop(head) + _snakeDirection.Y * _cellSize;

            if (x < 0 || y < 0 || x >= GameCanvas.ActualWidth || y >= GameCanvas.ActualHeight)
            {
                GameOver();
                return;
            }

            foreach (var part in _snakeParts.Skip(1))
            {
                if (Math.Abs(Canvas.GetLeft(part) - x) < _cellSize &&
                    Math.Abs(Canvas.GetTop(part) - y) < _cellSize)
                {
                    GameOver();
                    return;
                }
            }

            var newHead = new Rectangle
            {
                Width = _cellSize,
                Height = _cellSize,
                Fill = Brushes.LimeGreen
            };

            Canvas.SetLeft(newHead, x);
            Canvas.SetTop(newHead, y);
            GameCanvas.Children.Add(newHead);
            _snakeParts.Insert(0, newHead);

            if (Math.Abs(x - _foodPosition.X) < _cellSize && Math.Abs(y - _foodPosition.Y) < _cellSize)
            {
                SpawnFood();
            }
            else
            {
                var tail = _snakeParts.Last();
                GameCanvas.Children.Remove(tail);
                _snakeParts.Remove(tail);
            }
        }

        private void UpdateSnakeDirectionFromCamera()
        {
            switch (movementDirection)
            {
                case "Вверх":
                    if (_snakeDirection != new System.Windows.Point(0, 1))
                        _snakeDirection = new System.Windows.Point(0, -1);
                    break;
                case "Вниз":
                    if (_snakeDirection != new System.Windows.Point(0, -1))
                        _snakeDirection = new System.Windows.Point(0, 1);
                    break;
                case "Влево":
                    if (_snakeDirection != new System.Windows.Point(1, 0))
                        _snakeDirection = new System.Windows.Point(-1, 0);
                    break;
                case "Вправо":
                    if (_snakeDirection != new System.Windows.Point(-1, 0))
                        _snakeDirection = new System.Windows.Point(1, 0);
                    break;
            }
        }

        private void SpawnFood()
        {
            if (_food != null)
                GameCanvas.Children.Remove(_food);

            Random rand = new Random();
            _foodPosition = new System.Windows.Point(
                rand.Next(0, (int)(GameCanvas.ActualWidth / _cellSize)) * _cellSize,
                rand.Next(0, (int)(GameCanvas.ActualHeight / _cellSize)) * _cellSize);

            _food = new Rectangle
            {
                Width = _cellSize,
                Height = _cellSize,
                Fill = Brushes.Red
            };

            Canvas.SetLeft(_food, _foodPosition.X);
            Canvas.SetTop(_food, _foodPosition.Y);
            GameCanvas.Children.Add(_food);
        }

        private void GameOver()
        {
            _gameTimer.Stop();
            MessageBox.Show("Игра окончена!");
            GameCanvas.Visibility = Visibility.Collapsed;
        }
        private void PongGameButton_Click(object sender, RoutedEventArgs e)
        {
            //StartPongGame();
        }
    }
}