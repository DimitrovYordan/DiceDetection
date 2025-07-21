using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

public class DiceRecognizer
{
    // --- FINAL PARAMETERS FOR TUNING ---
    // Adjust these values to match your video's lighting conditions

    // HSV color range for BLACK objects (dice)
    private const int HUE_MIN = 0;
    private const int HUE_MAX = 179;
    private const int SATURATION_MIN = 0;
    private const int SATURATION_MAX = 50;
    private const int VALUE_MIN = 0;
    private const int VALUE_MAX = 70;

    // Adaptive Thresholding parameters for finding dots
    private const int ADAPTIVE_THRESH_BLOCK_SIZE = 11;
    private const int ADAPTIVE_THRESH_C = 2;

    // Area range for dot contours
    private const double DOT_MIN_AREA = 20;
    private const double DOT_MAX_AREA = 500;

    // BGR color threshold for dots. A HIGH value indicates WHITE (bright dots).
    private const int DOT_COLOR_THRESHOLD = 150;

    // --- NEW GEOMETRIC FILTERS ---
    // These are more robust than simple area filters
    private const double DICE_ASPECT_RATIO_MIN = 0.5;
    private const double DICE_ASPECT_RATIO_MAX = 1.5;
    private const double DICE_SOLIDITY_MIN = 0.9; // Higher value for a tighter filter
    private const double DICE_EXTENT_MIN = 0.7; // A good starting value

    // --- END OF TUNING PARAMETERS ---

    public static void RecognizeDiceInVideo(string videoPath)
    {
        using (VideoCapture capture = new VideoCapture(videoPath))
        {
            if (!capture.IsOpened)
            {
                Console.WriteLine($"Error: Cannot open video file at: {videoPath}");
                return;
            }

            capture.Set(CapProp.PosFrames, 482);

            CvInvoke.NamedWindow("Original Frame", WindowFlags.AutoSize);
            CvInvoke.NamedWindow("White Object Mask", WindowFlags.AutoSize);
            CvInvoke.NamedWindow("Dice Dots", WindowFlags.AutoSize);

            Mat frame = new Mat();
            int frameCount = 0;

            while (capture.Read(frame))
            {
                frameCount = (int)capture.Get(CapProp.PosFrames);

                if (frame.IsEmpty || frame.Width == 0 || frame.Height == 0)
                {
                    Console.WriteLine($"Frame {frameCount}: Original frame is empty or has zero dimensions. Stopping or skipping.");
                    break;
                }

                CvInvoke.Imshow("Original Frame", frame);

                Mat hsvFrame = new Mat();
                CvInvoke.CvtColor(frame, hsvFrame, ColorConversion.Bgr2Hsv);

                if (hsvFrame.IsEmpty || hsvFrame.Width == 0 || hsvFrame.Height == 0)
                {
                    Console.WriteLine($"Frame {frameCount}: HSV frame is empty or has zero dimensions after BGR to HSV conversion. Skipping.");
                    continue;
                }

                using (VectorOfMat hsvChannels = new VectorOfMat())
                {
                    CvInvoke.Split(hsvFrame, hsvChannels);

                    if (hsvChannels.Size < 3)
                    {
                        Console.WriteLine($"Error: Split operation failed to create all 3 channels on frame {frameCount}. Skipping.");
                        continue;
                    }

                    Mat h = hsvChannels[0];
                    Mat s = hsvChannels[1];
                    Mat v = hsvChannels[2];

                    Mat colorMask = new Mat();
                    ScalarArray lowerBlackHSV = new ScalarArray(new MCvScalar(HUE_MIN, SATURATION_MIN, VALUE_MIN));
                    ScalarArray upperBlackHSV = new ScalarArray(new MCvScalar(HUE_MAX, SATURATION_MAX, VALUE_MAX));
                    CvInvoke.InRange(hsvFrame, lowerBlackHSV, upperBlackHSV, colorMask);

                    Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(5, 5), new Point(-1, -1));
                    CvInvoke.MorphologyEx(colorMask, colorMask, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
                    CvInvoke.MorphologyEx(colorMask, colorMask, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

                    CvInvoke.Imshow("White Object Mask", colorMask);

                    VectorOfVectorOfPoint potentialObjects = new VectorOfVectorOfPoint();
                    Mat hierarchy = new Mat();
                    CvInvoke.FindContours(colorMask, potentialObjects, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                    Mat outputFrame = frame.Clone();
                    List<(Rectangle boundingBox, int dotCount)> foundDice = new List<(Rectangle, int)>();

                    for (int i = 0; i < potentialObjects.Size; i++)
                    {
                        VectorOfPoint potentialContour = potentialObjects[i];
                        double area = CvInvoke.ContourArea(potentialContour);

                        // Check if the contour is a square-like object
                        Rectangle boundingBox = CvInvoke.BoundingRectangle(potentialContour);

                        if (boundingBox.Width <= 0 || boundingBox.Height <= 0)
                        {
                            continue;
                        }

                        double aspectRatio = (double)boundingBox.Width / boundingBox.Height;

                        using (VectorOfPoint convexHull = new VectorOfPoint())
                        {
                            CvInvoke.ConvexHull(potentialContour, convexHull);
                            double hullArea = CvInvoke.ContourArea(convexHull);

                            if (hullArea == 0) continue;

                            double solidity = area / hullArea;
                            double extent = area / (double)(boundingBox.Width * boundingBox.Height);

                            // Apply all three geometric filters
                            if (aspectRatio >= DICE_ASPECT_RATIO_MIN && aspectRatio <= DICE_ASPECT_RATIO_MAX &&
                                solidity >= DICE_SOLIDITY_MIN &&
                                extent >= DICE_EXTENT_MIN)
                            {
                                // ... rest of the logic for dot recognition
                                // The dot recognition logic remains the same
                                // ...

                                Mat grayFrame = new Mat();
                                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                                if (boundingBox.X < 0 || boundingBox.Y < 0 ||
                                    boundingBox.X + boundingBox.Width > grayFrame.Width ||
                                    boundingBox.Y + boundingBox.Height > grayFrame.Height)
                                {
                                    continue;
                                }

                                Mat diceRoi = new Mat(grayFrame, boundingBox);

                                Mat maskForDots = Mat.Zeros(diceRoi.Rows, diceRoi.Cols, DepthType.Cv8U, 1);
                                int maskWidth = (int)(diceRoi.Width * 0.7);
                                int maskHeight = (int)(diceRoi.Height * 0.7);
                                int maskX = (diceRoi.Width - maskWidth) / 2;
                                int maskY = (diceRoi.Height - maskHeight) / 2;
                                Rectangle maskRect = new Rectangle(maskX, maskY, maskWidth, maskHeight);

                                CvInvoke.Rectangle(maskForDots, maskRect, new MCvScalar(255), -1);

                                Mat maskedDiceRoi = new Mat();
                                CvInvoke.BitwiseAnd(diceRoi, diceRoi, maskedDiceRoi, maskForDots);

                                Mat dotsThresh = new Mat();
                                CvInvoke.AdaptiveThreshold(maskedDiceRoi, dotsThresh, 255,
                                    AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C);

                                CvInvoke.Imshow("Dice Dots", dotsThresh);

                                VectorOfVectorOfPoint dotContours = new VectorOfVectorOfPoint();
                                Mat dotHierarchy = new Mat();
                                CvInvoke.FindContours(dotsThresh, dotContours, dotHierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                                int dotCount = 0;
                                for (int j = 0; j < dotContours.Size; j++)
                                {
                                    VectorOfPoint dotContour = dotContours[j];
                                    double dotArea = CvInvoke.ContourArea(dotContour);

                                    if (dotArea > DOT_MIN_AREA && dotArea < DOT_MAX_AREA)
                                    {
                                        using (VectorOfPoint dotConvexHull = new VectorOfPoint())
                                        {
                                            CvInvoke.ConvexHull(dotContour, dotConvexHull);
                                            double dotHullArea = CvInvoke.ContourArea(dotConvexHull);

                                            if (dotHullArea > 0)
                                            {
                                                double dotSolidity = dotArea / dotHullArea;
                                                Rectangle dotBoundingBox = CvInvoke.BoundingRectangle(dotContour);

                                                if (dotBoundingBox.Width <= 0 || dotBoundingBox.Height <= 0)
                                                {
                                                    continue;
                                                }

                                                double dotAspectRatio = (double)dotBoundingBox.Width / dotBoundingBox.Height;

                                                if (dotBoundingBox.X < 0 || dotBoundingBox.Y < 0 ||
                                                    dotBoundingBox.X + dotBoundingBox.Width > frame.Width ||
                                                    dotBoundingBox.Y + dotBoundingBox.Height > frame.Height)
                                                {
                                                    continue;
                                                }

                                                Mat dotRoiOriginalFrame = new Mat(frame, dotBoundingBox);
                                                MCvScalar meanColor = CvInvoke.Mean(dotRoiOriginalFrame);

                                                if (dotSolidity > 0.8 && dotAspectRatio > 0.8 && dotAspectRatio < 1.2 && meanColor.V0 > DOT_COLOR_THRESHOLD)
                                                {
                                                    dotCount++;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (dotCount > 0 && dotCount <= 6)
                                {
                                    foundDice.Add((boundingBox, dotCount));
                                }
                            }
                        }
                    }

                    if (foundDice.Count == 2)
                    {
                        string countsString = "";
                        foreach (var die in foundDice)
                        {
                            countsString += die.dotCount + " ";
                        }
                        Console.WriteLine($"Frame {frameCount}: Found dice with counts: {countsString}");
                    }
                    else
                    {
                        Console.WriteLine($"Frame {frameCount}: Waiting for 2 dice.");
                    }
                }

                CvInvoke.WaitKey(0); // Спира на всеки кадър, за да можете да наблюдавате прозорците.
            }
        }
        CvInvoke.DestroyAllWindows();
    }

    public static void Main(string[] args)
    {
        string videoFileName = @"videoFilePath.mp4";
        RecognizeDiceInVideo(videoFileName);
        Console.WriteLine("Dice recognition finished.");
        Console.ReadKey();
    }
}