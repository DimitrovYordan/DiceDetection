//using OpenCvSharp;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.IO;
//using Accord.MachineLearning;
using Emgu.CV;
using Emgu.CV.Structure;

class DiceTracker
{
    static void Main()
    {
        var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromDarknet("", "");

        //string videoPath = @"C:\\Help\\demo4.mp4"; // Задай път към видеото
        //string saveDir = "DetectedFrames";
        //Directory.CreateDirectory(saveDir);

        //using var capture = new VideoCapture(videoPath);
        //if (!capture.IsOpened())
        //{
        //    Console.WriteLine("Cannot open video.");
        //    return;
        //}

        //var window = new Window("Dice Detector");
        //var subtractor = BackgroundSubtractorMOG2.Create(history: 500, varThreshold: 50, detectShadows: false);

        //int frameIndex = 0;

        //while (true)
        //{
        //    using var frame = new Mat();
        //    if (!capture.Read(frame) || frame.Empty())
        //        break;

        //    frameIndex++;
        //    Console.WriteLine($"Frame: {frameIndex}");

        //    Mat gray = new Mat();
        //    Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
        //    Cv2.GaussianBlur(gray, gray, new OpenCvSharp.Size(5, 5), 0);

        //    Mat fgMask = new Mat();
        //    subtractor.Apply(gray, fgMask);
        //    Cv2.MorphologyEx(fgMask, fgMask, MorphTypes.Open,
        //        Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)), iterations: 2);

        //    Point[][] motionContours;
        //    HierarchyIndex[] hierarchy;
        //    Cv2.FindContours(fgMask, out motionContours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        //    List<Rect> motionAreas = new();
        //    foreach (var cnt in motionContours)
        //    {
        //        if (Cv2.ContourArea(cnt) > 100 && Cv2.ContourArea(cnt) < 1000)
        //            motionAreas.Add(Cv2.BoundingRect(cnt));
        //    }

        //    List<Point2f> allKeypoints = new();

        //    foreach (var area in motionAreas)
        //    {
        //        Mat roi = new Mat(gray, area);

        //        SimpleBlobDetector.Params blobParams = new SimpleBlobDetector.Params
        //        {
        //            FilterByColor = true,
        //            BlobColor = 0,
        //            FilterByArea = true,
        //            MinArea = 10,
        //            MaxArea = 300,
        //            FilterByCircularity = true,
        //            MinCircularity = 0.7f,
        //            FilterByInertia = false,
        //            FilterByConvexity = false
        //        };

        //        using var detector = SimpleBlobDetector.Create(blobParams);
        //        KeyPoint[] keypoints = detector.Detect(roi);

        //        foreach (var kp in keypoints)
        //        {
        //            Point2f globalPt = new Point2f(kp.Pt.X + area.X, kp.Pt.Y + area.Y);
        //            allKeypoints.Add(globalPt);
        //            Cv2.Circle(frame, (int)globalPt.X, (int)globalPt.Y, 10, Scalar.LimeGreen, 2);
        //        }
        //    }

        //    // DBSCAN clustering instead of K-means
        //    if (allKeypoints.Count >= 2)
        //    {
        //        double[][] data = allKeypoints.Select(p => new double[] { p.X, p.Y }).ToArray();
        //        var dbscan = new DBSCAN(epsilon: 30, minPoints: 2);
        //        int[] labels = dbscan.Learn(data).Decide(data);

        //        Dictionary<int, List<Point2f>> diceClusters = new();
        //        for (int i = 0; i < allKeypoints.Count; i++)
        //        {
        //            int label = labels[i];
        //            if (label == -1) continue; // noise
        //            if (!diceClusters.ContainsKey(label))
        //                diceClusters[label] = new List<Point2f>();
        //            diceClusters[label].Add(allKeypoints[i]);
        //        }

        //        int dieIndex = 1;
        //        foreach (var cluster in diceClusters.Values)
        //        {
        //            if (cluster.Count < 1 || cluster.Count > 6)
        //                continue;

        //            var center = cluster.Aggregate(new Point2f(0, 0), (a, b) => a + b * (1.0f / cluster.Count));
        //            Cv2.PutText(frame, $"Die {dieIndex}: {cluster.Count}",
        //                new OpenCvSharp.Point((int)center.X, (int)center.Y),
        //                HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);

        //            foreach (var p in cluster)
        //                Cv2.Circle(frame, new OpenCvSharp.Point((int)p.X, (int)p.Y), 8, Scalar.Red, 2);

        //            dieIndex++;
        //        }
        //    }

        //    Cv2.PutText(frame, $"Frame: {frameIndex}", new OpenCvSharp.Point(10, 25),
        //        HersheyFonts.HersheySimplex, 0.8, Scalar.Yellow, 2);

        //    string frameFile = Path.Combine(saveDir, $"frame_{frameIndex:D5}.jpg");
        //    Cv2.ImWrite(frameFile, frame);

        //    window.ShowImage(frame);
        //    int key = Cv2.WaitKey(0);
        //    if (key == 27) break;
        //}

        //Cv2.DestroyAllWindows();
    }
}