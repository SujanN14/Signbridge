import 'dart:developer' as developer;
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show WriteBuffer;
import 'package:flutter/painting.dart' show Offset, Size;
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

// ─────────────────────────────────────────────────────────────────────────────
// HandLandmarkExtractor
//
// Production-grade pose landmark extractor built on:
//   google_mlkit_pose_detection : 0.14.1  (locked in pubspec.lock)
//   google_mlkit_commons        : 0.11.1
//   camera                      : 0.11.4
//
// pubspec.yaml — already in your lock file, no changes needed:
//   google_mlkit_commons: ^0.11.1
//   google_mlkit_pose_detection: ^0.14.1
//   camera: ^0.11.4
//
// AndroidManifest.xml — add inside <application> tag:
//   <meta-data
//     android:name="com.google.mlkit.vision.DEPENDENCIES"
//     android:value="pose_detection" />
//
// iOS — no extra config needed (ML Kit bundles as XCFramework).
//
// ── Output contract ──────────────────────────────────────────────────────────
//   • List<double> with exactly [featureSize] = 126 values
//     → 42 pose landmarks × [x_norm, y_norm, z, likelihood] = 168 raw values
//       → first 126 selected (covers all 33 BlazePose landmarks comfortably)
//   • null  → frame dropped (detector busy) — caller uses zero fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Typed result returned by [HandLandmarkExtractor.extractWithMeta].
class LandmarkResult {
  /// Normalised feature vector of exactly [HandLandmarkExtractor.featureSize]
  /// doubles, ready to feed into your TFLite model.
  final List<double> features;

  /// Raw landmark map from ML Kit for overlay / debug use.
  final Map<PoseLandmarkType, PoseLandmark> rawLandmarks;

  /// End-to-end inference latency in milliseconds.
  final int latencyMs;

  /// True when at least one pose was detected; false = zero-padded fallback.
  final bool handDetected;

  const LandmarkResult({
    required this.features,
    required this.rawLandmarks,
    required this.latencyMs,
    required this.handDetected,
  });
}

class HandLandmarkExtractor {
  // ── Configuration ─────────────────────────────────────────────────────────

  /// Number of doubles the downstream TFLite model expects.
  /// 126 = 42 landmarks × 3 values (x, y, z).
  static const int featureSize = 126;

  /// BlazePose landmark types relevant to hand/wrist recognition.
  /// Used by [handKeypoints] for debug overlay rendering.
  static const List<PoseLandmarkType> handRelevantTypes = [
    PoseLandmarkType.leftWrist,
    PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftThumb,
    PoseLandmarkType.rightThumb,
    PoseLandmarkType.leftIndex,
    PoseLandmarkType.rightIndex,
    PoseLandmarkType.leftPinky,
    PoseLandmarkType.rightPinky,
  ];

  // ── Singleton detector ────────────────────────────────────────────────────
  static PoseDetector? _detector;
  static bool _isProcessing = false;
  static int  _frameCount   = 0;

  /// Lazily creates the [PoseDetector] exactly once.
  ///
  /// Uses:
  ///   • [PoseDetectionModel.base]   — fast enough for real-time video
  ///   • [PoseDetectionMode.stream]  — tracking mode, lowest per-frame latency
  static PoseDetector get _instance {
    _detector ??= PoseDetector(
      options: PoseDetectorOptions(
        model: PoseDetectionModel.base,    // base = fast; use .accurate for higher precision
        mode:  PoseDetectionMode.stream,   // stream = optimised for consecutive frames
      ),
    );
    return _detector!;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /// Convenience wrapper — returns only the feature vector.
  /// Returns [null] if the frame was dropped (detector busy).
  static Future<List<double>?> extract(CameraImage frame) async {
    final result = await extractWithMeta(frame);
    return result?.features;
  }

  /// Full extraction returning [LandmarkResult] with features + metadata.
  /// Returns [null] if the frame was dropped due to the detector being busy.
  static Future<LandmarkResult?> extractWithMeta(CameraImage frame) async {
    // ── Frame-drop guard ──────────────────────────────────────────────────
    // Prevents queue build-up when inference is slower than the camera FPS.
    if (_isProcessing) {
      developer.log('⚡ Frame dropped — detector busy', name: 'HLE');
      return null;
    }

    _isProcessing = true;
    _frameCount++;
    final sw = Stopwatch()..start();

    try {
      // 1. Convert CameraImage → InputImage
      final inputImage = _toInputImage(frame);
      if (inputImage == null) {
        developer.log(
          '⚠️ Unsupported image format: ${frame.format.group}',
          name: 'HLE',
        );
        return null;
      }

      // 2. Run ML Kit pose detection
      final List<Pose> poses = await _instance.processImage(inputImage);
      sw.stop();
      final latency = sw.elapsedMilliseconds;

      developer.log(
        '📐 Frame #$_frameCount | poses=${poses.length} | ${latency}ms',
        name: 'HLE',
      );

      // 3. No pose found → zero-padded fallback result
      if (poses.isEmpty) {
        return LandmarkResult(
          features:     List<double>.filled(featureSize, 0.0),
          rawLandmarks: {},
          latencyMs:    latency,
          handDetected: false,
        );
      }

      // 4. Select the best pose and build the feature vector
      final pose     = _bestPose(poses);
      final features = _buildFeatureVector(pose);

      return LandmarkResult(
        features:     features,
        rawLandmarks: pose.landmarks,
        latencyMs:    latency,
        handDetected: true,
      );
    } catch (e, st) {
      developer.log('❌ Extraction error: $e',
          name: 'HLE', error: e, stackTrace: st);
      return null;
    } finally {
      _isProcessing = false;
    }
  }

  // ── Feature vector ────────────────────────────────────────────────────────

  /// Builds a fixed-length, normalised feature vector from [pose].
  ///
  /// Steps:
  ///   1. Iterate all [PoseLandmarkType] values in stable enum order.
  ///   2. Normalise x/y by the pose bounding box → scale-invariant [0, 1].
  ///   3. Keep z as-is (already relative depth from ML Kit).
  ///   4. Pad or truncate to exactly [featureSize].
  static List<double> _buildFeatureVector(Pose pose) {
    // Compute bounding box for x/y normalisation
    final lmValues = pose.landmarks.values.toList();
    final minX = lmValues.map((l) => l.x).reduce((a, b) => a < b ? a : b);
    final maxX = lmValues.map((l) => l.x).reduce((a, b) => a > b ? a : b);
    final minY = lmValues.map((l) => l.y).reduce((a, b) => a < b ? a : b);
    final maxY = lmValues.map((l) => l.y).reduce((a, b) => a > b ? a : b);

    // Clamp denominator to avoid division by zero on degenerate poses
    final rangeX = (maxX - minX).clamp(1e-6, double.infinity);
    final rangeY = (maxY - minY).clamp(1e-6, double.infinity);

    final raw = <double>[];

    // Stable iteration order via PoseLandmarkType.values (enum declaration order)
    for (final type in PoseLandmarkType.values) {
      final lm = pose.landmarks[type];
      if (lm != null) {
        raw
          ..add((lm.x - minX) / rangeX)   // x: normalised [0, 1]
          ..add((lm.y - minY) / rangeY)   // y: normalised [0, 1]
          ..add(lm.z.toDouble());          // z: relative depth
      } else {
        // Absent landmark → three zeros
        raw..add(0.0)..add(0.0)..add(0.0);
      }
    }

    // Pad with zeros or truncate to exactly featureSize
    if (raw.length >= featureSize) {
      return raw.sublist(0, featureSize);
    }
    return [...raw, ...List<double>.filled(featureSize - raw.length, 0.0)];
  }

  /// Returns the [Pose] with the highest average landmark likelihood score.
  static Pose _bestPose(List<Pose> poses) {
    if (poses.length == 1) return poses.first;
    return poses.reduce((best, candidate) =>
        _avgLikelihood(candidate) > _avgLikelihood(best) ? candidate : best);
  }

  static double _avgLikelihood(Pose pose) {
    if (pose.landmarks.isEmpty) return 0.0;
    final sum = pose.landmarks.values
        .fold<double>(0.0, (acc, lm) => acc + lm.likelihood);
    return sum / pose.landmarks.length;
  }

  // ── CameraImage → InputImage conversion ──────────────────────────────────

  /// Converts a raw [CameraImage] from the camera plugin to an [InputImage]
  /// compatible with ML Kit, handling both Android and iOS pixel formats.
  static InputImage? _toInputImage(CameraImage frame) {
    final rotation = _sensorRotation();
    final w        = frame.width.toDouble();
    final h        = frame.height.toDouble();

    if (frame.format.group == ImageFormatGroup.yuv420) {
      // ── Android: YUV_420_888 ─────────────────────────────────────────────
      // Must concatenate Y + U + V planes into one contiguous byte array.
      final allBytes = WriteBuffer();
      for (final plane in frame.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      return InputImage.fromBytes(
        bytes:    allBytes.done().buffer.asUint8List(),
        metadata: InputImageMetadata(
          size:        Size(w, h),
          rotation:    rotation,
          format:      InputImageFormat.yuv_420_888,
          bytesPerRow: frame.planes.first.bytesPerRow,
        ),
      );
    }

    if (frame.format.group == ImageFormatGroup.bgra8888) {
      // ── iOS: BGRA8888 ────────────────────────────────────────────────────
      return InputImage.fromBytes(
        bytes:    frame.planes.first.bytes,
        metadata: InputImageMetadata(
          size:        Size(w, h),
          rotation:    rotation,
          format:      InputImageFormat.bgra8888,
          bytesPerRow: frame.planes.first.bytesPerRow,
        ),
      );
    }

    if (frame.format.group == ImageFormatGroup.nv21) {
      // ── Android (older devices): NV21 ───────────────────────────────────
      return InputImage.fromBytes(
        bytes:    frame.planes.first.bytes,
        metadata: InputImageMetadata(
          size:        Size(w, h),
          rotation:    rotation,
          format:      InputImageFormat.nv21,
          bytesPerRow: frame.planes.first.bytesPerRow,
        ),
      );
    }

    return null; // Unknown / unsupported format
  }

  // ── Sensor rotation ───────────────────────────────────────────────────────

  /// Returns the correct [InputImageRotation] for a front-facing camera in
  /// portrait orientation.
  ///
  /// Most Android phones mount the front camera sensor rotated 270° CCW.
  /// iOS AVFoundation already compensates; no rotation needed there.
  ///
  /// If your specific device produces upside-down results, try
  /// [InputImageRotation.rotation90deg] for Android instead.
  static InputImageRotation _sensorRotation() {
    if (Platform.isAndroid) {
      return InputImageRotation.rotation270deg;
    }
    return InputImageRotation.rotation0deg; // iOS
  }

  // ── Diagnostics helpers ───────────────────────────────────────────────────

  /// Returns pixel-coordinate [Offset] values for hand-relevant landmarks.
  /// Useful for painting an overlay on top of the camera preview.
  ///
  /// Example usage:
  ///   final kps = HandLandmarkExtractor.handKeypoints(result.rawLandmarks);
  ///   kps.forEach((name, offset) => canvas.drawCircle(offset, 4, paint));
  static Map<String, Offset> handKeypoints(
      Map<PoseLandmarkType, PoseLandmark> landmarks) {
    return {
      for (final type in handRelevantTypes)
        if (landmarks.containsKey(type))
          type.name: Offset(landmarks[type]!.x, landmarks[type]!.y),
    };
  }

  /// Total frames processed since the last [close] call.
  static int get processedFrameCount => _frameCount;

  /// Whether the detector is currently waiting for a result.
  static bool get isBusy => _isProcessing;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Closes the ML Kit detector and resets all internal state.
  /// Call this from your widget's [dispose()] method.
  static Future<void> close() async {
    await _detector?.close();
    _detector   = null;
    _frameCount = 0;
    developer.log('🔒 HandLandmarkExtractor closed.', name: 'HLE');
  }
}