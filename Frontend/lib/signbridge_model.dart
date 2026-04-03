import 'dart:math' as math;
import 'dart:developer' as dev;
import 'package:tflite_flutter/tflite_flutter.dart';

/// TFLite wrapper for isl_twohand_mlp.tflite
///
/// Model facts (confirmed via check_model.py):
///   Input  : [1, 126]  float32
///   Output : [1, 26]   float32  — already softmax probabilities (A–Z)
///   Ops    : 6
///
/// Usage:
///   final model = await SignBridgeModel.create();
///   final probs = model.predict(features); // List<double> length 26
class SignBridgeModel {
  late final Interpreter _interp;
  static const String _asset = 'assets/isl_twohand_mlp.tflite';
  static const int    _inSize  = 126;
  static const int    _outSize = 26;

  SignBridgeModel._();

  /// Async factory — loads and allocates the TFLite model.
  static Future<SignBridgeModel> create() async {
    final m = SignBridgeModel._();
    await m._load();
    return m;
  }

  Future<void> _load() async {
    try {
      // Use NNAPI on Android for GPU/DSP acceleration when available,
      // fall back to CPU automatically if not supported.
      final opts = InterpreterOptions()
        ..threads = 4
        ..useNnApiForAndroid = true;

      _interp = await Interpreter.fromAsset(_asset, options: opts);
      _interp.allocateTensors();

      final inShape  = _interp.getInputTensor(0).shape;
      final outShape = _interp.getOutputTensor(0).shape;
      dev.log('Loaded | in:$inShape out:$outShape', name: 'SignBridgeModel');

      // Safety checks
      assert(inShape.last  == _inSize,  'Expected input  126, got ${inShape.last}');
      assert(outShape.last == _outSize, 'Expected output  26, got ${outShape.last}');
    } catch (e) {
      dev.log('Load failed: $e', name: 'SignBridgeModel');
      rethrow;
    }
  }

  /// Run inference.
  ///
  /// [features] — 126 float64 values (wrist-normalised + StandardScaler applied)
  /// Returns 26 probabilities for letters A–Z.
  /// Index 0 = A, 1 = B, … 25 = Z.
  List<double> predict(List<double> features) {
    assert(features.length == _inSize,
        'predict() needs $_inSize features, got ${features.length}');

    // TFLite expects List<List<double>> shaped [1, 126]
    final input  = [features];
    // Output buffer [1, 26]
    final output = [List<double>.filled(_outSize, 0.0)];

    _interp.run(input, output);

    final raw = output[0]; // 26 values

    // The model's final layer is already Softmax — values already sum to ~1.0
    // Just clamp negatives (rare numerical noise) and renormalise.
    return _safeNormalise(raw);
  }

  /// Clamp any negative noise and renormalise so values sum to 1.0.
  /// Does NOT apply softmax again — model output is already probabilities.
  List<double> _safeNormalise(List<double> probs) {
    final clamped = probs.map((v) => v < 0.0 ? 0.0 : v).toList();
    final sum = clamped.fold(0.0, (a, b) => a + b);
    if (sum <= 0.0) {
      // Degenerate — return uniform
      return List<double>.filled(_outSize, 1.0 / _outSize);
    }
    return clamped.map((v) => v / sum).toList();
  }

  void dispose() {
    _interp.close();
    dev.log('Disposed', name: 'SignBridgeModel');
  }
}