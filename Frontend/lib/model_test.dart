import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text("SignBridge Model Test")),
        body: const Center(child: ModelTestWidget()),
      ),
    );
  }
}

class ModelTestWidget extends StatefulWidget {
  const ModelTestWidget({super.key});

  @override
  State<ModelTestWidget> createState() => _ModelTestWidgetState();
}

class _ModelTestWidgetState extends State<ModelTestWidget> {
  String result = "Loading model...";

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  void _loadModel() async {
    try {
      final interpreter = await Interpreter.fromAsset('assets/isl_twohand_mlp.tflite');
      setState(() {
        result = "✅ Model loaded successfully!";
      });
    } catch (e) {
      setState(() {
        result = "❌ Failed to load model: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Text(result, style: const TextStyle(fontSize: 20));
  }
}