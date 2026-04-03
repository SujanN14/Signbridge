import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import 'auth_screen.dart';
import 'signbridge_model.dart';

enum _CamState { off, starting, on, stopping, error }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {

  // ── Palette ────────────────────────────────────────────────────────────────
  static const _bg      = Color(0xFFF8F9FF);
  static const _primary = Color(0xFF4F6EF7);
  static const _purple  = Color(0xFF7C3AED);
  static const _dark    = Color(0xFF1A1D2E);
  static const _grey    = Color(0xFF8890AD);
  static const _green   = Color(0xFF22C55E);
  static const _red     = Color(0xFFE53E6D);
  static const _amber   = Color(0xFFF59E0B);
  static const _cyan    = Color(0xFF0EA5E9);

  // ── Model ──────────────────────────────────────────────────────────────────
  SignBridgeModel? _model;
  bool   _modelLoading = true;
  String _modelError   = '';

  // ── MediaPipe ─────────────────────────────────────────────────────────────
  HandLandmarkerPlugin? _handPlugin;

  // ── Camera ─────────────────────────────────────────────────────────────────
  List<CameraDescription> _cameras  = [];
  CameraController?       _cam;
  _CamState               _camState = _CamState.off;
  String                  _camError = '';
  bool                    _isDetecting = false;

  // ── Live prediction ────────────────────────────────────────────────────────
  String     _letter    = '';
  double     _conf      = 0.0;
  bool       _handFound = false;
  List<Hand> _hands     = [];

  // ── Skeleton display rotation ──────────────────────────────────────────────
  // FIX: _rot is read dynamically inside _processFrame via `this._rot`,
  // so changing _rotIdx always takes effect on the very next frame.
  // _SkeletonPainter now receives _rot and applies a coordinate transform
  // so the overlay actually moves to match the rotation value.
  static const List<int> _rotations = [0, 90, 180, 270];
  int _rotIdx = 1; // default 90°
  int get _rot => _rotations[_rotIdx];

  // ── FIX: Change rotation and immediately restart stream so detect() 
  //    picks up the new rotation value for its internal frame processing ──────
  void _changeRotation(int delta) async {
    setState(() => _rotIdx = (_rotIdx + delta + _rotations.length) % _rotations.length);
    // Restart stream with new rotation so hand_landmarker processes frames
    // at the new angle — this fixes skeleton not responding to button taps
    if (_camState == _CamState.on && _cam != null) {
      try {
        await _cam!.stopImageStream();
        await _cam!.startImageStream(_processFrame);
      } catch (e) {
        debugPrint('rotation restart: $e');
      }
    }
  }

  // ── Word builder ───────────────────────────────────────────────────────────
  String _currentWord   = '';
  String _lastAdded     = '';
  int    _lastAddedTime = 0;
  static const int _cooldownMs = 1200;

  // ── Prediction stability buffers ──────────────────────────────────────────
  static const int    _bufferSize          = 7;
  static const double _confidenceThreshold = 0.70;
  final List<String>  _predBuffer          = [];
  final List<double>  _confBuffer          = [];
  List<double>?       _emaProbs;

  // ── StandardScaler ────────────────────────────────────────────────────────
  static const List<double> _scalerMean = [
    0.0000000000, 0.0000000000, 0.0000000000,
    0.1084681654,-0.2176748675,-0.0638308577,
    0.2621829557,-0.4480067815,-0.1437947918,
    0.4032417573,-0.5708807857,-0.2260575994,
    0.4941554229,-0.6397139098,-0.3041629524,
    0.3101029048,-0.7333337974,-0.1861621800,
    0.4693131170,-0.9275093914,-0.3314401281,
    0.5409644014,-1.0114832403,-0.4217244866,
    0.5816851290,-1.0816660893,-0.4749080020,
    0.3223572246,-0.6466416408,-0.2278974780,
    0.5377174445,-0.7162734388,-0.3659207953,
    0.5841171763,-0.6791343273,-0.4107447981,
    0.5971448089,-0.6782690121,-0.4284130291,
    0.3196050121,-0.4984959758,-0.2728374550,
    0.5080159605,-0.5116959797,-0.3919139156,
    0.5173165800,-0.4423037182,-0.3766451462,
    0.5031633904,-0.4194737970,-0.3482829154,
    0.3066706226,-0.3216600845,-0.3233755110,
    0.4456347422,-0.3316983836,-0.3987171916,
    0.4566361508,-0.2983114114,-0.3678713195,
    0.4485560543,-0.2897738312,-0.3308760695,
    0.0000000000, 0.0000000000, 0.0000000000,
   -0.1022560962,-0.0896503693,-0.0364519953,
   -0.1944252495,-0.1705143216,-0.0743841451,
   -0.2623257064,-0.2070566049,-0.1113759345,
   -0.3041294103,-0.2238558872,-0.1487726336,
   -0.1593535771,-0.2847952302,-0.0857696838,
   -0.2553771832,-0.3231485254,-0.1451793588,
   -0.2862976724,-0.3301448166,-0.1828638054,
   -0.2958133185,-0.3435529072,-0.2067564948,
   -0.1219170288,-0.2401501550,-0.0987595792,
   -0.2433102804,-0.2164088443,-0.1533204168,
   -0.2597055874,-0.1802267658,-0.1706842653,
   -0.2492865616,-0.1746112066,-0.1807614040,
   -0.0903801993,-0.1715413263,-0.1141845775,
   -0.2047584735,-0.1372755879,-0.1615789349,
   -0.2155745935,-0.1118827606,-0.1573608919,
   -0.2025527811,-0.1141786926,-0.1514130272,
   -0.0657085425,-0.0930293081,-0.1330403705,
   -0.1566043256,-0.0658336533,-0.1697046903,
   -0.1705244625,-0.0541531210,-0.1639381221,
   -0.1629398634,-0.0600330404,-0.1546154633,
  ];

  static const List<double> _scalerScale = [
    1.0000000000, 1.0000000000, 1.0000000000,
    0.2962303829, 0.2573273078, 0.1172992160,
    0.5185442144, 0.4435367933, 0.1702074704,
    0.6712618067, 0.5992586499, 0.2075593582,
    0.7790185019, 0.7249462451, 0.2498439220,
    0.5359131521, 0.3823236525, 0.2074610784,
    0.7789192845, 0.6173683139, 0.2595153637,
    0.8858785645, 0.7604440538, 0.2824086624,
    0.9647412375, 0.8738327530, 0.2988333175,
    0.4890500934, 0.3906028153, 0.1851015778,
    0.7353867499, 0.6632934185, 0.2411855803,
    0.7969281269, 0.7743208871, 0.2459287737,
    0.8430891722, 0.8546874173, 0.2519935889,
    0.4629691939, 0.4306772247, 0.1800550923,
    0.6813729318, 0.6630928471, 0.2266022109,
    0.7269577828, 0.6810650157, 0.2186772951,
    0.7644006614, 0.6782404398, 0.2214344216,
    0.4619232859, 0.4835301231, 0.1979475917,
    0.6159084232, 0.6472817015, 0.2244920463,
    0.6512270180, 0.6612282174, 0.2208682492,
    0.6820740830, 0.6572214811, 0.2278275931,
    1.0000000000, 1.0000000000, 1.0000000000,
    0.1723874751, 0.2223212426, 0.0855563137,
    0.3305150263, 0.4042231929, 0.1416087628,
    0.4559874235, 0.5286814661, 0.1901080711,
    0.5442727231, 0.6203598126, 0.2421348449,
    0.3534954771, 0.4891173217, 0.1806426167,
    0.5357060167, 0.6595770020, 0.2518091026,
    0.5960846417, 0.7435571207, 0.2928207323,
    0.6354286585, 0.8115429242, 0.3198183285,
    0.3322021411, 0.4502762152, 0.1783628468,
    0.5227448927, 0.5934017781, 0.2461421918,
    0.5539529439, 0.6167670888, 0.2640094164,
    0.5656029628, 0.6436297264, 0.2789137410,
    0.3140061359, 0.4152758872, 0.1839467112,
    0.4860252747, 0.5509140255, 0.2454857744,
    0.5086556661, 0.5647056920, 0.2421116360,
    0.5150057786, 0.5783329494, 0.2417926939,
    0.3046277528, 0.3982965192, 0.2011309980,
    0.4311358045, 0.5091017429, 0.2506476872,
    0.4479199854, 0.5233224696, 0.2466129428,
    0.4504275365, 0.5292001686, 0.2416948571,
  ];

  // ── How-it-works ───────────────────────────────────────────────────────────
  late AnimationController _howCtrl;
  int  _activeStep  = 0;
  bool _howExpanded = false;

  static const _steps = [
    (icon: Icons.videocam_rounded,          color: Color(0xFF4F6EF7),
     title: 'Open Camera',
     desc:  'Tap "Start Camera" — point back camera at your hands.',
     art:   Icons.phone_android_rounded),
    (icon: Icons.accessibility_new_rounded, color: Color(0xFF7C3AED),
     title: 'Position Hands',
     desc:  'Hold both hands at chest level, fully visible.',
     art:   Icons.pan_tool_rounded),
    (icon: Icons.gesture_rounded,           color: Color(0xFF0EA5E9),
     title: 'Make the Sign',
     desc:  'Hold steady — green skeleton confirms detection.',
     art:   Icons.sign_language_rounded),
    (icon: Icons.translate_rounded,         color: Color(0xFF22C55E),
     title: 'Build Words',
     desc:  'Letters added automatically. Tap Clear to reset.',
     art:   Icons.chat_bubble_rounded),
  ];

  // ── Animations ─────────────────────────────────────────────────────────────
  late AnimationController _pageAnim, _pulseAnim, _letterAnim;
  late Animation<double>   _fadeIn, _pulse, _letterScale;
  late Animation<Offset>   _slideIn;

  // ──────────────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _setupAnims();
    _loadModel();
    _discoverCameras();
  }

  void _setupAnims() {
    _pageAnim    = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 800));
    _fadeIn      = CurvedAnimation(parent: _pageAnim, curve: Curves.easeOut);
    _slideIn     = Tween<Offset>(begin: const Offset(0, .05), end: Offset.zero)
        .animate(CurvedAnimation(parent: _pageAnim, curve: Curves.easeOut));
    _pulseAnim   = AnimationController(vsync: this,
        duration: const Duration(seconds: 2))..repeat(reverse: true);
    _pulse       = Tween<double>(begin: .88, end: 1.0)
        .animate(CurvedAnimation(parent: _pulseAnim, curve: Curves.easeInOut));
    _letterAnim  = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 350));
    _letterScale = Tween<double>(begin: .4, end: 1.0)
        .animate(CurvedAnimation(parent: _letterAnim,
            curve: Curves.elasticOut));
    _howCtrl     = AnimationController(vsync: this,
        duration: const Duration(seconds: 3))
      ..addStatusListener((s) {
          if (s == AnimationStatus.completed && _howExpanded) {
            _howCtrl.reset();
            setState(() => _activeStep = (_activeStep + 1) % _steps.length);
            _howCtrl.forward();
          }
        });
    _pageAnim.forward();
  }

  Future<void> _loadModel() async {
    setState(() { _modelLoading = true; _modelError = ''; });
    try {
      final m = await SignBridgeModel.create();
      if (!mounted) return;
      setState(() { _model = m; _modelLoading = false; });
    } catch (e) {
      if (!mounted) return;
      setState(() { _modelLoading = false; _modelError = 'Model error: $e'; });
    }
  }

  Future<void> _discoverCameras() async {
    try { _cameras = await availableCameras(); }
    catch (e) { debugPrint('cam: $e'); }
  }

  Future<void> _startCamera() async {
    if (_cameras.isEmpty) {
      setState(() { _camState = _CamState.error; _camError = 'No cameras found.'; });
      return;
    }
    setState(() { _camState = _CamState.starting; _camError = ''; });
    try {
      final desc = _cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => _cameras.first);
      final ctrl = CameraController(desc, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
      await ctrl.initialize();
      if (!mounted) { await ctrl.dispose(); return; }
      _handPlugin = HandLandmarkerPlugin.create(numHands: 2);
      _cam = ctrl;
      setState(() => _camState = _CamState.on);
      // FIX: _processFrame is a method reference on `this`.
      // Inside it, `_rot` is a getter → always reads the current _rotIdx.
      // Restarting the stream in _changeRotation ensures detect() also
      // gets the new rotation on the very next frame.
      await _cam!.startImageStream(_processFrame);
    } catch (e) {
      if (!mounted) return;
      setState(() { _camState = _CamState.error; _camError = e.toString(); });
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  FRAME PIPELINE
  // ═══════════════════════════════════════════════════════════════════════════
  void _processFrame(CameraImage image) {
    if (_isDetecting || _model == null || _handPlugin == null) return;
    _isDetecting = true;
    try {
      // `_rot` is a getter — always reads the CURRENT _rotIdx value.
      // After _changeRotation() restarts the stream, the new _rot is
      // immediately used for every subsequent frame.
      final hands = _handPlugin!.detect(image, _rot);

      if (hands.isEmpty) {
        _emaProbs = null;
        if (mounted) setState(() { _handFound = false; _hands = []; });
        return;
      }

      final sorted = List<Hand>.from(hands)
        ..sort((a, b) => a.landmarks[0].x.compareTo(b.landmarks[0].x));

      final rawFeatures = _buildFeatureVector(sorted);
      final features    = _applyScaler(rawFeatures);
      final rawProbs    = _model!.predict(features);
      final probs       = _smoothProbs(rawProbs);

      int best = 0; double bestVal = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > bestVal) { bestVal = probs[i]; best = i; }
      }
      final letter = String.fromCharCode('A'.codeUnitAt(0) + best);

      if (bestVal > _confidenceThreshold) {
        _predBuffer.add(letter); _confBuffer.add(bestVal);
        if (_predBuffer.length > _bufferSize) {
          _predBuffer.removeAt(0); _confBuffer.removeAt(0);
        }
      }

      String stable = letter;
      if (_predBuffer.length >= _bufferSize) {
        final w = <String, double>{};
        for (int i = 0; i < _predBuffer.length; i++) {
          w[_predBuffer[i]] = (w[_predBuffer[i]] ?? 0) + _confBuffer[i];
        }
        stable = w.entries.reduce((a, b) => a.value > b.value ? a : b).key;
      }

      final now = DateTime.now().millisecondsSinceEpoch;
      if (stable != _lastAdded &&
          bestVal > _confidenceThreshold &&
          _predBuffer.length >= _bufferSize &&
          now - _lastAddedTime > _cooldownMs) {
        _currentWord  += stable;
        _lastAdded     = stable;
        _lastAddedTime = now;
      }

      if (mounted) {
        if (stable != _letter) _letterAnim.forward(from: 0);
        setState(() {
          _hands = hands; _letter = stable;
          _conf = bestVal; _handFound = true;
        });
      }
    } catch (e) {
      debugPrint('frame: $e');
    } finally {
      _isDetecting = false;
    }
  }

  // ── Canonical orientation-invariant normalisation ──────────────────────────
  List<double>? _normalizeHand(List<Landmark> lms) {
    if (lms.length < 21) return null;
    final pts = List.generate(21, (i) => [
      lms[i].x.toDouble(), lms[i].y.toDouble(), lms[i].z.toDouble(),
    ]);
    final wX = pts[0][0], wY = pts[0][1], wZ = pts[0][2];
    for (int i = 0; i < 21; i++) {
      pts[i][0] -= wX; pts[i][1] -= wY; pts[i][2] -= wZ;
    }
    final mX = pts[9][0], mY = pts[9][1], mZ = pts[9][2];
    final scale = math.sqrt(mX * mX + mY * mY + mZ * mZ);
    if (scale < 1e-6) return null;
    for (int i = 0; i < 21; i++) {
      pts[i][0] /= scale; pts[i][1] /= scale; pts[i][2] /= scale;
    }
    final phi  = math.atan2(pts[9][1], pts[9][0]);
    final theta = -math.pi / 2.0 - phi;
    final cosT = math.cos(theta), sinT = math.sin(theta);
    for (int i = 0; i < 21; i++) {
      final ox = pts[i][0], oy = pts[i][1];
      pts[i][0] = ox * cosT - oy * sinT;
      pts[i][1] = ox * sinT + oy * cosT;
    }
    final out = <double>[];
    for (int i = 0; i < 21; i++) {
      out..add(pts[i][0])..add(pts[i][1])..add(pts[i][2]);
    }
    return out;
  }

  List<double> _buildFeatureVector(List<Hand> sorted) {
    final vec = <double>[];
    for (int h = 0; h < 2; h++) {
      vec.addAll(h < sorted.length
          ? (_normalizeHand(sorted[h].landmarks) ??
             List<double>.filled(63, 0.0))
          : List<double>.filled(63, 0.0));
    }
    return vec;
  }

  List<double> _applyScaler(List<double> raw) {
    final out = List<double>.filled(126, 0.0);
    for (int i = 0; i < 126; i++) {
      out[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];
    }
    return out;
  }

  List<double> _smoothProbs(List<double> probs) {
    if (_emaProbs == null || _emaProbs!.length != probs.length) {
      _emaProbs = List<double>.from(probs);
      return _emaProbs!;
    }
    for (int i = 0; i < probs.length; i++) {
      _emaProbs![i] = 0.4 * probs[i] + 0.6 * _emaProbs![i];
    }
    return _emaProbs!;
  }

  Future<void> _stopCamera() async {
    setState(() => _camState = _CamState.stopping);
    try { await _cam?.stopImageStream(); } catch (_) {}
    _handPlugin?.dispose(); _handPlugin = null;
    await _cam?.dispose(); _cam = null;
    _predBuffer.clear(); _confBuffer.clear(); _emaProbs = null;
    if (!mounted) return;
    setState(() {
      _camState = _CamState.off; _hands = [];
      _letter = ''; _conf = 0.0; _handFound = false; _isDetecting = false;
    });
  }

  void _onToggle() {
    if (_camState == _CamState.off || _camState == _CamState.error) {
      _startCamera();
    } else if (_camState == _CamState.on) {
      _stopCamera();
    }
  }

  void _clearWord() => setState(() {
    _currentWord = ''; _lastAdded = ''; _lastAddedTime = 0;
    _predBuffer.clear(); _confBuffer.clear(); _emaProbs = null;
    _letter = ''; _conf = 0.0;
  });

  void _deleteLastLetter() {
    if (_currentWord.isNotEmpty) {
      setState(() =>
          _currentWord = _currentWord.substring(0, _currentWord.length - 1));
    }
  }

  void _toggleHow() {
    setState(() {
      _howExpanded = !_howExpanded;
      if (_howExpanded) { _activeStep = 0; _howCtrl.forward(); }
      else { _howCtrl..stop()..reset(); }
    });
  }

  void _logout() async {
    await _stopCamera();
    await FirebaseAuth.instance.signOut();
    if (!mounted) return;
    Navigator.pushReplacement(context,
        MaterialPageRoute(builder: (_) => const AuthScreen()));
  }

  @override
  void dispose() {
    _isDetecting = false;
    _cam?.stopImageStream().catchError((_) {});
    _cam?.dispose();
    _handPlugin?.dispose();
    _model?.dispose();
    _pageAnim.dispose(); _pulseAnim.dispose();
    _letterAnim.dispose(); _howCtrl.dispose();
    super.dispose();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  BUILD
  // ═══════════════════════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    final email = FirebaseAuth.instance.currentUser?.email ?? 'User';
    return Scaffold(
      backgroundColor: _bg,
      body: Stack(children: [
        _bg2(),
        SafeArea(child: FadeTransition(opacity: _fadeIn,
          child: SlideTransition(position: _slideIn,
            child: Column(children: [
              _topBar(email),
              Expanded(child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const SizedBox(height: 8),
                    _greeting(email),
                    const SizedBox(height: 16),
                    _modelBadge(),
                    const SizedBox(height: 16),
                    _howCard(),
                    const SizedBox(height: 16),
                    _statsRow(),
                    const SizedBox(height: 16),
                    _cameraCard(),
                    const SizedBox(height: 16),
                    _wordCard(),
                    const SizedBox(height: 16),
                    if (_camState == _CamState.on) _predCard(),
                    if (_camState == _CamState.on) const SizedBox(height: 16),
                    _tipsCard(),
                    const SizedBox(height: 32),
                  ],
                ),
              )),
            ]),
          ),
        )),
      ]),
    );
  }

  Widget _bg2() => Stack(children: [
    Positioned(top: -80, right: -50, child: _blob(240, _primary, 0.10)),
    Positioned(bottom: -60, left: -40, child: _blob(200, _purple, 0.08)),
  ]);
  Widget _blob(double s, Color c, double o) => Container(
      width: s, height: s,
      decoration: BoxDecoration(shape: BoxShape.circle, color: c.withOpacity(o)));

  Widget _topBar(String email) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
    child: Row(children: [
      Container(width: 40, height: 40,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
              colors: [_primary, _purple],
              begin: Alignment.topLeft, end: Alignment.bottomRight),
          borderRadius: BorderRadius.circular(12),
          boxShadow: [BoxShadow(
              color: _primary.withOpacity(0.35),
              blurRadius: 12, offset: const Offset(0, 4))]),
        child: const Icon(Icons.sign_language_rounded, size: 22, color: Colors.white)),
      const SizedBox(width: 10),
      const Text('SignBridge', style: TextStyle(
          fontSize: 20, fontWeight: FontWeight.w800, color: _dark, letterSpacing: -.5)),
      const Spacer(),
      TextButton.icon(
        onPressed: _logout,
        icon: const Icon(Icons.logout_rounded, size: 16, color: _red),
        label: const Text('Logout', style: TextStyle(
            fontSize: 13, fontWeight: FontWeight.w600, color: _red)),
        style: TextButton.styleFrom(
          backgroundColor: _red.withOpacity(0.08),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8)),
      ),
    ]),
  );

  Widget _greeting(String email) {
    final name = email.split('@').first;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
            colors: [_primary, _purple],
            begin: Alignment.topLeft, end: Alignment.bottomRight),
        borderRadius: BorderRadius.circular(22),
        boxShadow: [BoxShadow(
            color: _primary.withOpacity(0.35),
            blurRadius: 24, offset: const Offset(0, 10))]),
      child: Row(children: [
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Welcome back,', style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75))),
          Text(name, style: const TextStyle(
              fontSize: 24, fontWeight: FontWeight.w800, color: Colors.white, letterSpacing: -.5)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.18),
                borderRadius: BorderRadius.circular(20)),
            child: const Text('ISL Real-Time Alphabet Translator ✦',
                style: TextStyle(fontSize: 11, color: Colors.white, fontWeight: FontWeight.w600))),
        ])),
        const SizedBox(width: 12),
        AnimatedBuilder(animation: _pulse,
          builder: (_, child) => Transform.scale(scale: _pulse.value, child: child),
          child: Container(width: 58, height: 58,
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.18), shape: BoxShape.circle),
            child: const Icon(Icons.waving_hand_rounded, color: Colors.white, size: 28))),
      ]),
    );
  }

  Widget _modelBadge() {
    if (_modelLoading) return _chip(Icons.hourglass_top_rounded, 'Loading isl_twohand_mlp.tflite…', _amber);
    if (_modelError.isNotEmpty) return _chip(Icons.error_outline_rounded, _modelError, _red);
    return _chip(Icons.check_circle_rounded, 'isl_twohand_mlp.tflite ready ✓', _green);
  }

  Widget _chip(IconData icon, String msg, Color c) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
    decoration: BoxDecoration(
        color: c.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: c.withOpacity(0.30))),
    child: Row(children: [
      _modelLoading
          ? SizedBox(width: 16, height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: c))
          : Icon(icon, size: 16, color: c),
      const SizedBox(width: 10),
      Expanded(child: Text(msg, style: TextStyle(
          fontSize: 12.5, color: c, fontWeight: FontWeight.w600))),
    ]),
  );

  Widget _howCard() => _Card(child: Column(children: [
    GestureDetector(
      onTap: _toggleHow,
      behavior: HitTestBehavior.opaque,
      child: Row(children: [
        Container(width: 38, height: 38,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [_primary, _purple],
                begin: Alignment.topLeft, end: Alignment.bottomRight),
            borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.play_lesson_rounded, size: 18, color: Colors.white)),
        const SizedBox(width: 10),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('How SignBridge Works', style: TextStyle(
              fontSize: 15, fontWeight: FontWeight.w700, color: _dark)),
          Text(_howExpanded ? 'Tap to collapse' : 'Tap to see steps',
              style: const TextStyle(fontSize: 11, color: _grey)),
        ])),
        AnimatedRotation(
          turns: _howExpanded ? 0.5 : 0,
          duration: const Duration(milliseconds: 300),
          child: const Icon(Icons.keyboard_arrow_down_rounded, color: _grey)),
      ]),
    ),
    AnimatedCrossFade(
      duration: const Duration(milliseconds: 400),
      crossFadeState: _howExpanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
      firstChild: const SizedBox.shrink(),
      secondChild: _howBody(),
    ),
  ]));

  Widget _howBody() {
    final s = _steps[_activeStep];
    return Padding(padding: const EdgeInsets.only(top: 16),
      child: Column(children: [
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 400),
          child: _StepIll(color: s.color, icon: s.art, key: ValueKey(_activeStep))),
        const SizedBox(height: 14),
        Row(children: List.generate(_steps.length, (i) {
          final active = i == _activeStep;
          final st = _steps[i];
          return Expanded(child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 3),
            child: GestureDetector(
              onTap: () {
                if (!_howExpanded) return;
                _howCtrl.reset();
                setState(() => _activeStep = i);
                _howCtrl.forward();
              },
              child: Column(children: [
                ClipRRect(borderRadius: BorderRadius.circular(4),
                  child: SizedBox(height: 4,
                    child: active
                        ? AnimatedBuilder(animation: _howCtrl,
                            builder: (_, __) => LinearProgressIndicator(
                              value: _howCtrl.value,
                              backgroundColor: st.color.withOpacity(0.15),
                              valueColor: AlwaysStoppedAnimation(st.color)))
                        : Container(color: i < _activeStep
                            ? _steps[i].color.withOpacity(0.40)
                            : const Color(0xFFEEF0F6)))),
                const SizedBox(height: 8),
                AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: active ? st.color.withOpacity(0.12) : Colors.transparent,
                    borderRadius: BorderRadius.circular(8)),
                  child: Icon(st.icon, size: 18,
                      color: active ? st.color : _grey.withOpacity(0.50))),
              ]),
            ),
          ));
        })),
        const SizedBox(height: 12),
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 350),
          child: Column(key: ValueKey(_activeStep),
            crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                    color: s.color.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(6)),
                child: Text('Step ${_activeStep + 1} of ${_steps.length}',
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                        color: s.color, letterSpacing: .3))),
              const SizedBox(width: 8),
              Text(s.title, style: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.w700, color: _dark)),
            ]),
            const SizedBox(height: 6),
            Text(s.desc, style: const TextStyle(fontSize: 12.5, color: _grey, height: 1.55)),
          ])),
      ]));
  }

  Widget _statsRow() => Row(children: [
    Expanded(child: _Stat(icon: Icons.back_hand_rounded,
        val: '26+', lbl: 'Signs\nSupported', color: _primary)),
    const SizedBox(width: 12),
    Expanded(child: _Stat(icon: Icons.bar_chart_rounded,
        val: '94%', lbl: 'Accuracy\nRate', color: _purple)),
    const SizedBox(width: 12),
    Expanded(child: _Stat(icon: Icons.speed_rounded,
        val: '0.3s', lbl: 'Avg.\nLatency', color: _cyan)),
  ]);

  // ── Camera card ────────────────────────────────────────────────────────────
  Widget _cameraCard() => _Card(child: Column(
    crossAxisAlignment: CrossAxisAlignment.stretch, children: [
    Row(children: [
      const Icon(Icons.videocam_rounded, size: 20, color: _primary),
      const SizedBox(width: 8),
      const Text('Live Camera', style: TextStyle(
          fontSize: 16, fontWeight: FontWeight.w700, color: _dark)),
      const Spacer(),
      if (_camState == _CamState.on) _liveDot(),
    ]),
    const SizedBox(height: 6),
    Text(_camState == _CamState.on
        ? 'Back camera active — green skeleton = hand detected.'
        : 'Tap "Start Camera" — point back camera at your hands.',
        style: const TextStyle(fontSize: 12, color: _grey, height: 1.5)),
    const SizedBox(height: 8),

    // ── FIX: Rotation selector — now calls _changeRotation() which
    //    restarts the image stream so the new angle actually takes effect ─────
    Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF4F5FB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE0E3F0))),
      child: Row(children: [
        const Icon(Icons.rotate_right_rounded, size: 16, color: _primary),
        const SizedBox(width: 8),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Skeleton display rotation',
              style: TextStyle(fontSize: 12, color: _grey, fontWeight: FontWeight.w600)),
          Text('Tap ◀ ▶ until skeleton aligns with your hand',
              style: TextStyle(fontSize: 10, color: _grey.withOpacity(0.70))),
        ])),
        const SizedBox(width: 8),
        // ◀ button — now uses _changeRotation(-1)
        GestureDetector(
          onTap: () => _changeRotation(-1),
          child: Container(width: 32, height: 32,
            decoration: BoxDecoration(
                color: _primary.withOpacity(0.10),
                borderRadius: BorderRadius.circular(8)),
            child: const Icon(Icons.chevron_left_rounded, size: 22, color: _primary))),
        const SizedBox(width: 6),
        Container(width: 52, alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(vertical: 6),
          decoration: BoxDecoration(color: _primary, borderRadius: BorderRadius.circular(8)),
          child: Text('$_rot°', style: const TextStyle(
              fontSize: 14, fontWeight: FontWeight.w800, color: Colors.white))),
        const SizedBox(width: 6),
        // ▶ button — now uses _changeRotation(+1)
        GestureDetector(
          onTap: () => _changeRotation(1),
          child: Container(width: 32, height: 32,
            decoration: BoxDecoration(
                color: _primary.withOpacity(0.10),
                borderRadius: BorderRadius.circular(8)),
            child: const Icon(Icons.chevron_right_rounded, size: 22, color: _primary))),
      ]),
    ),
    const SizedBox(height: 14),

    ClipRRect(borderRadius: BorderRadius.circular(16),
        child: SizedBox(height: 300, child: _viewport())),
    const SizedBox(height: 14),
    _toggleBtn(),
    if (_camState == _CamState.error) ...[
      const SizedBox(height: 10),
      Container(padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
            color: _red.withOpacity(0.07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: _red.withOpacity(0.20))),
        child: Text(_camError, style: TextStyle(fontSize: 12, color: _red.withOpacity(0.85)))),
    ],
  ]));

  Widget _liveDot() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(color: _red.withOpacity(0.12), borderRadius: BorderRadius.circular(20)),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 6, height: 6,
          decoration: const BoxDecoration(shape: BoxShape.circle, color: _red)),
      const SizedBox(width: 5),
      const Text('LIVE', style: TextStyle(
          fontSize: 9, fontWeight: FontWeight.w800, color: _red, letterSpacing: .8)),
    ]),
  );

  // ── Camera viewport ────────────────────────────────────────────────────────
  Widget _viewport() {
    const dark = Color(0xFF0D0D1A);
    if (_camState == _CamState.off) {
      return Container(color: dark, child: Column(
        mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.videocam_off_rounded, size: 52, color: Colors.white.withOpacity(0.18)),
          const SizedBox(height: 14),
          Text('Camera is off', style: TextStyle(
              fontSize: 14, color: Colors.white.withOpacity(0.35), fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          Text('Tap "Start Camera" below',
              style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.20))),
        ]));
    }
    if (_camState == _CamState.error) {
      return Container(color: dark, child: Column(
        mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.error_outline_rounded, size: 44, color: _red.withOpacity(0.70)),
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text(_camError, textAlign: TextAlign.center,
                style: TextStyle(fontSize: 12, color: _red.withOpacity(0.80)))),
        ]));
    }
    if (_cam == null || !_cam!.value.isInitialized) {
      return Container(color: dark, child: Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center, children: [
          SizedBox(width: 36, height: 36,
              child: CircularProgressIndicator(strokeWidth: 2.5, color: _primary.withOpacity(0.70))),
          const SizedBox(height: 14),
          Text(_camState == _CamState.stopping ? 'Stopping…' : 'Starting camera…',
              style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.45))),
        ])));
    }

    return Stack(fit: StackFit.expand, children: [
      FittedBox(fit: BoxFit.cover,
        child: SizedBox(
          width:  _cam!.value.previewSize!.height,
          height: _cam!.value.previewSize!.width,
          child:  CameraPreview(_cam!))),

      // FIX: Pass _rot to _SkeletonPainter so it applies the same coordinate
      // transform that hand_landmarker uses internally, making the overlay
      // dots/lines actually match the rotated detection output on screen.
      if (_hands.isNotEmpty)
        Positioned.fill(
            child: CustomPaint(painter: _SkeletonPainter(_hands, _rot))),

      Positioned(top:10,   left:10,  child: _Cor(c: _primary.withOpacity(.80))),
      Positioned(top:10,   right:10, child: _Cor(c: _primary.withOpacity(.80), fH: true)),
      Positioned(bottom:10,left:10,  child: _Cor(c: _primary.withOpacity(.80), fV: true)),
      Positioned(bottom:10,right:10, child: _Cor(c: _primary.withOpacity(.80), fH: true, fV: true)),

      Positioned(bottom: 10, left: 0, right: 0,
        child: Center(child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
          decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.58),
              borderRadius: BorderRadius.circular(20)),
          child: Text(
            _handFound
                ? '✋ ${_hands.length == 2 ? "Both hands" : "1 hand"} detected'
                : '👋 Point back camera at your hands',
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600,
                color: _handFound ? _green.withOpacity(0.90) : Colors.white.withOpacity(0.75)))))),

      if (_letter.isNotEmpty)
        Positioned(top: 10, right: 10,
          child: ScaleTransition(scale: _letterScale,
            child: Container(width: 56, height: 56,
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.65),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: _handFound ? _green.withOpacity(0.85) : _amber.withOpacity(0.60),
                    width: 2.5)),
              child: Center(child: Text(_letter, style: TextStyle(
                  fontSize: 30, fontWeight: FontWeight.w900,
                  color: _handFound ? _green : _amber)))))),
    ]);
  }

  Widget _toggleBtn() {
    final busy = _camState == _CamState.starting || _camState == _CamState.stopping;
    String lbl; IconData ico; Color col;
    switch (_camState) {
      case _CamState.starting: lbl='Starting…';   ico=Icons.hourglass_top_rounded; col=_primary; break;
      case _CamState.stopping: lbl='Stopping…';   ico=Icons.hourglass_top_rounded; col=_grey;    break;
      case _CamState.on:       lbl='Stop Camera'; ico=Icons.stop_rounded;          col=_red;     break;
      case _CamState.error:    lbl='Retry';       ico=Icons.refresh_rounded;       col=_amber;   break;
      default:                 lbl='Start Camera';ico=Icons.play_arrow_rounded;    col=_primary; break;
    }
    return SizedBox(height: 52, child: ElevatedButton.icon(
      onPressed: (_modelLoading || busy) ? null : _onToggle,
      icon: busy
          ? const SizedBox(width: 18, height: 18,
              child: CircularProgressIndicator(strokeWidth: 2.2, color: Colors.white))
          : Icon(ico, size: 20),
      label: Text(lbl, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
      style: ElevatedButton.styleFrom(
        backgroundColor: col,
        disabledBackgroundColor: _primary.withOpacity(0.35),
        foregroundColor: Colors.white, elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
    ));
  }

  Widget _wordCard() => _Card(child: Column(
    crossAxisAlignment: CrossAxisAlignment.stretch, children: [
    Row(children: [
      const Icon(Icons.text_fields_rounded, size: 20, color: _primary),
      const SizedBox(width: 8),
      const Text('Word Builder', style: TextStyle(
          fontSize: 16, fontWeight: FontWeight.w700, color: _dark)),
      const Spacer(),
      if (_currentWord.isNotEmpty) ...[
        GestureDetector(
          onTap: _deleteLastLetter,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
                color: _amber.withOpacity(0.10),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: _amber.withOpacity(0.30))),
            child: const Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.backspace_outlined, size: 14, color: _amber),
              SizedBox(width: 4),
              Text('Del', style: TextStyle(fontSize: 12, color: _amber, fontWeight: FontWeight.w600)),
            ]))),
        const SizedBox(width: 8),
      ],
      GestureDetector(
        onTap: _clearWord,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
              color: _red.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _red.withOpacity(0.25))),
          child: const Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.clear_rounded, size: 14, color: _red),
            SizedBox(width: 4),
            Text('Clear', style: TextStyle(fontSize: 12, color: _red, fontWeight: FontWeight.w600)),
          ]))),
    ]),
    const SizedBox(height: 12),
    Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: _currentWord.isEmpty ? const Color(0xFFF4F5FB) : _primary.withOpacity(0.04),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: _currentWord.isEmpty ? const Color(0xFFE8EAF4) : _primary.withOpacity(0.20))),
      child: _currentWord.isEmpty
          ? Row(children: [
              Icon(Icons.gesture_rounded, size: 16, color: _grey.withOpacity(0.50)),
              const SizedBox(width: 8),
              Text('Make signs to build words…',
                  style: TextStyle(fontSize: 14, color: _grey.withOpacity(0.60), fontStyle: FontStyle.italic)),
            ])
          : Text(_currentWord, style: const TextStyle(
              fontSize: 22, fontWeight: FontWeight.w700, color: _dark, letterSpacing: 1.2)),
    ),
    const SizedBox(height: 8),
    Text(
      _currentWord.isEmpty ? ' '
          : '${_currentWord.length} letter${_currentWord.length == 1 ? "" : "s"} — hold sign for 1.2s to add next',
      style: TextStyle(fontSize: 11, color: _grey.withOpacity(0.70))),
  ]));

  Widget _predCard() => _Card(child: Column(
    crossAxisAlignment: CrossAxisAlignment.stretch, children: [
    const Row(children: [
      Icon(Icons.auto_awesome_rounded, size: 18, color: _primary),
      SizedBox(width: 8),
      Text('Real-Time Prediction', style: TextStyle(
          fontSize: 15, fontWeight: FontWeight.w700, color: _dark)),
    ]),
    const SizedBox(height: 14),
    if (_letter.isEmpty)
      Container(
        padding: const EdgeInsets.symmetric(vertical: 18),
        decoration: BoxDecoration(color: _grey.withOpacity(0.07), borderRadius: BorderRadius.circular(12)),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          SizedBox(width: 18, height: 18,
              child: CircularProgressIndicator(strokeWidth: 2, color: _grey)),
          const SizedBox(width: 12),
          Text('Point camera at your hands…', style: TextStyle(fontSize: 13, color: _grey)),
        ]))
    else
      _result(),
  ]));

  Widget _result() {
    final col = _handFound ? _green : _amber;
    final pct = (_conf * 100).toStringAsFixed(1);
    return Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
      ScaleTransition(scale: _letterScale,
        child: Container(width: 84, height: 84,
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: [col.withOpacity(0.18), col.withOpacity(0.06)]),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: col.withOpacity(0.50), width: 2)),
          child: Center(child: Text(_letter, style: TextStyle(
              fontSize: 44, fontWeight: FontWeight.w900, color: col))))),
      const SizedBox(width: 16),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Detected ISL Letter',
            style: TextStyle(fontSize: 11, color: _grey, fontWeight: FontWeight.w600)),
        Text(_letter, style: const TextStyle(
            fontSize: 28, fontWeight: FontWeight.w900, color: _dark, letterSpacing: -.5)),
        const SizedBox(height: 10),
        ClipRRect(borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(value: _conf, minHeight: 8,
              backgroundColor: col.withOpacity(0.12),
              valueColor: AlwaysStoppedAnimation<Color>(col))),
        const SizedBox(height: 6),
        Row(children: [
          Text('$pct%', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: col)),
          const SizedBox(width: 10),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(color: col.withOpacity(0.10), borderRadius: BorderRadius.circular(20)),
            child: Text(
              _handFound ? '✋ ${_hands.length == 2 ? "2 hands" : "1 hand"}' : '🤚 No hand',
              style: TextStyle(fontSize: 10.5, color: col, fontWeight: FontWeight.w600))),
        ]),
      ])),
    ]);
  }

  Widget _tipsCard() {
    const tips = [
      (Icons.wb_sunny_rounded,              'Good lighting helps MediaPipe detect hands'),
      (Icons.pan_tool_rounded,              'Keep both hands fully visible in frame'),
      (Icons.rotate_90_degrees_ccw_rounded, 'Works in portrait, landscape, or any tilt ✓'),
      (Icons.crop_free_rounded,             'Plain background gives best accuracy'),
    ];
    return _Card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Row(children: [
        Icon(Icons.lightbulb_outline_rounded, size: 18, color: _primary),
        SizedBox(width: 8),
        Text('Tips for best results', style: TextStyle(
            fontSize: 15, fontWeight: FontWeight.w700, color: _dark)),
      ]),
      const SizedBox(height: 14),
      ...tips.map((t) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Row(children: [
          Container(width: 30, height: 30,
            decoration: BoxDecoration(color: _primary.withOpacity(0.08), borderRadius: BorderRadius.circular(8)),
            child: Icon(t.$1, size: 15, color: _primary)),
          const SizedBox(width: 10),
          Expanded(child: Text(t.$2, style: const TextStyle(fontSize: 13, color: _grey, height: 1.4))),
        ]))),
    ]));
  }
}

// ─── Hand Skeleton Painter — FIX: accepts rotation and transforms coords ──────
class _SkeletonPainter extends CustomPainter {
  final List<Hand> hands;
  final int rotation; // 0, 90, 180, or 270 — must match what detect() used

  const _SkeletonPainter(this.hands, this.rotation);

  static const _conn = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  /// FIX: hand_landmarker returns landmark coords normalised to the INPUT IMAGE
  /// frame, which is rotated relative to the screen display.
  /// We must apply the inverse rotation to map them back to screen space.
  ///
  /// For each rotation value, the transform is:
  ///   0°   → (x, y) — no change needed
  ///   90°  → (1−y, x)  — most common for Android portrait + back camera
  ///   180° → (1−x, 1−y)
  ///   270° → (y, 1−x)
  Offset _toScreen(double x, double y, Size size) {
    double sx, sy;
    switch (rotation) {
      case 90:
        sx = (1.0 - y) * size.width;
        sy = x * size.height;
        break;
      case 180:
        sx = (1.0 - x) * size.width;
        sy = (1.0 - y) * size.height;
        break;
      case 270:
        sx = y * size.width;
        sy = (1.0 - x) * size.height;
        break;
      default: // 0°
        sx = x * size.width;
        sy = y * size.height;
        break;
    }
    return Offset(sx, sy);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final lp = Paint()
      ..color = const Color(0xFF50C850)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    final dp = Paint()
      ..color = const Color(0xFFFF3030)
      ..style = PaintingStyle.fill;

    for (final hand in hands) {
      final lms = hand.landmarks;
      if (lms.length < 21) continue;

      // Apply rotation-aware coord transform for each landmark
      final pts = lms.map((l) => _toScreen(l.x, l.y, size)).toList();

      for (final c in _conn) {
        canvas.drawLine(pts[c[0]], pts[c[1]], lp);
      }
      for (final p in pts) {
        canvas.drawCircle(p, 5, dp);
      }
    }
  }

  @override
  bool shouldRepaint(_SkeletonPainter o) =>
      o.hands != hands || o.rotation != rotation;
}

// ─── Step illustration ────────────────────────────────────────────────────────
class _StepIll extends StatefulWidget {
  final Color color; final IconData icon;
  const _StepIll({required this.color, required this.icon, super.key});
  @override State<_StepIll> createState() => _StepIllState();
}
class _StepIllState extends State<_StepIll> with SingleTickerProviderStateMixin {
  late AnimationController _c;
  late Animation<double> _f;
  @override void initState() {
    super.initState();
    _c = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat(reverse: true);
    _f = Tween<double>(begin: -6, end: 6)
        .animate(CurvedAnimation(parent: _c, curve: Curves.easeInOut));
  }
  @override void dispose() { _c.dispose(); super.dispose(); }
  @override
  Widget build(BuildContext ctx) => Container(
    height: 120,
    decoration: BoxDecoration(
      gradient: LinearGradient(colors: [
          widget.color.withOpacity(0.10), widget.color.withOpacity(0.03)],
          begin: Alignment.topLeft, end: Alignment.bottomRight),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: widget.color.withOpacity(0.15))),
    child: Stack(alignment: Alignment.center, children: [
      Container(width: 80, height: 80, decoration: BoxDecoration(
          shape: BoxShape.circle, color: widget.color.withOpacity(0.06))),
      Container(width: 56, height: 56, decoration: BoxDecoration(
          shape: BoxShape.circle, color: widget.color.withOpacity(0.10))),
      AnimatedBuilder(animation: _f,
        builder: (_, child) => Transform.translate(offset: Offset(0, _f.value), child: child),
        child: Container(width: 52, height: 52,
          decoration: BoxDecoration(
            shape: BoxShape.circle, color: widget.color,
            boxShadow: [BoxShadow(
                color: widget.color.withOpacity(0.40),
                blurRadius: 14, offset: const Offset(0, 5))]),
          child: Icon(widget.icon, size: 26, color: Colors.white))),
    ]),
  );
}

// ─── Stat chip ────────────────────────────────────────────────────────────────
class _Stat extends StatelessWidget {
  final IconData icon; final String val, lbl; final Color color;
  const _Stat({required this.icon, required this.val, required this.lbl, required this.color});
  @override
  Widget build(BuildContext ctx) => Container(
    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      boxShadow: [BoxShadow(
          color: color.withOpacity(0.08), blurRadius: 18, offset: const Offset(0, 6))],
      border: Border.all(color: const Color(0xFFEEF0F6))),
    child: Column(children: [
      Container(width: 36, height: 36,
        decoration: BoxDecoration(color: color.withOpacity(0.10), borderRadius: BorderRadius.circular(10)),
        child: Icon(icon, size: 18, color: color)),
      const SizedBox(height: 8),
      Text(val, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800,
          color: Color(0xFF1A1D2E), letterSpacing: -.5)),
      const SizedBox(height: 3),
      Text(lbl, textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 10, color: Color(0xFF8890AD), height: 1.3)),
    ]),
  );
}

// ─── Card shell ───────────────────────────────────────────────────────────────
class _Card extends StatelessWidget {
  final Widget child;
  const _Card({required this.child});
  @override
  Widget build(BuildContext ctx) => Container(
    padding: const EdgeInsets.all(18),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(20),
      boxShadow: [
        BoxShadow(color: const Color(0xFF4F6EF7).withOpacity(0.07),
            blurRadius: 24, offset: const Offset(0, 8)),
        BoxShadow(color: Colors.black.withOpacity(0.03),
            blurRadius: 4, offset: const Offset(0, 2)),
      ],
      border: Border.all(color: const Color(0xFFEEF0F6))),
    child: child,
  );
}

// ─── Corner HUD ───────────────────────────────────────────────────────────────
class _Cor extends StatelessWidget {
  final Color c; final bool fH, fV;
  const _Cor({required this.c, this.fH = false, this.fV = false});
  @override
  Widget build(BuildContext ctx) => Transform.scale(
    scaleX: fH ? -1 : 1, scaleY: fV ? -1 : 1,
    child: CustomPaint(size: const Size(22, 22), painter: _CorP(c)));
}
class _CorP extends CustomPainter {
  final Color c; const _CorP(this.c);
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()..color = c..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round..style = PaintingStyle.stroke;
    canvas.drawLine(Offset.zero, Offset(size.width, 0), p);
    canvas.drawLine(Offset.zero, Offset(0, size.height), p);
  }
  @override bool shouldRepaint(_CorP o) => o.c != c;
}