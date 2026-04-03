import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'auth_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait orientation
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Transparent status bar to blend with splash
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(const MyApp());
}

// ─────────────────────────────── App Root ─────────────────────────────────── //

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignBridge',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4F6EF7),
          brightness: Brightness.light,
        ),
        fontFamily: 'SF Pro Display',
        scaffoldBackgroundColor: const Color(0xFFF8F9FF),
        useMaterial3: true,
      ),
      home: const SplashScreen(),
    );
  }
}

// ──────────────────────────── Splash Screen ───────────────────────────────── //

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  // ─── Design tokens (matches auth & home screens) ──────────────────────────
  static const _primary   = Color(0xFF4F6EF7);
  static const _secondary = Color(0xFF7C3AED);
  static const _surface   = Color(0xFFF8F9FF);
  static const _onSurface = Color(0xFF1A1D2E);
  static const _subtle    = Color(0xFF8890AD);

  // ─── Animation controllers ────────────────────────────────────────────────
  late AnimationController _logoController;
  late AnimationController _textController;
  late AnimationController _taglineController;
  late AnimationController _dotsController;
  late AnimationController _exitController;

  late Animation<double>  _logoScale;
  late Animation<double>  _logoOpacity;
  late Animation<double>  _textOpacity;
  late Animation<Offset>  _textSlide;
  late Animation<double>  _taglineOpacity;
  late Animation<double>  _exitOpacity;

  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _startSequence();
  }

  void _setupAnimations() {
    // Logo pop-in
    _logoController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _logoScale = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.elasticOut),
    );
    _logoOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _logoController, curve: const Interval(0, 0.4)),
    );

    // Title slide-up
    _textController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _textOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _textController, curve: Curves.easeOut),
    );
    _textSlide = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _textController, curve: Curves.easeOut));

    // Tagline fade
    _taglineController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _taglineOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _taglineController, curve: Curves.easeIn),
    );

    // Loading dots
    _dotsController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat();

    // Exit fade-out
    _exitController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _exitOpacity = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _exitController, curve: Curves.easeIn),
    );
  }

  Future<void> _startSequence() async {
    await Future.delayed(const Duration(milliseconds: 200));
    _logoController.forward();

    await Future.delayed(const Duration(milliseconds: 500));
    _textController.forward();

    await Future.delayed(const Duration(milliseconds: 300));
    _taglineController.forward();

    // Hold on splash for ~2.5 s total, then exit
    await Future.delayed(const Duration(milliseconds: 1800));
    _dotsController.stop();
    await _exitController.forward();

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      PageRouteBuilder(
        pageBuilder: (_, __, ___) => const AuthScreen(),
        transitionsBuilder: (_, animation, __, child) => FadeTransition(
          opacity: animation,
          child: child,
        ),
        transitionDuration: const Duration(milliseconds: 500),
      ),
    );
  }

  @override
  void dispose() {
    _logoController.dispose();
    _textController.dispose();
    _taglineController.dispose();
    _dotsController.dispose();
    _exitController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: _surface,
      body: FadeTransition(
        opacity: _exitOpacity,
        child: Stack(
          children: [
            // ── Background decoration ──────────────────────────────────────
            _SplashBackground(size: size),

            // ── Main content ───────────────────────────────────────────────
            SafeArea(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Spacer(flex: 2),

                    // ── Animated logo mark ─────────────────────────────────
                    ScaleTransition(
                      scale: _logoScale,
                      child: FadeTransition(
                        opacity: _logoOpacity,
                        child: _LogoMark(),
                      ),
                    ),

                    const SizedBox(height: 28),

                    // ── App name ───────────────────────────────────────────
                    SlideTransition(
                      position: _textSlide,
                      child: FadeTransition(
                        opacity: _textOpacity,
                        child: const Text(
                          'SignBridge',
                          style: TextStyle(
                            fontSize: 38,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1.0,
                            color: _onSurface,
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 8),

                    // ── Tagline ────────────────────────────────────────────
                    FadeTransition(
                      opacity: _taglineOpacity,
                      child: const Text(
                        'Connecting signs to text seamlessly',
                        style: TextStyle(
                          fontSize: 14,
                          color: _subtle,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ),

                    const Spacer(flex: 2),

                    // ── Animated loading dots ──────────────────────────────
                    FadeTransition(
                      opacity: _taglineOpacity,
                      child: _LoadingDots(controller: _dotsController),
                    ),

                    const SizedBox(height: 48),

                    // ── Version tag ────────────────────────────────────────
                    FadeTransition(
                      opacity: _taglineOpacity,
                      child: Text(
                        'v1.0.0',
                        style: TextStyle(
                          fontSize: 11,
                          color: _subtle.withOpacity(0.5),
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),

                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ────────────────────────── Logo Mark Widget ──────────────────────────────── //

class _LogoMark extends StatelessWidget {
  static const _primary   = Color(0xFF4F6EF7);
  static const _secondary = Color(0xFF7C3AED);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 100,
      height: 100,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [_primary, _secondary],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: _primary.withOpacity(0.45),
            blurRadius: 36,
            offset: const Offset(0, 14),
          ),
          BoxShadow(
            color: _secondary.withOpacity(0.20),
            blurRadius: 60,
            spreadRadius: 4,
            offset: const Offset(0, 20),
          ),
        ],
      ),
      child: const Icon(
        Icons.sign_language_rounded,
        size: 52,
        color: Colors.white,
      ),
    );
  }
}

// ─────────────────────────── Loading Dots ─────────────────────────────────── //

class _LoadingDots extends StatelessWidget {
  final AnimationController controller;
  const _LoadingDots({required this.controller});

  static const _primary = Color(0xFF4F6EF7);

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (_, __) {
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(3, (i) {
            // Each dot animates with a staggered phase
            final phase  = (controller.value - i * 0.25).clamp(0.0, 1.0);
            final bounce = (phase < 0.5 ? phase : 1.0 - phase) * 2;
            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 4),
              width: 7,
              height: 7 + bounce * 5,
              decoration: BoxDecoration(
                color: _primary.withOpacity(0.3 + bounce * 0.7),
                borderRadius: BorderRadius.circular(4),
              ),
            );
          }),
        );
      },
    );
  }
}

// ──────────────────────── Splash Background ───────────────────────────────── //

class _SplashBackground extends StatelessWidget {
  final Size size;
  const _SplashBackground({required this.size});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned(
          top: -110,
          left: -70,
          child: Container(
            width: 300,
            height: 300,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFF4F6EF7).withOpacity(0.10),
            ),
          ),
        ),
        Positioned(
          bottom: -90,
          right: -60,
          child: Container(
            width: 260,
            height: 260,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFF7C3AED).withOpacity(0.09),
            ),
          ),
        ),
        Positioned.fill(
          child: CustomPaint(painter: _GridPainter()),
        ),
      ],
    );
  }
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF4F6EF7).withOpacity(0.04)
      ..strokeWidth = 1;
    const step = 40.0;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(_GridPainter _) => false;
}