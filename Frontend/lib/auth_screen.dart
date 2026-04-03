import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'home_screen.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen>
    with SingleTickerProviderStateMixin {
  // ─── Core state ───────────────────────────────────────────────────────────
  final _auth = FirebaseAuth.instance;
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool isLogin = true;

  // ─── Animation ────────────────────────────────────────────────────────────
  late AnimationController _controller;
  late Animation<double> _fadeIn;

  // ─── UI state ─────────────────────────────────────────────────────────────
  bool _isLoading = false;
  bool _obscurePassword = true;
  final _formKey = GlobalKey<FormState>();

  // ─── Design tokens ────────────────────────────────────────────────────────
  static const _primary   = Color(0xFF4F6EF7);
  static const _secondary = Color(0xFF7C3AED);
  static const _surface   = Color(0xFFF8F9FF);
  static const _onSurface = Color(0xFF1A1D2E);
  static const _subtle    = Color(0xFF8890AD);
  static const _error     = Color(0xFFE53E6D);

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _fadeIn = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  // ─── Validation ───────────────────────────────────────────────────────────
  bool validatePassword(String password) {
    final regex = RegExp(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{6,}$');
    return regex.hasMatch(password);
  }

  // ─── Snackbar helper ──────────────────────────────────────────────────────
  void _showSnack(String message, {bool isError = true}) {
    ScaffoldMessenger.of(context).clearSnackBars();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(
              isError
                  ? Icons.error_outline_rounded
                  : Icons.check_circle_outline_rounded,
              color: Colors.white,
              size: 18,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                message,
                style: const TextStyle(
                  fontFamily: 'SF Pro Display',
                  fontSize: 13.5,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ),
        backgroundColor: isError ? _error : const Color(0xFF22C55E),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        duration: const Duration(seconds: 3),
      ),
    );
  }

  // ─── Submit ───────────────────────────────────────────────────────────────
  Future<void> _submit() async {
    final email    = _emailController.text.trim();
    final password = _passwordController.text.trim();

    if (!validatePassword(password)) {
      _showSnack(
          'Password must have upper, lower, number, min 6 chars, no special chars');
      return;
    }

    setState(() => _isLoading = true);

    try {
      if (isLogin) {
        final userCredential = await _auth.signInWithEmailAndPassword(
          email: email,
          password: password,
        );

        if (userCredential.user != null &&
            userCredential.user!.emailVerified) {
          if (!mounted) return;
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => const HomeScreen()),
          );
        } else {
          _showSnack('Please verify your email before logging in.');
        }
      } else {
        final userCredential = await _auth.createUserWithEmailAndPassword(
          email: email,
          password: password,
        );
        await userCredential.user?.sendEmailVerification();
        _showSnack(
          'Account created! Check your inbox for a verification link.',
          isError: false,
        );
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        _showSnack('User ID not found. Please register first.');
      } else if (e.code == 'wrong-password') {
        _showSnack('Incorrect password.');
      } else if (e.code == 'email-already-in-use') {
        _showSnack('This email is already registered.');
      } else {
        _showSnack('Error: ${e.message}');
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  // ─── Resend verification ──────────────────────────────────────────────────
  Future<void> _resendVerification() async {
    try {
      await _auth.currentUser?.sendEmailVerification();
      _showSnack('Verification email resent!', isError: false);
    } catch (e) {
      _showSnack('Error: $e');
    }
  }

  // ─── ✅ NEW: Forgot Password dialog ───────────────────────────────────────
  void _showForgotPasswordDialog() {
    final resetEmailController = TextEditingController();
    bool isSending = false;

    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return Dialog(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(24),
              ),
              backgroundColor: Colors.white,
              child: Padding(
                padding: const EdgeInsets.all(28),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Icon
                    Container(
                      width: 56,
                      height: 56,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        color: _primary.withOpacity(0.10),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.lock_reset_rounded,
                        color: _primary,
                        size: 28,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Title
                    const Text(
                      'Reset Password',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: _onSurface,
                        letterSpacing: -0.3,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'Enter your registered email and we\'ll send you a reset link.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 13,
                        color: _subtle,
                        height: 1.5,
                      ),
                    ),
                    const SizedBox(height: 24),

                    // Email input
                    _InputField(
                      controller: resetEmailController,
                      label: 'Email address',
                      hint: 'you@example.com',
                      icon: Icons.alternate_email_rounded,
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 24),

                    // Send button
                    SizedBox(
                      height: 50,
                      child: ElevatedButton(
                        onPressed: isSending
                            ? null
                            : () async {
                                final email =
                                    resetEmailController.text.trim();
                                if (email.isEmpty) {
                                  _showSnack(
                                      'Please enter your email address.');
                                  return;
                                }
                                setDialogState(() => isSending = true);
                                try {
                                  await _auth
                                      .sendPasswordResetEmail(email: email);
                                  if (!mounted) return;
                                  Navigator.of(dialogContext).pop();
                                  _showSnack(
                                    'Reset link sent! Check your inbox.',
                                    isError: false,
                                  );
                                } on FirebaseAuthException catch (e) {
                                  if (e.code == 'user-not-found') {
                                    _showSnack(
                                        'No account found with this email.');
                                  } else if (e.code ==
                                      'invalid-email') {
                                    _showSnack(
                                        'Please enter a valid email address.');
                                  } else {
                                    _showSnack('Error: ${e.message}');
                                  }
                                } finally {
                                  if (mounted) {
                                    setDialogState(
                                        () => isSending = false);
                                  }
                                }
                              },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _primary,
                          disabledBackgroundColor:
                              _primary.withOpacity(0.6),
                          foregroundColor: Colors.white,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: isSending
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 2.5,
                                ),
                              )
                            : const Text(
                                'Send Reset Link',
                                style: TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Cancel button
                    TextButton(
                      onPressed: () => Navigator.of(dialogContext).pop(),
                      child: const Text(
                        'Cancel',
                        style: TextStyle(
                          color: _subtle,
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  // ─── Toggle login/signup ──────────────────────────────────────────────────
  void _toggleMode() {
    _controller.reset();
    setState(() => isLogin = !isLogin);
    _controller.forward();
  }

  // ─── Build ────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: _surface,
      body: Stack(
        children: [
          _BackgroundDecoration(size: size),
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding:
                    const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
                child: FadeTransition(
                  opacity: _fadeIn,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _BrandHeader(),
                      const SizedBox(height: 36),
                      _AuthCard(
                        isLogin: isLogin,
                        isLoading: _isLoading,
                        obscurePassword: _obscurePassword,
                        emailController: _emailController,
                        passwordController: _passwordController,
                        onToggleObscure: () => setState(
                            () => _obscurePassword = !_obscurePassword),
                        onSubmit: _submit,
                        onResendVerification: _resendVerification,
                        onToggleMode: _toggleMode,
                        onForgotPassword: _showForgotPasswordDialog, // ✅ NEW
                      ),
                      const SizedBox(height: 24),
                      Text(
                        'By continuing, you agree to SignBridge\'s\nTerms of Service and Privacy Policy.',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 11,
                          color: _subtle.withOpacity(0.8),
                          height: 1.6,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────── Background Decoration ────────────────────────── //

class _BackgroundDecoration extends StatelessWidget {
  final Size size;
  const _BackgroundDecoration({required this.size});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned(
          top: -80,
          left: -80,
          child: _Blob(
            size: 260,
            color: const Color(0xFF4F6EF7).withOpacity(0.18),
          ),
        ),
        Positioned(
          bottom: -60,
          right: -60,
          child: _Blob(
            size: 220,
            color: const Color(0xFF7C3AED).withOpacity(0.15),
          ),
        ),
        Positioned.fill(
          child: CustomPaint(painter: _GridPainter()),
        ),
      ],
    );
  }
}

class _Blob extends StatelessWidget {
  final double size;
  final Color color;
  const _Blob({required this.size, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(shape: BoxShape.circle, color: color),
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
  bool shouldRepaint(_GridPainter oldDelegate) => false;
}

// ─────────────────────────────── Brand Header ─────────────────────────────── //

class _BrandHeader extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 64,
          height: 64,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF4F6EF7), Color(0xFF7C3AED)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(18),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF4F6EF7).withOpacity(0.40),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: const Icon(
            Icons.sign_language_rounded,
            size: 34,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 14),
        const Text(
          'SignBridge',
          style: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.8,
            color: Color(0xFF1A1D2E),
          ),
        ),
        const SizedBox(height: 4),
        const Text(
          'Connecting signs to text seamlessly',
          style: TextStyle(
            fontSize: 13.5,
            color: Color(0xFF8890AD),
            letterSpacing: 0.2,
          ),
        ),
      ],
    );
  }
}

// ──────────────────────────────── Auth Card ───────────────────────────────── //

class _AuthCard extends StatelessWidget {
  final bool isLogin;
  final bool isLoading;
  final bool obscurePassword;
  final TextEditingController emailController;
  final TextEditingController passwordController;
  final VoidCallback onToggleObscure;
  final VoidCallback onSubmit;
  final VoidCallback onResendVerification;
  final VoidCallback onToggleMode;
  final VoidCallback onForgotPassword; // ✅ NEW

  const _AuthCard({
    required this.isLogin,
    required this.isLoading,
    required this.obscurePassword,
    required this.emailController,
    required this.passwordController,
    required this.onToggleObscure,
    required this.onSubmit,
    required this.onResendVerification,
    required this.onToggleMode,
    required this.onForgotPassword, // ✅ NEW
  });

  static const _primary   = Color(0xFF4F6EF7);
  static const _onSurface = Color(0xFF1A1D2E);
  static const _subtle    = Color(0xFF8890AD);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF4F6EF7).withOpacity(0.10),
            blurRadius: 40,
            offset: const Offset(0, 16),
          ),
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Mode heading ─────────────────────────────────────────────────
            Row(
              children: [
                Expanded(
                  child: Text(
                    isLogin ? 'Welcome back' : 'Create account',
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: _onSurface,
                      letterSpacing: -0.4,
                    ),
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _primary.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    isLogin ? 'Sign In' : 'Sign Up',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: _primary,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              isLogin
                  ? 'Enter your credentials to continue'
                  : 'Fill in your details to get started',
              style: const TextStyle(fontSize: 13, color: _subtle),
            ),
            const SizedBox(height: 28),

            // ── Email field ──────────────────────────────────────────────────
            _InputField(
              controller: emailController,
              label: 'Email address',
              hint: 'you@example.com',
              icon: Icons.alternate_email_rounded,
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 16),

            // ── Password field ───────────────────────────────────────────────
            _InputField(
              controller: passwordController,
              label: 'Password',
              hint: '••••••••',
              icon: Icons.lock_outline_rounded,
              obscureText: obscurePassword,
              suffixIcon: IconButton(
                icon: Icon(
                  obscurePassword
                      ? Icons.visibility_off_outlined
                      : Icons.visibility_outlined,
                  size: 20,
                  color: _subtle,
                ),
                onPressed: onToggleObscure,
              ),
            ),

            // ── Password hint (signup only) ──────────────────────────────────
            if (!isLogin) ...[
              const SizedBox(height: 8),
              const Text(
                'Min 6 chars · Uppercase · Lowercase · Number',
                style: TextStyle(fontSize: 11.5, color: _subtle),
              ),
            ],

            // ── ✅ NEW: Forgot Password link (login only) ────────────────────
            if (isLogin) ...[
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerRight,
                child: GestureDetector(
                  onTap: onForgotPassword,
                  child: const Text(
                    'Forgot password?',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: _primary,
                    ),
                  ),
                ),
              ),
            ],

            const SizedBox(height: 28),

            // ── Submit button ────────────────────────────────────────────────
            SizedBox(
              height: 52,
              child: ElevatedButton(
                onPressed: isLoading ? null : onSubmit,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _primary,
                  disabledBackgroundColor: _primary.withOpacity(0.6),
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shadowColor: Colors.transparent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                child: isLoading
                    ? const SizedBox(
                        width: 22,
                        height: 22,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2.5,
                        ),
                      )
                    : Text(
                        isLogin ? 'Sign In' : 'Create Account',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 16),

            // ── Resend verification (login only) ─────────────────────────────
            if (isLogin)
              Center(
                child: TextButton.icon(
                  onPressed: onResendVerification,
                  icon: const Icon(Icons.mark_email_unread_outlined, size: 15),
                  label: const Text('Resend verification email'),
                  style: TextButton.styleFrom(
                    foregroundColor: _subtle,
                    textStyle: const TextStyle(fontSize: 13),
                  ),
                ),
              ),

            const Divider(
                height: 28, thickness: 0.8, color: Color(0xFFEEF0F6)),

            // ── Toggle login/signup ──────────────────────────────────────────
            Center(
              child: GestureDetector(
                onTap: onToggleMode,
                child: RichText(
                  text: TextSpan(
                    style: const TextStyle(fontSize: 13.5, color: _subtle),
                    children: [
                      TextSpan(
                        text: isLogin
                            ? "Don't have an account? "
                            : 'Already have an account? ',
                      ),
                      TextSpan(
                        text: isLogin ? 'Create one' : 'Sign in',
                        style: const TextStyle(
                          color: _primary,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────── Input Field ──────────────────────────────── //

class _InputField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;
  final IconData icon;
  final bool obscureText;
  final TextInputType? keyboardType;
  final Widget? suffixIcon;

  const _InputField({
    required this.controller,
    required this.label,
    required this.hint,
    required this.icon,
    this.obscureText = false,
    this.keyboardType,
    this.suffixIcon,
  });

  static const _primary   = Color(0xFF4F6EF7);
  static const _onSurface = Color(0xFF1A1D2E);
  static const _subtle    = Color(0xFF8890AD);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12.5,
            fontWeight: FontWeight.w600,
            color: _onSurface,
            letterSpacing: 0.1,
          ),
        ),
        const SizedBox(height: 6),
        TextField(
          controller: controller,
          obscureText: obscureText,
          keyboardType: keyboardType,
          style: const TextStyle(fontSize: 14.5, color: _onSurface),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle:
                TextStyle(color: _subtle.withOpacity(0.7), fontSize: 14),
            prefixIcon: Icon(icon, size: 19, color: _subtle),
            suffixIcon: suffixIcon,
            filled: true,
            fillColor: const Color(0xFFF4F6FF),
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide.none,
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(
                color: Color(0xFFE8EBFB),
                width: 1.2,
              ),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: _primary, width: 1.8),
            ),
          ),
        ),
      ],
    );
  }
}