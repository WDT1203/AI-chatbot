import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'fitbot_chat.dart'; // Adjust path as needed

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await Firebase.initializeApp();
    debugPrint("Firebase initialized successfully");
  } catch (e) {
    debugPrint("Firebase initialization failed: $e");
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFF17181D),
        fontFamily: 'Inter',
      ),
      home: const FitbotChat(),
    );
  }
}