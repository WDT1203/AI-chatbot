import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:fit_bot/main.dart'; // Ensure this points to your main.dart

void main() {
  // Setup Firebase mock or initialization
  setUpAll(() async {
    // Mock Firebase for testing (simplest approach)
    await Firebase.initializeApp(
      options: const FirebaseOptions(
        apiKey: 'test-api-key',
        appId: 'test-app-id',
        messagingSenderId: 'test-sender-id',
        projectId: 'test-project-id',
      ),
    );
  });

  testWidgets('Fitbot chat smoke test', (WidgetTester tester) async {
    // Build the app and trigger a frame
    await tester.pumpWidget(const MyApp());

    // Trigger a frame to allow async initialization (e.g., Firebase)
    await tester.pumpAndSettle();

    // Verify the app bar content
    expect(find.text('Prevention Assistant'), findsOneWidget);
    expect(find.text('Always active'), findsOneWidget);

    // Verify the initial bot message
    expect(
      find.text(
          "Hello! I’m here to help you with drug addiction prevention. How can I assist you today?"),
      findsOneWidget,
    );

    // Verify the message input field
    expect(find.widgetWithText(TextField, 'Ask about prevention...'),
        findsOneWidget);

    // Verify the send and mic icons
    expect(find.byIcon(Icons.mic), findsOneWidget);
    expect(find.byIcon(Icons.send), findsOneWidget);

    // Optional: Test sending a message (requires mocking HTTP if Flask is involved)
    await tester.enterText(
        find.byType(TextField), 'How can youth avoid drugs?');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pumpAndSettle(); // Wait for UI updates
    // Note: Without mocking HTTP, this won't show a response from Flask
  });
}