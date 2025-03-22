// lib/parent_dashboard.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart'; // Fix 1, 2, 3, 4
import 'fitbot_chat.dart'; // Adjust path if needed

class ParentDashboard extends StatefulWidget {
  const ParentDashboard({super.key});

  @override
  State<ParentDashboard> createState() => _ParentDashboardState();
}

class _ParentDashboardState extends State<ParentDashboard> {
  final TextEditingController _childUidController = TextEditingController();

  Future<void> _linkChildAccount() async {
    final parentUser = FirebaseAuth.instance.currentUser;
    if (parentUser == null) {
      if (!mounted) return; // Fix 5
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please sign in as a parent")),
      );
      return;
    }
    final childUid = _childUidController.text.trim();
    if (childUid.isEmpty) {
      if (!mounted) return; // Fix 5
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please enter a child UID")),
      );
      return;
    }

    try {
      await FirebaseFirestore.instance
          .collection('parent_child_links')
          .doc(parentUser.uid)
          .set(
            {'child_uids': FieldValue.arrayUnion([childUid])}, // Fix 3
            SetOptions(merge: true), // Fix 4
          );
      if (!mounted) return; // Fix 5
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Child account linked successfully")),
      );
    } catch (e) {
      debugPrint("Error linking child: $e");
      if (!mounted) return; // Fix 5
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to link child: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    return Scaffold(
      appBar: AppBar(
        title: const Text("Parent Dashboard"),
        actions: [
          IconButton(
            icon: const Icon(Icons.chat),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => const FitbotChat()),
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text("Signed in as: ${user?.uid ?? 'Not signed in'}"),
            const SizedBox(height: 20),
            TextField(
              controller: _childUidController,
              decoration: const InputDecoration(
                labelText: "Enter Child's UID",
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _linkChildAccount,
              child: const Text("Link Child Account"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                final childUid = _childUidController.text.trim();
                if (childUid.isNotEmpty) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => FitbotChat(targetUid: childUid),
                    ),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Please enter a child UID")),
                  );
                }
              },
              child: const Text("View Child's Chat History"),
            ),
          ],
        ),
      ),
    );
  }
}