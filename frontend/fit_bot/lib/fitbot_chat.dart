import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:google_fonts/google_fonts.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Message {
  final String text;
  final bool isUser;
  final DateTime time;
  String displayText;
  bool isComplete;
  bool isLoading;

  Message({
    required this.text,
    required this.isUser,
    DateTime? time,
  })  : time = time ?? DateTime.now(),
        displayText = isUser ? text : "",
        isComplete = isUser,
        isLoading = !isUser;
}

class FitbotChat extends StatefulWidget {
  final String? targetUid; // Optional child UID for parents
  const FitbotChat({super.key, this.targetUid});

  @override
  State<FitbotChat> createState() => _FitbotChatState();
}

class _FitbotChatState extends State<FitbotChat> with TickerProviderStateMixin {  final TextEditingController _messageController = TextEditingController();
  final List<Message> _messages = [];
  bool _isLoading = false;
  Timer? _animationTimer;
  Timer? _loadingDotsTimer;
  final ScrollController _scrollController = ScrollController();
  int _dotsCount = 0;
  late AnimationController _fadeController;

  @override
  void initState() {
    super.initState();
    _initializeFirebaseAuth(); // Add this
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
    final welcomeMessage = Message(
      text: "Hello! I’m here to help you with drug addiction prevention. How can I assist you today?",
      isUser: false,
    );
    _messages.add(welcomeMessage);
    _showLoadingDots(welcomeMessage);
  }

  Future<void> _initializeFirebaseAuth() async {
    try {
      if (FirebaseAuth.instance.currentUser == null) {
        await FirebaseAuth.instance.signInAnonymously();
        debugPrint("Signed in anonymously: ${FirebaseAuth.instance.currentUser!.uid}");
      } else {
        debugPrint("Already signed in: ${FirebaseAuth.instance.currentUser!.uid}");
      }
    } catch (e) {
      debugPrint("Firebase Auth initialization failed: $e");
    }
  }

  @override
  void dispose() {
    _animationTimer?.cancel();
    _loadingDotsTimer?.cancel();
    _fadeController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _showLoadingDots(Message message) {
    _dotsCount = 0;
    setState(() {
      message.displayText = ".";
      message.isLoading = true;
    });

    _loadingDotsTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      _dotsCount = (_dotsCount + 1) % 4;
      setState(() {
        message.displayText = "." * _dotsCount;
      });
      if (timer.tick >= 4) {
        timer.cancel();
        setState(() {
          message.isLoading = false;
          message.displayText = "";
        });
        _animateText(message);
      }
    });
  }

  void _animateText(Message message) {
    message.displayText = "";
    int currentIndex = 0;
    const duration = Duration(milliseconds: 35);

    _animationTimer = Timer.periodic(duration, (timer) {
      if (currentIndex < message.text.length) {
        _fadeController.reset();
        _fadeController.forward();

        setState(() {
          message.displayText = message.text.substring(0, currentIndex + 1);
          currentIndex++;
        });
        _scrollToBottom();
      } else {
        setState(() {
          message.isComplete = true;
        });
        timer.cancel();
      }
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage() async {
    final userMessage = _messageController.text.trim();
    if (userMessage.isEmpty) return;

    setState(() {
      _messages.add(Message(text: userMessage, isUser: true));
      _isLoading = true;
      _messageController.clear();
    });

    _scrollToBottom();

    try {
      final response = await _fetchBackendResponse(userMessage);

      final newMessage = Message(text: response, isUser: false);
      setState(() {
        _messages.add(newMessage);
        _isLoading = false;
      });

      _showLoadingDots(newMessage);
    } catch (e) {
      debugPrint("Error occurred: $e");
      final errorMessage = Message(
        text: "Sorry, I encountered an error. Please try again later.",
        isUser: false,
      );
      setState(() {
        _messages.add(errorMessage);
        _isLoading = false;
      });
      _showLoadingDots(errorMessage);
    }
  }

  Future<String> _fetchBackendResponse(String userMessage) async {
    try {
        String? idToken;
        final user = FirebaseAuth.instance.currentUser;
        if (user != null) {
            idToken = await user.getIdToken();
            debugPrint("Sending message with idToken: $idToken");
        } else {
            throw Exception("No signed-in user");
        }
        const backendUrl = 'http://localhost:5000/chat';
        final Map<String, dynamic> requestBody = {
            'query': userMessage,
            'idToken': idToken,
        };
        final response = await http.post(
            Uri.parse(backendUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(requestBody),
        );
        debugPrint("Backend Response Status: ${response.statusCode}");
        debugPrint("Response body: ${response.body}");
        if (response.statusCode == 200) {
            final Map<String, dynamic> responseData = jsonDecode(response.body);
            return responseData['response'] ?? "No response from server.";
        } else {
            return "Server error: ${response.statusCode}";
        }
    } catch (e) {
        debugPrint("Exception in backend call: $e");
        return "I encountered a technical issue. Please try again later.";
    }
}

  Future<List<Map<String, String>>> _fetchConversationHistory() async {
  try {
    String? idToken;
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      idToken = await user.getIdToken();
      debugPrint("Fetching history with idToken: $idToken");
    } else {
      debugPrint("No signed-in user for history fetch");
      throw Exception("No signed-in user");
    }
    const backendUrl = 'http://localhost:5000/conversation_history';
    final Map<String, dynamic> requestBody = {
      'idToken': idToken,
      if (widget.targetUid != null) 'target_uid': widget.targetUid,
    };
    final response = await http.post(
      Uri.parse(backendUrl),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(requestBody),
    );
    debugPrint("History fetch status: ${response.statusCode}");
    debugPrint("History fetch response: ${response.body}");
    if (response.statusCode == 200) {
      final Map<String, dynamic> responseData = jsonDecode(response.body);
      final List<dynamic> history = responseData['history'] ?? [];
      debugPrint("History items fetched: ${history.length}");
      return history
          .map((item) => {
                'prompt': item['prompt'] as String,
                'answer': item['answer'] as String? ?? 'No response',
                'timestamp': item['timestamp'] as String? ?? 'N/A',
              })
          .toList();
    } else {
      throw Exception("Server error: ${response.statusCode}");
    }
  } catch (e) {
    debugPrint("Error fetching history: $e");
    return []; // Explicitly return an empty list on error
  }
}


  void _showConversationHistory() async {
    final history = await _fetchConversationHistory();
    if (!mounted) return;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(
          'Conversation History',
          style: GoogleFonts.ebGaramond(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        content: SizedBox(
          width: double.maxFinite,
          height: 400,
          child: history.isEmpty
              ? Center(
                  child: Text(
                    'No conversation history yet.',
                    style: GoogleFonts.ebGaramond(fontSize: 16),
                  ),
                )
              : ListView.builder(
                  itemCount: history.length,
                  itemBuilder: (context, index) {
                    final item = history[index];
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'You: ${item['prompt']}',
                            style: GoogleFonts.ebGaramond(
                              fontSize: 16,
                              color: const Color(0xFF8C6E63),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Assistant: ${item['answer']}',
                            style: GoogleFonts.ebGaramond(
                              fontSize: 16,
                              color: const Color(0xFF3E2522),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Time: ${item['timestamp']}',
                            style: GoogleFonts.ebGaramond(
                              fontSize: 14,
                              color: Colors.grey,
                            ),
                          ),
                          const Divider(),
                        ],
                      ),
                    );
                  },
                ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'Close',
              style: GoogleFonts.ebGaramond(
                fontSize: 16,
                color: const Color(0xFF3E2522),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF3E2522),
        elevation: 0,
        toolbarHeight: 70,
        title: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: const Color(0xFFFFF2DF),
                borderRadius: BorderRadius.circular(20),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: ColorFiltered(
                  colorFilter: const ColorFilter.mode(
                    Color(0xFF3E2522),
                    BlendMode.srcATop,
                  ),
                  child: Image.asset(
                    'lib/assets/converted_image-4.png',
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Prevention Assistant',
                  style: GoogleFonts.ebGaramond(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: const Color(0xFF8C6E63),
                  ),
                ),
                Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: Colors.green,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Always active',
                      style: GoogleFonts.ebGaramond(
                        fontSize: 14,
                        color: Colors.green,
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.history, color: Color(0xFF8C6E63)),
            onPressed: _showConversationHistory,
            tooltip: 'View Conversation History',
          ),
          const SizedBox(width: 10),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('lib/assets/chat_background.jpeg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.all(16),
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  final message = _messages[index];
                  return _buildMessageBubble(message);
                },
              ),
            ),
            if (_isLoading)
              Container(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: const Center(
                  child: CircularProgressIndicator(
                    color: Color(0xFF8C6E63),
                  ),
                ),
              ),
            Container(
              color: Colors.transparent,
              padding: const EdgeInsets.all(16),
              child: Container(
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF2DF),
                  borderRadius: BorderRadius.circular(25),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 4,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.mic, color: Color(0xFF3E2522), size: 24),
                      onPressed: () {
                        // Speech to text functionality
                      },
                    ),
                    Expanded(
                      child: TextField(
                        controller: _messageController,
                        style: GoogleFonts.ebGaramond(
                          color: const Color(0xFF3E2522),
                          fontSize: 18,
                        ),
                        decoration: InputDecoration(
                          hintText: 'Ask about prevention...',
                          hintStyle: GoogleFonts.ebGaramond(
                            color: const Color(0xFF8C6E63),
                            fontSize: 18,
                          ),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(vertical: 14),
                        ),
                        onSubmitted: (_) => _sendMessage(),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.send, color: Color(0xFF3E2522), size: 24),
                      onPressed: _sendMessage,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageBubble(Message message) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: const Color(0xFFFFF2DF),
                borderRadius: BorderRadius.circular(16),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: ColorFiltered(
                  colorFilter: const ColorFilter.mode(
                    Color(0xFF3E2522),
                    BlendMode.srcATop,
                  ),
                  child: Image.asset(
                    'lib/assets/converted_image-4.png',
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              decoration: BoxDecoration(
                color: message.isUser ? const Color(0xFF8C6E63) : const Color(0xFFFFF2DF),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 3,
                    offset: const Offset(0, 1),
                  ),
                ],
              ),
              child: message.isLoading && !message.isUser
                  ? AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      child: Text(
                        message.displayText,
                        key: ValueKey<String>(message.displayText),
                        style: GoogleFonts.ebGaramond(
                          color: const Color(0xFF3E2522),
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    )
                  : AnimatedBuilder(
                      animation: _fadeController,
                      builder: (context, child) {
                        if (!message.isUser && !message.isComplete) {
                          String visibleText = message.displayText;
                          String lastChar = "";

                          if (visibleText.isNotEmpty) {
                            lastChar = visibleText.substring(visibleText.length - 1);
                            visibleText = visibleText.substring(0, visibleText.length - 1);
                          }

                          return RichText(
                            text: TextSpan(
                              children: [
                                TextSpan(
                                  text: visibleText,
                                  style: GoogleFonts.ebGaramond(
                                    color: message.isUser
                                        ? const Color(0xFFFFF2DF)
                                        : const Color(0xFF3E2522),
                                    fontSize: 18,
                                  ),
                                ),
                                TextSpan(
                                  text: lastChar,
                                  style: GoogleFonts.ebGaramond(
                                    color: message.isUser
                                        ? const Color(0xFFFFF2DF)
                                        : Color(0xFF3E2522).withValues(alpha: _fadeController.value),
                                    fontSize: 18,
                                  ),
                                ),
                              ],
                            ),
                          );
                        } else {
                          return Text(
                            message.isUser ? message.text : message.displayText,
                            style: GoogleFonts.ebGaramond(
                              color: message.isUser ? const Color(0xFFFFF2DF) : const Color(0xFF3E2522),
                              fontSize: 18,
                            ),
                          );
                        }
                      },
                    ),
            ),
          ),
          if (message.isUser) const SizedBox(width: 8),
        ],
      ),
    );
  }
}