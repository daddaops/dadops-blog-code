"""
Generate a realistic 10K-query dataset with paraphrase clusters for cache benchmarking.

Blog post: https://dadops.dev/blog/caching-llm-responses/
Code Block 1.

The blog shows 3 example intents with a "... 137 more intents" comment.
We generate a full 140-intent dataset to match the blog's claim of ~3,140 unique intents.
"""
import hashlib
import random
import json


# Each intent has a canonical form and natural paraphrases.
# 140 intents across 5 categories to match blog's claimed dataset.
INTENTS = {
    # ── Category: QA (28 intents) ──
    "return_policy": {
        "category": "qa",
        "paraphrases": [
            "What is your return policy?",
            "How do I return an item?",
            "Can I send something back if I don't like it?",
            "What are the rules for returns?",
            "I want to return a purchase, how does that work?",
        ]
    },
    "shipping_time": {
        "category": "qa",
        "paraphrases": [
            "How long does shipping take?",
            "When will my order arrive?",
            "What's the estimated delivery time?",
            "How many days until I get my package?",
        ]
    },
    "payment_methods": {
        "category": "qa",
        "paraphrases": [
            "What payment methods do you accept?",
            "Can I pay with PayPal?",
            "Do you take credit cards?",
            "What are the available payment options?",
        ]
    },
    "account_delete": {
        "category": "qa",
        "paraphrases": [
            "How do I delete my account?",
            "Can I remove my profile?",
            "I want to close my account",
            "Delete my user account please",
        ]
    },
    "password_reset": {
        "category": "qa",
        "paraphrases": [
            "How do I reset my password?",
            "I forgot my password, what do I do?",
            "Can I change my password?",
            "Password reset instructions please",
        ]
    },
    "order_tracking": {
        "category": "qa",
        "paraphrases": [
            "How do I track my order?",
            "Where's my package?",
            "Can I see the delivery status?",
            "Track my shipment please",
        ]
    },
    "warranty_info": {
        "category": "qa",
        "paraphrases": [
            "What's the warranty on this product?",
            "How long is the warranty?",
            "Is there a guarantee?",
            "Warranty terms and conditions",
        ]
    },
    "pricing_info": {
        "category": "qa",
        "paraphrases": [
            "How much does it cost?",
            "What's the price?",
            "Can you tell me the pricing?",
            "How much do I have to pay?",
        ]
    },
    "cancel_order": {
        "category": "qa",
        "paraphrases": [
            "How do I cancel my order?",
            "I want to cancel a purchase",
            "Can I stop my order?",
            "Cancel my recent order please",
        ]
    },
    "contact_support": {
        "category": "qa",
        "paraphrases": [
            "How do I reach customer support?",
            "What's the support email?",
            "Can I talk to a human?",
            "How to contact you?",
        ]
    },
    "business_hours": {
        "category": "qa",
        "paraphrases": [
            "What are your business hours?",
            "When are you open?",
            "What time do you close?",
            "Operating hours please",
        ]
    },
    "size_guide": {
        "category": "qa",
        "paraphrases": [
            "Where's the size guide?",
            "How do I find my size?",
            "What size should I order?",
            "Size chart please",
        ]
    },
    "gift_cards": {
        "category": "qa",
        "paraphrases": [
            "Do you sell gift cards?",
            "Can I buy a gift certificate?",
            "How do gift cards work?",
            "Gift card options please",
        ]
    },
    "subscription_cancel": {
        "category": "qa",
        "paraphrases": [
            "How do I cancel my subscription?",
            "I want to unsubscribe",
            "Stop my recurring payment",
            "Cancel my plan",
        ]
    },
    "international_shipping": {
        "category": "qa",
        "paraphrases": [
            "Do you ship internationally?",
            "Can I order from outside the US?",
            "International delivery options?",
            "Shipping to Europe, is it possible?",
        ]
    },
    "discount_codes": {
        "category": "qa",
        "paraphrases": [
            "Do you have any discount codes?",
            "Are there any promo codes?",
            "How do I apply a coupon?",
            "Discount code not working",
        ]
    },
    "privacy_policy": {
        "category": "qa",
        "paraphrases": [
            "Where's your privacy policy?",
            "How do you handle my data?",
            "What data do you collect?",
            "Privacy and data protection info",
        ]
    },
    "bulk_orders": {
        "category": "qa",
        "paraphrases": [
            "Do you offer bulk pricing?",
            "Can I place a wholesale order?",
            "Bulk discount available?",
            "Large quantity order options",
        ]
    },
    "product_availability": {
        "category": "qa",
        "paraphrases": [
            "Is this product in stock?",
            "When will this be available again?",
            "Check product availability",
            "Is this item still for sale?",
        ]
    },
    "refund_status": {
        "category": "qa",
        "paraphrases": [
            "Where's my refund?",
            "How long does a refund take?",
            "Refund status check",
            "When will I get my money back?",
        ]
    },
    "exchange_policy": {
        "category": "qa",
        "paraphrases": [
            "Can I exchange an item?",
            "How do exchanges work?",
            "I want to swap for a different size",
            "Exchange policy details",
        ]
    },
    "loyalty_program": {
        "category": "qa",
        "paraphrases": [
            "Do you have a loyalty program?",
            "How do I earn rewards?",
            "Points and rewards info",
            "Tell me about your membership program",
        ]
    },
    "installation_help": {
        "category": "qa",
        "paraphrases": [
            "How do I install this?",
            "Installation instructions please",
            "Setup guide for this product",
            "Help me set this up",
        ]
    },
    "compatibility_check": {
        "category": "qa",
        "paraphrases": [
            "Is this compatible with my device?",
            "Will this work with my phone?",
            "Compatibility information please",
            "Check if this fits my setup",
        ]
    },
    "api_rate_limits": {
        "category": "qa",
        "paraphrases": [
            "What are the API rate limits?",
            "Rate limits for the API?",
            "How many requests can I make?",
            "API throttling limits",
        ]
    },
    "data_export": {
        "category": "qa",
        "paraphrases": [
            "Can I export my data?",
            "How do I download my data?",
            "Data export options",
            "Get a copy of my data",
        ]
    },
    "two_factor_auth": {
        "category": "qa",
        "paraphrases": [
            "How do I enable two-factor authentication?",
            "Set up 2FA on my account",
            "Two-step verification setup",
            "Enable MFA please",
        ]
    },
    "accessibility_features": {
        "category": "qa",
        "paraphrases": [
            "What accessibility features do you offer?",
            "Is your site screen reader friendly?",
            "Accessibility options available?",
            "ADA compliance information",
        ]
    },

    # ── Category: Summarization (28 intents) ──
    "summarize_email": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this email for me: {context}",
            "Give me a brief summary of the following email: {context}",
            "TL;DR this email: {context}",
            "What's the key point of this email? {context}",
        ]
    },
    "summarize_article": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this article: {context}",
            "Give me the main points of this article: {context}",
            "What's this article about? {context}",
            "Brief summary of the article: {context}",
        ]
    },
    "summarize_meeting": {
        "category": "summarization",
        "paraphrases": [
            "Summarize these meeting notes: {context}",
            "What were the key decisions from this meeting? {context}",
            "Meeting summary please: {context}",
            "Main takeaways from the meeting: {context}",
        ]
    },
    "summarize_report": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this report: {context}",
            "Key findings from this report: {context}",
            "Give me the executive summary: {context}",
            "What does this report say? {context}",
        ]
    },
    "summarize_document": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this document: {context}",
            "What's in this document? {context}",
            "Brief overview of this document: {context}",
            "Document summary please: {context}",
        ]
    },
    "summarize_conversation": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this conversation: {context}",
            "What was discussed? {context}",
            "Key points from this chat: {context}",
            "Conversation summary: {context}",
        ]
    },
    "summarize_research": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this research paper: {context}",
            "What are the findings? {context}",
            "Research paper summary: {context}",
            "Key results from this study: {context}",
        ]
    },
    "summarize_news": {
        "category": "summarization",
        "paraphrases": [
            "Summarize today's news: {context}",
            "What happened today? {context}",
            "News summary: {context}",
            "Headlines recap: {context}",
        ]
    },
    "summarize_book": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this book chapter: {context}",
            "What's the chapter about? {context}",
            "Chapter summary: {context}",
            "Key themes in this chapter: {context}",
        ]
    },
    "summarize_contract": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this contract: {context}",
            "What are the key terms? {context}",
            "Contract highlights: {context}",
            "Important clauses in this contract: {context}",
        ]
    },
    "summarize_review": {
        "category": "summarization",
        "paraphrases": [
            "Summarize these reviews: {context}",
            "What do the reviews say? {context}",
            "Overall review sentiment: {context}",
            "Review summary: {context}",
        ]
    },
    "summarize_changelog": {
        "category": "summarization",
        "paraphrases": [
            "Summarize the changelog: {context}",
            "What changed in this release? {context}",
            "Release notes summary: {context}",
            "What's new? {context}",
        ]
    },
    "summarize_transcript": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this transcript: {context}",
            "Key points from the transcript: {context}",
            "What was said? {context}",
            "Transcript summary please: {context}",
        ]
    },
    "summarize_proposal": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this proposal: {context}",
            "What's being proposed? {context}",
            "Proposal highlights: {context}",
            "Key points of the proposal: {context}",
        ]
    },
    "summarize_policy": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this policy document: {context}",
            "What does the policy say? {context}",
            "Policy summary: {context}",
            "Key policy requirements: {context}",
        ]
    },
    "summarize_incident": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this incident report: {context}",
            "What happened in this incident? {context}",
            "Incident summary: {context}",
            "Root cause and impact: {context}",
        ]
    },
    "summarize_feedback": {
        "category": "summarization",
        "paraphrases": [
            "Summarize customer feedback: {context}",
            "What are customers saying? {context}",
            "Feedback themes: {context}",
            "Common feedback points: {context}",
        ]
    },
    "summarize_legal": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this legal document: {context}",
            "What are the legal implications? {context}",
            "Legal brief summary: {context}",
            "Key legal points: {context}",
        ]
    },
    "summarize_spec": {
        "category": "summarization",
        "paraphrases": [
            "Summarize the tech spec: {context}",
            "What does the spec require? {context}",
            "Spec overview: {context}",
            "Technical requirements summary: {context}",
        ]
    },
    "summarize_guidelines": {
        "category": "summarization",
        "paraphrases": [
            "Summarize these guidelines: {context}",
            "What are the main guidelines? {context}",
            "Guidelines overview: {context}",
            "Key rules and guidelines: {context}",
        ]
    },
    "summarize_financial": {
        "category": "summarization",
        "paraphrases": [
            "Summarize the financial report: {context}",
            "What's the financial outlook? {context}",
            "Financial highlights: {context}",
            "Key financial metrics: {context}",
        ]
    },
    "summarize_strategy": {
        "category": "summarization",
        "paraphrases": [
            "Summarize the strategy document: {context}",
            "What's the strategic plan? {context}",
            "Strategy overview: {context}",
            "Key strategic initiatives: {context}",
        ]
    },
    "summarize_tutorial": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this tutorial: {context}",
            "What does the tutorial cover? {context}",
            "Tutorial outline: {context}",
            "Main steps in the tutorial: {context}",
        ]
    },
    "summarize_complaint": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this complaint: {context}",
            "What's the customer complaining about? {context}",
            "Complaint summary: {context}",
            "Issue description: {context}",
        ]
    },
    "summarize_survey": {
        "category": "summarization",
        "paraphrases": [
            "Summarize the survey results: {context}",
            "What did the survey find? {context}",
            "Survey highlights: {context}",
            "Key survey findings: {context}",
        ]
    },
    "summarize_whitepaper": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this whitepaper: {context}",
            "What's the whitepaper about? {context}",
            "Key arguments in the whitepaper: {context}",
            "Whitepaper summary: {context}",
        ]
    },
    "summarize_presentation": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this presentation: {context}",
            "What's the presentation about? {context}",
            "Slide deck summary: {context}",
            "Key presentation points: {context}",
        ]
    },
    "summarize_rfc": {
        "category": "summarization",
        "paraphrases": [
            "Summarize this RFC: {context}",
            "What's being proposed in this RFC? {context}",
            "RFC overview: {context}",
            "Key design decisions: {context}",
        ]
    },

    # ── Category: Extraction (28 intents) ──
    "extract_entities": {
        "category": "extraction",
        "paraphrases": [
            "Extract all named entities from this text: {context}",
            "Find the people, places, and organizations mentioned: {context}",
            "Named entity extraction: {context}",
            "What entities are in this text? {context}",
        ]
    },
    "extract_dates": {
        "category": "extraction",
        "paraphrases": [
            "Extract all dates from this text: {context}",
            "Find the dates mentioned: {context}",
            "What dates appear in this text? {context}",
            "Date extraction: {context}",
        ]
    },
    "extract_prices": {
        "category": "extraction",
        "paraphrases": [
            "Extract all prices from this text: {context}",
            "Find the monetary amounts: {context}",
            "What are the prices mentioned? {context}",
            "Price extraction: {context}",
        ]
    },
    "extract_contacts": {
        "category": "extraction",
        "paraphrases": [
            "Extract contact information from this text: {context}",
            "Find emails and phone numbers: {context}",
            "What contact details are in this text? {context}",
            "Contact info extraction: {context}",
        ]
    },
    "extract_keywords": {
        "category": "extraction",
        "paraphrases": [
            "Extract the main keywords: {context}",
            "What are the key terms? {context}",
            "Keyword extraction from this text: {context}",
            "Identify important terms: {context}",
        ]
    },
    "extract_action_items": {
        "category": "extraction",
        "paraphrases": [
            "Extract action items from these notes: {context}",
            "What tasks need to be done? {context}",
            "List the action items: {context}",
            "Find the to-do items: {context}",
        ]
    },
    "extract_requirements": {
        "category": "extraction",
        "paraphrases": [
            "Extract the requirements from this document: {context}",
            "What are the requirements? {context}",
            "List all requirements: {context}",
            "Requirements extraction: {context}",
        ]
    },
    "extract_metrics": {
        "category": "extraction",
        "paraphrases": [
            "Extract all metrics and numbers: {context}",
            "Find the key metrics: {context}",
            "What numbers are mentioned? {context}",
            "Metric extraction: {context}",
        ]
    },
    "extract_urls": {
        "category": "extraction",
        "paraphrases": [
            "Extract all URLs from this text: {context}",
            "Find the links: {context}",
            "What URLs are mentioned? {context}",
            "URL extraction: {context}",
        ]
    },
    "extract_skills": {
        "category": "extraction",
        "paraphrases": [
            "Extract skills from this resume: {context}",
            "What skills are listed? {context}",
            "Skill extraction from resume: {context}",
            "List the candidate's skills: {context}",
        ]
    },
    "extract_locations": {
        "category": "extraction",
        "paraphrases": [
            "Extract locations from this text: {context}",
            "Find all place names: {context}",
            "What locations are mentioned? {context}",
            "Geographic extraction: {context}",
        ]
    },
    "extract_products": {
        "category": "extraction",
        "paraphrases": [
            "Extract product names from this text: {context}",
            "What products are mentioned? {context}",
            "Product extraction: {context}",
            "Find all product references: {context}",
        ]
    },
    "extract_sentiment_phrases": {
        "category": "extraction",
        "paraphrases": [
            "Extract sentiment-bearing phrases: {context}",
            "Find the opinions in this text: {context}",
            "What are the sentiment expressions? {context}",
            "Opinion phrase extraction: {context}",
        ]
    },
    "extract_technical_terms": {
        "category": "extraction",
        "paraphrases": [
            "Extract technical terms: {context}",
            "Find the jargon and technical words: {context}",
            "Technical terminology extraction: {context}",
            "What technical terms are used? {context}",
        ]
    },
    "extract_relationships": {
        "category": "extraction",
        "paraphrases": [
            "Extract relationships between entities: {context}",
            "Find the connections between people and organizations: {context}",
            "Relationship extraction: {context}",
            "How are the entities related? {context}",
        ]
    },
    "extract_abbreviations": {
        "category": "extraction",
        "paraphrases": [
            "Extract abbreviations and their meanings: {context}",
            "Find all acronyms: {context}",
            "What abbreviations are used? {context}",
            "Acronym extraction: {context}",
        ]
    },
    "extract_citations": {
        "category": "extraction",
        "paraphrases": [
            "Extract citations and references: {context}",
            "Find all cited works: {context}",
            "What papers are referenced? {context}",
            "Citation extraction: {context}",
        ]
    },
    "extract_deadlines": {
        "category": "extraction",
        "paraphrases": [
            "Extract deadlines from this text: {context}",
            "What are the due dates? {context}",
            "Find all deadlines mentioned: {context}",
            "Deadline extraction: {context}",
        ]
    },
    "extract_risks": {
        "category": "extraction",
        "paraphrases": [
            "Extract risks from this document: {context}",
            "What risks are identified? {context}",
            "Risk extraction: {context}",
            "List all potential risks: {context}",
        ]
    },
    "extract_dependencies": {
        "category": "extraction",
        "paraphrases": [
            "Extract dependencies from this spec: {context}",
            "What dependencies are mentioned? {context}",
            "Dependency extraction: {context}",
            "Find all prerequisites: {context}",
        ]
    },
    "extract_assumptions": {
        "category": "extraction",
        "paraphrases": [
            "Extract assumptions from this plan: {context}",
            "What assumptions are being made? {context}",
            "Assumption extraction: {context}",
            "List the underlying assumptions: {context}",
        ]
    },
    "extract_definitions": {
        "category": "extraction",
        "paraphrases": [
            "Extract definitions from this document: {context}",
            "What terms are defined? {context}",
            "Definition extraction: {context}",
            "Find all defined terms: {context}",
        ]
    },
    "extract_events": {
        "category": "extraction",
        "paraphrases": [
            "Extract events from this timeline: {context}",
            "What events are described? {context}",
            "Event extraction: {context}",
            "List all events mentioned: {context}",
        ]
    },
    "extract_arguments": {
        "category": "extraction",
        "paraphrases": [
            "Extract the main arguments: {context}",
            "What arguments are made? {context}",
            "Argument extraction: {context}",
            "List the key arguments: {context}",
        ]
    },
    "extract_questions": {
        "category": "extraction",
        "paraphrases": [
            "Extract questions from this text: {context}",
            "What questions are asked? {context}",
            "Question extraction: {context}",
            "Find all questions: {context}",
        ]
    },
    "extract_instructions": {
        "category": "extraction",
        "paraphrases": [
            "Extract the instructions: {context}",
            "What are the steps? {context}",
            "Instruction extraction: {context}",
            "List the procedural steps: {context}",
        ]
    },
    "extract_conclusions": {
        "category": "extraction",
        "paraphrases": [
            "Extract the conclusions: {context}",
            "What are the conclusions? {context}",
            "Conclusion extraction: {context}",
            "Final conclusions: {context}",
        ]
    },
    "extract_recommendations": {
        "category": "extraction",
        "paraphrases": [
            "Extract recommendations: {context}",
            "What is recommended? {context}",
            "Find the recommendations: {context}",
            "List all recommendations: {context}",
        ]
    },

    # ── Category: Classification (28 intents) ──
    "classify_sentiment": {
        "category": "classification",
        "paraphrases": [
            "What's the sentiment of this text? {context}",
            "Is this positive or negative? {context}",
            "Classify the sentiment: {context}",
            "Sentiment analysis: {context}",
        ]
    },
    "classify_topic": {
        "category": "classification",
        "paraphrases": [
            "What topic is this about? {context}",
            "Classify the topic of this text: {context}",
            "What category does this belong to? {context}",
            "Topic classification: {context}",
        ]
    },
    "classify_urgency": {
        "category": "classification",
        "paraphrases": [
            "How urgent is this? {context}",
            "Is this high or low priority? {context}",
            "Urgency classification: {context}",
            "Rate the urgency: {context}",
        ]
    },
    "classify_spam": {
        "category": "classification",
        "paraphrases": [
            "Is this spam? {context}",
            "Classify as spam or not spam: {context}",
            "Spam detection: {context}",
            "Is this message legitimate? {context}",
        ]
    },
    "classify_language": {
        "category": "classification",
        "paraphrases": [
            "What language is this? {context}",
            "Detect the language: {context}",
            "Language classification: {context}",
            "Which language is this written in? {context}",
        ]
    },
    "classify_intent": {
        "category": "classification",
        "paraphrases": [
            "What's the user's intent? {context}",
            "Classify the intent of this message: {context}",
            "Intent detection: {context}",
            "What does the user want? {context}",
        ]
    },
    "classify_complexity": {
        "category": "classification",
        "paraphrases": [
            "How complex is this query? {context}",
            "Rate the complexity: {context}",
            "Simple or complex question? {context}",
            "Complexity classification: {context}",
        ]
    },
    "classify_category": {
        "category": "classification",
        "paraphrases": [
            "Categorize this item: {context}",
            "What category is this? {context}",
            "Product categorization: {context}",
            "Assign a category: {context}",
        ]
    },
    "classify_safety": {
        "category": "classification",
        "paraphrases": [
            "Is this content safe? {context}",
            "Content safety classification: {context}",
            "Flag unsafe content: {context}",
            "Safety check on this text: {context}",
        ]
    },
    "classify_readability": {
        "category": "classification",
        "paraphrases": [
            "What's the reading level? {context}",
            "Readability classification: {context}",
            "Grade level of this text: {context}",
            "How easy is this to read? {context}",
        ]
    },
    "classify_emotion": {
        "category": "classification",
        "paraphrases": [
            "What emotion is expressed? {context}",
            "Emotion classification: {context}",
            "What's the emotional tone? {context}",
            "Detect the emotion: {context}",
        ]
    },
    "classify_formality": {
        "category": "classification",
        "paraphrases": [
            "Is this formal or informal? {context}",
            "Formality level: {context}",
            "Register classification: {context}",
            "How formal is this text? {context}",
        ]
    },
    "classify_document_type": {
        "category": "classification",
        "paraphrases": [
            "What type of document is this? {context}",
            "Document type classification: {context}",
            "Is this a letter, report, or memo? {context}",
            "Classify the document type: {context}",
        ]
    },
    "classify_source": {
        "category": "classification",
        "paraphrases": [
            "Where is this from? {context}",
            "Source classification: {context}",
            "Identify the source: {context}",
            "What platform is this from? {context}",
        ]
    },
    "classify_truthfulness": {
        "category": "classification",
        "paraphrases": [
            "Is this claim true or false? {context}",
            "Fact check this: {context}",
            "Truthfulness assessment: {context}",
            "Verify this claim: {context}",
        ]
    },
    "classify_bias": {
        "category": "classification",
        "paraphrases": [
            "Is this text biased? {context}",
            "Detect bias in this text: {context}",
            "Bias classification: {context}",
            "Is there a political leaning? {context}",
        ]
    },
    "classify_sarcasm": {
        "category": "classification",
        "paraphrases": [
            "Is this sarcastic? {context}",
            "Sarcasm detection: {context}",
            "Literal or sarcastic? {context}",
            "Detect sarcasm: {context}",
        ]
    },
    "classify_severity": {
        "category": "classification",
        "paraphrases": [
            "How severe is this issue? {context}",
            "Severity classification: {context}",
            "Rate the severity: critical, high, medium, low? {context}",
            "Issue severity level: {context}",
        ]
    },
    "classify_domain": {
        "category": "classification",
        "paraphrases": [
            "What domain is this about? {context}",
            "Domain classification: {context}",
            "Tech, finance, health, or other? {context}",
            "Identify the domain: {context}",
        ]
    },
    "classify_quality": {
        "category": "classification",
        "paraphrases": [
            "Rate the quality of this text: {context}",
            "Quality assessment: {context}",
            "Is this well-written? {context}",
            "Text quality classification: {context}",
        ]
    },
    "classify_relevance": {
        "category": "classification",
        "paraphrases": [
            "Is this relevant to the query? {context}",
            "Relevance classification: {context}",
            "How relevant is this result? {context}",
            "Relevance scoring: {context}",
        ]
    },
    "classify_completeness": {
        "category": "classification",
        "paraphrases": [
            "Is this response complete? {context}",
            "Completeness check: {context}",
            "Is anything missing? {context}",
            "Completeness classification: {context}",
        ]
    },
    "classify_audience": {
        "category": "classification",
        "paraphrases": [
            "Who is the target audience? {context}",
            "Audience classification: {context}",
            "Is this for beginners or experts? {context}",
            "Target audience identification: {context}",
        ]
    },
    "classify_tone": {
        "category": "classification",
        "paraphrases": [
            "What's the tone of this text? {context}",
            "Tone classification: {context}",
            "Is this friendly, neutral, or hostile? {context}",
            "Detect the tone: {context}",
        ]
    },
    "classify_genre": {
        "category": "classification",
        "paraphrases": [
            "What genre is this? {context}",
            "Genre classification: {context}",
            "Is this fiction or non-fiction? {context}",
            "Literary genre: {context}",
        ]
    },
    "classify_credibility": {
        "category": "classification",
        "paraphrases": [
            "How credible is this source? {context}",
            "Credibility assessment: {context}",
            "Is this a reliable source? {context}",
            "Source credibility: {context}",
        ]
    },
    "classify_technical_level": {
        "category": "classification",
        "paraphrases": [
            "What's the technical level? {context}",
            "Technical depth classification: {context}",
            "Is this beginner or advanced? {context}",
            "Skill level required: {context}",
        ]
    },
    "classify_action_required": {
        "category": "classification",
        "paraphrases": [
            "Is action required? {context}",
            "Does this need a response? {context}",
            "Action required classification: {context}",
            "Is this actionable? {context}",
        ]
    },

    # ── Category: Generation (28 intents) ──
    "generate_email": {
        "category": "generation",
        "paraphrases": [
            "Write a professional email about: {context}",
            "Draft an email regarding: {context}",
            "Compose an email about: {context}",
            "Help me write an email: {context}",
        ]
    },
    "generate_summary": {
        "category": "generation",
        "paraphrases": [
            "Write a summary of: {context}",
            "Create an executive summary: {context}",
            "Draft a brief overview: {context}",
            "Generate a summary: {context}",
        ]
    },
    "generate_response": {
        "category": "generation",
        "paraphrases": [
            "Write a response to this: {context}",
            "Draft a reply: {context}",
            "Help me respond to: {context}",
            "Compose a response: {context}",
        ]
    },
    "generate_description": {
        "category": "generation",
        "paraphrases": [
            "Write a product description for: {context}",
            "Create a description: {context}",
            "Draft a description: {context}",
            "Generate a product listing: {context}",
        ]
    },
    "generate_headline": {
        "category": "generation",
        "paraphrases": [
            "Write a headline for: {context}",
            "Create a catchy title: {context}",
            "Draft a headline: {context}",
            "Generate headlines: {context}",
        ]
    },
    "generate_code": {
        "category": "generation",
        "paraphrases": [
            "Write code for: {context}",
            "Generate a function that: {context}",
            "Code this: {context}",
            "Help me implement: {context}",
        ]
    },
    "generate_test": {
        "category": "generation",
        "paraphrases": [
            "Write tests for: {context}",
            "Generate unit tests: {context}",
            "Create test cases for: {context}",
            "Help me test: {context}",
        ]
    },
    "generate_docs": {
        "category": "generation",
        "paraphrases": [
            "Write documentation for: {context}",
            "Generate API docs: {context}",
            "Create documentation: {context}",
            "Document this code: {context}",
        ]
    },
    "generate_blog": {
        "category": "generation",
        "paraphrases": [
            "Write a blog post about: {context}",
            "Draft a blog article: {context}",
            "Create blog content: {context}",
            "Help me write a blog: {context}",
        ]
    },
    "generate_social": {
        "category": "generation",
        "paraphrases": [
            "Write a social media post about: {context}",
            "Create a tweet about: {context}",
            "Draft a LinkedIn post: {context}",
            "Generate social content: {context}",
        ]
    },
    "generate_presentation": {
        "category": "generation",
        "paraphrases": [
            "Create a presentation outline: {context}",
            "Write slide content for: {context}",
            "Draft presentation notes: {context}",
            "Help me build a deck: {context}",
        ]
    },
    "generate_report": {
        "category": "generation",
        "paraphrases": [
            "Write a report on: {context}",
            "Generate a status report: {context}",
            "Create a report: {context}",
            "Draft an analysis report: {context}",
        ]
    },
    "generate_proposal": {
        "category": "generation",
        "paraphrases": [
            "Write a proposal for: {context}",
            "Draft a business proposal: {context}",
            "Create a project proposal: {context}",
            "Help me write a proposal: {context}",
        ]
    },
    "generate_faq": {
        "category": "generation",
        "paraphrases": [
            "Generate FAQ for: {context}",
            "Write frequently asked questions: {context}",
            "Create an FAQ section: {context}",
            "Draft common questions: {context}",
        ]
    },
    "generate_tutorial": {
        "category": "generation",
        "paraphrases": [
            "Write a tutorial on: {context}",
            "Create a how-to guide: {context}",
            "Draft step-by-step instructions: {context}",
            "Generate a tutorial: {context}",
        ]
    },
    "generate_review": {
        "category": "generation",
        "paraphrases": [
            "Write a review of: {context}",
            "Generate a product review: {context}",
            "Create a review: {context}",
            "Draft a review: {context}",
        ]
    },
    "generate_pitch": {
        "category": "generation",
        "paraphrases": [
            "Write an elevator pitch for: {context}",
            "Create a sales pitch: {context}",
            "Draft a pitch: {context}",
            "Help me pitch: {context}",
        ]
    },
    "generate_bio": {
        "category": "generation",
        "paraphrases": [
            "Write a bio for: {context}",
            "Create a professional biography: {context}",
            "Draft a short bio: {context}",
            "Generate a bio: {context}",
        ]
    },
    "generate_newsletter": {
        "category": "generation",
        "paraphrases": [
            "Write a newsletter about: {context}",
            "Create newsletter content: {context}",
            "Draft a newsletter: {context}",
            "Generate newsletter copy: {context}",
        ]
    },
    "generate_script": {
        "category": "generation",
        "paraphrases": [
            "Write a script for: {context}",
            "Create a video script: {context}",
            "Draft a script: {context}",
            "Generate a script: {context}",
        ]
    },
    "generate_tagline": {
        "category": "generation",
        "paraphrases": [
            "Write a tagline for: {context}",
            "Create a slogan: {context}",
            "Draft a catchy tagline: {context}",
            "Generate taglines: {context}",
        ]
    },
    "generate_outline": {
        "category": "generation",
        "paraphrases": [
            "Create an outline for: {context}",
            "Write a content outline: {context}",
            "Draft an outline: {context}",
            "Generate an article outline: {context}",
        ]
    },
    "generate_checklist": {
        "category": "generation",
        "paraphrases": [
            "Create a checklist for: {context}",
            "Write a checklist: {context}",
            "Draft a task checklist: {context}",
            "Generate a checklist: {context}",
        ]
    },
    "generate_comparison": {
        "category": "generation",
        "paraphrases": [
            "Write a comparison of: {context}",
            "Create a comparison chart: {context}",
            "Compare these options: {context}",
            "Generate a comparison: {context}",
        ]
    },
    "generate_instructions": {
        "category": "generation",
        "paraphrases": [
            "Write instructions for: {context}",
            "Create setup instructions: {context}",
            "Draft installation guide: {context}",
            "Generate instructions: {context}",
        ]
    },
    "generate_announcement": {
        "category": "generation",
        "paraphrases": [
            "Write an announcement about: {context}",
            "Create a press release: {context}",
            "Draft an announcement: {context}",
            "Generate an announcement: {context}",
        ]
    },
    "generate_agenda": {
        "category": "generation",
        "paraphrases": [
            "Create a meeting agenda for: {context}",
            "Write an agenda: {context}",
            "Draft a meeting plan: {context}",
            "Generate an agenda: {context}",
        ]
    },
    "generate_talking_points": {
        "category": "generation",
        "paraphrases": [
            "Create talking points for: {context}",
            "Write key talking points: {context}",
            "Draft discussion points: {context}",
            "Generate talking points: {context}",
        ]
    },
}


def generate_query_dataset(n=10000, unique_ratio=0.30, seed=42):
    """Build a realistic query dataset with paraphrase clusters."""
    random.seed(seed)
    queries = []
    intents = list(INTENTS.keys())

    # 70% of queries come from paraphrase clusters
    n_clustered = int(n * (1 - unique_ratio))
    for _ in range(n_clustered):
        intent = random.choice(intents)
        phrase = random.choice(INTENTS[intent]["paraphrases"])
        queries.append({
            "text": phrase,
            "intent": intent,
            "category": INTENTS[intent]["category"],
        })

    # 30% are unique, one-off queries
    for i in range(n - n_clustered):
        queries.append({
            "text": f"Unique query about topic #{i}: explain in detail",
            "intent": f"unique_{i}",
            "category": random.choice(["qa", "summarization",
                                       "extraction", "classification",
                                       "generation"]),
        })

    random.shuffle(queries)
    return queries


if __name__ == "__main__":
    print("=== Query Dataset Generation ===\n")

    dataset = generate_query_dataset(10000)
    print(f"Total queries: {len(dataset)}")
    n_unique = len(set(q['intent'] for q in dataset))
    print(f"Unique intents: {n_unique}")
    print(f"Intent count: {len(INTENTS)} defined intents")

    # Category distribution
    categories = {}
    for q in dataset:
        categories[q["category"]] = categories.get(q["category"], 0) + 1
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Sample queries
    print("\nSample queries:")
    for q in dataset[:5]:
        print(f"  [{q['category']}] {q['text'][:60]}...")
