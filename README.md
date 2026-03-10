## Overview

This project uses an XGBoost model to predict a Fantasy Premier
League player's score for an upcoming gameweek. XGBoost was chosen
because it performs well on structured tabular data, handles noisy
and missing values naturally, and trains efficiently — all of which
suit the nature of FPL data.

## Data Pipeline

Player and fixture data is retrieved from the official FPL API.
Since fetching per-gameweek history for all 820 players takes
approximately two minutes, a local cache stores the raw API
responses (bootstrap, fixtures and full player history). On
subsequent runs the model checks whether a new gameweek has
finished — if not, it loads from cache in under a second. On a
cache miss it re-fetches from the API, retrains the model, and
overwrites the cache.

Once predictions are generated they are exported to an Amazon S3
bucket. A Windows Task Scheduler job runs the model daily at 12 pm,
automatically detecting new gameweeks and uploading fresh
predictions to S3.

## Chatbot Architecture

The user interacts with a Telegram bot. Each message is forwarded
by Telegram to an AWS API Gateway endpoint via a registered
webhook. API Gateway routes the request to an AWS Lambda function
which reads the latest gameweek predictions from S3 and sends them
alongside the user's question to the Claude API (Anthropic).
Claude generates a natural language answer which Lambda sends back
to the user via the Telegram Bot API. Conversation history is stored
in DynamoDB so the bot understands follow-up questions.
