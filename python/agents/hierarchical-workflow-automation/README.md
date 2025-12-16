# Cookie Delivery Agent System

A multi-agent system built with Google ADK that automates cookie delivery order processing, scheduling, and customer communication. The system integrates with BigQuery for order management, Google Calendar for delivery scheduling, and Gmail for customer notifications.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Root Agent    │───►│ Sequential Agent │───►│  Sub-Agents     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Database Agent  │    │  Calendar Agent  │    │   Email Agent   │
│ (BigQuery ADK)  │    │    MCP Server    │    │ (LangChain API) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    BigQuery     │    │ Google Calendar  │    │     Gmail       │
│   (GCP Acct)    │    │  (Business Acct) │    │ (Business Acct) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Agent Workflow

1. **Database Agent**: Fetches new orders from BigQuery using Google's first-party ADK toolset with status "order_placed"
2. **Calendar Agent**: Checks availability and schedules delivery appointments via MCP server
3. **Email Agent**: Generates personalized confirmation emails using LangChain Community Gmail toolkit and updates order status in BigQuery

## Setup Instructions

> ** Important for Argolis Users**: If you're using an Argolis corporate account, Gmail and Calendar access are restricted due to Google security policies. You'll need to create or use a **free secondary Gmail account** for Calendar and Gmail integration. While you can use Outlook or other services, these instructions are designed around Google Workspace/Gmail.

### 1. Get Code & Install

**Prerequisites:**
- Python 3.8+
- Google Cloud Project with BigQuery enabled
- Google ADK installed
- Secondary Gmail account (for Argolis users)

**Installation:**
```bash
cd cookie_scheduler_agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration (details in Environment Setup section below)
```

### 3. Global Authentication

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 4. Enable Calendar API

> ** For Argolis Users**: Use your **secondary Gmail account** for this step, not your corporate Argolis account.

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to "APIs & Services" → "Library"
3. Search for "Google Calendar API"
4. Click "Enable"
5. Go to "APIs & Services" → "OAuth consent screen"
6. Choose "External" user type (required for secondary accounts)
7. Add your secondary Gmail account as a test user

### 5. Enable Gmail API

> ** For Argolis Users**: Use your **secondary Gmail account** for this step, not your corporate Argolis account.

1. In Google Cloud Console, navigate to "APIs & Services" → "Library"
2. Search for "Gmail API"
3. Click "Enable"
4. Ensure your secondary Gmail account is added as a test user in OAuth consent screen

### 6. Enable BigQuery API

1. In Google Cloud Console, navigate to "APIs & Services" → "Library"
2. Search for "BigQuery API"
3. Click "Enable"
4. Verify your project has BigQuery permissions

### 7. Set Up Calendar Credentials

> ** For Argolis Users**: Create credentials using your **secondary Gmail account**, not your corporate account.

```bash
# Navigate to calendar MCP directory
cd mcp-servers/calendar/

# Create OAuth 2.0 credentials:
# 1. Go to Google Cloud Console → "APIs & Services" → "Credentials"
# 2. Click "Create Credentials" → "OAuth 2.0 Client ID"
# 3. Choose "Desktop Application"
# 4. Download the JSON file and save as 'calendar_credentials.json' in this directory
```

### 8. Set Up Gmail Credentials

> ** For Argolis Users**: Use the same **secondary Gmail account** credentials from the Calendar setup.

```bash
# Navigate to Gmail directory
cd gmail_langchain/

# Use the same OAuth 2.0 credentials from Calendar setup:
# 1. Copy the credentials JSON file from calendar setup
# 2. Save as 'gmail_credentials.json' in this directory
# OR create new credentials following the same process as Calendar
```

### 9. Setup BigQuery Environment

```bash
# Enable BigQuery Integration in .env file:
USE_BIGQUERY=true

# Optional: Create sample data for testing
python bigquery_utils/create_bigquery_environment.py
```

### 10. Validate BigQuery ADK

```bash
cd bigquery_utils/
python test_adk_bigquery_unit.py
python test_adk_integration.py

# Expected output:
# BigQuery ADK Test Suite Runner
# Unit Tests: PASSED (14/14 - 100.0%)
# Integration Tests: PASSED (9/9 - 100.0%)
# 🎉 ALL TESTS PASSED!
```

### 11. Run Calendar MCP

```bash
# Start the Calendar MCP server
cd mcp-servers/calendar/
python calendar_mcp_server.py

# In a separate terminal, test the server
python test_calendar_functions.py
```

### 12. Validate Calendar MCP

```bash
cd mcp-servers/calendar/
python test_calendar_functions.py

# Expected output:
# Calendar MCP: Successfully connected to Google Calendar
# Calendar events retrieved successfully
# Event creation and availability checking working
```

### 13. Run Main Agent

```bash
# From the main directory, start the agent system
adk web

# The system will automatically:
# - Use real Google Calendar if MCP configured
# - Use real Gmail if LangChain configured  
# - Use BigQuery ADK toolset for data management
# - Fall back to dummy data for missing services
```


## Environment Setup

Create a `.env` file in the main project directory with the following configuration:

> ** For Argolis Users**: Use your **secondary Gmail account email address** for `BUSINESS_EMAIL`, not your corporate Argolis email.

### Required Environment Variables

```bash
# =============================================================================
# GOOGLE CLOUD CONFIGURATION
# =============================================================================
# Your Google Cloud Project ID where BigQuery dataset will be created
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Model configuration for Google ADK
MODEL=gemini-2.5-flash

# =============================================================================
# GMAIL LANGCHAIN INTEGRATION
# =============================================================================
# Set to 'true' to use real Gmail via LangChain Community toolkit
USE_GMAIL_LANGCHAIN=true

# Business email address for sending customer communications
# For Argolis users: Use your secondary Gmail account, not corporate email
BUSINESS_EMAIL=your-secondary-email@gmail.com

# =============================================================================
# CALENDAR MCP INTEGRATION
# =============================================================================
# Set to 'true' to use real Google Calendar via MCP server
USE_CALENDAR_MCP=true

# Google Calendar ID for delivery scheduling
# For Argolis users with secondary accounts: Use 'primary' or specific calendar ID
# This should correspond to your secondary Gmail account's calendar
BUSINESS_CALENDAR_ID=primary

# =============================================================================
# BIGQUERY ADK INTEGRATION
# =============================================================================
# Set to 'true' to use Google's first-party ADK BigQuery toolset
# Set to 'false' to use dummy data for development/testing
USE_BIGQUERY=true

# =============================================================================
# BUSINESS ACCOUNT CONFIGURATION
# =============================================================================
# Additional business configuration for enhanced features
# Business phone number for delivery coordination
BUSINESS_PHONE=+1-555-0199

# =============================================================================
# DEVELOPMENT/TESTING
# =============================================================================
# Set to 'development' to use dummy data instead of real services
ENVIRONMENT=production

# Logging level
LOG_LEVEL=INFO
```

### Gmail LangChain Integration Setup

> ** Important for Argolis Users**: Corporate Gmail access is restricted. You **must** use a free secondary Gmail account for this integration. The instructions below are designed for Google Workspace/Gmail, though you can adapt them for Outlook or other providers.

The Gmail integration uses **LangChain Community Gmail toolkit** for complete Gmail API functionality. Here's what's available:

#### Features:
- Gmail API authentication via OAuth2 with automatic token refresh
- Email sending with HTML and plain text support
- Message search with powerful Gmail query syntax
- Message retrieval and thread management
- Graceful fallback to dummy data when not configured
- Comprehensive error handling and logging

#### Setup Steps:
1. **Enable Gmail API** in Google Cloud Console (using secondary account for Argolis users)
2. **Install LangChain Community**: `pip install langchain-community`
3. **Create OAuth2 Credentials** (Desktop Application) - must be associated with secondary account
4. **Save credentials** as `gmail_langchain/gmail_credentials.json`
5. **Test the integration**: `python gmail_langchain/test_gmail_integration.py`

#### File Structure:
```
gmail_langchain/
├── gmail_manager.py             # Main LangChain Gmail manager class
├── email_utils.py               # Utility functions for agent integration
├── test_gmail_integration.py    # Test script
├── gmail_credentials.json       # Your OAuth2 credentials
├── gmail_token.json             # Auto-generated tokens
└── README.md                    # Setup documentation
```

### Calendar MCP Server Setup

> ** Important for Argolis Users**: Corporate Calendar access is restricted. You **must** use a free secondary Gmail account's calendar for this integration. These instructions are designed for Google Calendar, though you can adapt them for Outlook or other calendar providers.

The Calendar MCP server provides Google Calendar integration with the following features:

#### Features:
- Google Calendar API authentication via OAuth2
- Event creation, reading, and availability checking
- RFC3339 datetime formatting for Google Calendar
- Automatic fallback to dummy data if unavailable
- Comprehensive error handling and logging

#### Setup Steps:
1. **Enable Calendar API** in Google Cloud Console (using secondary account for Argolis users)
2. **Create OAuth2 Credentials** (Desktop Application) - must be associated with secondary account
3. **Save credentials** as `mcp-servers/calendar/calendar_credentials.json`
4. **Test the integration**: `python mcp-servers/calendar/test_calendar_functions.py`

#### File Structure:
```
mcp-servers/calendar/
├── calendar_mcp_server.py       # Main MCP server (CalendarManager class)
├── calendar_credentials.json    # Your OAuth2 credentials
├── calendar_token.json          # Auto-generated tokens
└── test_calendar_functions.py   # Test script
```

## BigQuery Schema

The system creates the following BigQuery structure:

### Dataset: `cookie_delivery`
### Table: `orders`

```sql
CREATE TABLE `{PROJECT_ID}.cookie_delivery.orders` (
  order_id STRING NOT NULL,
  order_number STRING NOT NULL,
  customer_email STRING NOT NULL,
  customer_name STRING NOT NULL,
  customer_phone STRING,
  order_items ARRAY<STRUCT<
    item_name STRING,
    quantity INT64,
    unit_price FLOAT64
  >>,
  delivery_address STRUCT<
    street STRING,
    city STRING,
    state STRING,
    zip_code STRING,
    country STRING
  >,
  delivery_location STRING,
  delivery_request_date DATE,
  delivery_time_preference STRING,  -- 'morning', 'afternoon', 'evening'
  order_status STRING NOT NULL,     -- 'order_placed', 'confirmed', 'scheduled', 'delivered'
  total_amount FLOAT64,
  order_date TIMESTAMP,
  special_instructions STRING,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

### Sample Data Insert

```sql
INSERT INTO `{PROJECT_ID}.cookie_delivery.orders` VALUES (
  'ORD12345',
  'ORD12345',
  'customer@example.com',
  'John Doe',
  '+1-555-0123',
  [
    STRUCT('Chocolate Chip', 12, 2.50),
    STRUCT('Oatmeal Raisin', 6, 2.75)
  ],
  STRUCT('123 Main St', 'Anytown', 'CA', '12345', 'USA'),
  '123 Main St, Anytown, CA 12345, USA',
  '2025-09-10',
  'morning',
  'order_placed',
  63.50,
  '2025-09-04T10:30:00Z',
  'Please ring doorbell twice',
  CURRENT_TIMESTAMP(),
  CURRENT_TIMESTAMP()
);
```

## MCP Server and LangChain Integration Setup

The system uses both Model Context Protocol (MCP) servers and LangChain Community tools for external service integration.

### Running Services

**Calendar MCP Server:**
```bash
python calendar_mcp_server.py
```

**Gmail LangChain Integration:**
```bash
# Gmail runs directly within the agent using LangChain Community toolkit
# No separate server needed - OAuth2 authentication handled automatically
python gmail_langchain/test_gmail_integration.py  # Test integration
```

### Integration Features

#### Calendar MCP Server
- `get_events`: Fetch delivery schedule
- `create_event`: Schedule new deliveries  
- `check_availability`: Verify time slot availability
- `update_event`: Modify existing appointments

#### Gmail LangChain Integration
- `send_email`: Send customer confirmation emails with HTML formatting
- `search_messages`: Search Gmail with powerful query syntax
- `get_message`: Retrieve specific emails and thread details
- `oauth2_authentication`: Automatic token refresh and credential management

## Current Implementation

### BigQuery ADK Integration
- **Google's First-Party ADK Toolset**: Uses official BigQuery ADK integration
- **Application Default Credentials**: Secure authentication via ADC
- **WriteMode Configuration**: Proper data access control (BLOCKED, ALLOWED, PROTECTED)
- **Async Compatibility**: Resolved async conflicts for ADK web interface usage
- **Available Tools**: list_dataset_ids, get_dataset_info, list_table_ids, get_table_info, execute_sql, ask_data_insights

### Gmail LangChain Integration
- **LangChain Community Gmail Toolkit**: Uses official LangChain integration for Gmail API
- **OAuth2 Authentication**: Secure authentication with automatic token refresh
- **HTML Email Support**: Rich formatting for professional customer communications
- **Message Search & Retrieval**: Full Gmail query capabilities for business operations
- **Graceful Fallback**: Uses dummy data when Gmail not configured

### Calendar Agent with Real Google Calendar
- **Real Google Calendar Integration**: Creates actual calendar events via MCP server
- **Smart Fallback**: Uses dummy data when Calendar MCP unavailable
- **Business Calendar Support**: Configurable calendar ID for business account
- **RFC3339 Datetime**: Proper timezone handling for Google Calendar API

### Agent Workflow (Sequential Processing)
1. **Database Agent**: Fetches orders using BigQuery ADK toolset with production-ready data access
2. **Calendar Agent**: Real Google Calendar scheduling via MCP server
3. **Email Agent**: Professional email communications via LangChain Gmail toolkit with BigQuery integration for order updates
4. **Haiku Writer Sub-Agent**: Generates creative seasonal content

### Error Handling & Resilience
- **Graceful Degradation**: Falls back to dummy data when services unavailable
- **Comprehensive Logging**: Detailed operation tracking and error reporting
- **Authentication Recovery**: Handles OAuth2 token refresh automatically for both Calendar and Gmail
- **Service Availability Checks**: Smart detection of configured vs. fallback services

## Workflow Process

1. **Order Detection**: Database agent fetches latest order with "order_placed" status
2. **Schedule Analysis**: Calendar agent checks availability for requested delivery date
3. **Appointment Creation**: Calendar agent schedules delivery appointment
4. **Haiku Generation**: Email agent delegates to haiku writer for personalized content
5. **Customer Notification**: Email agent sends confirmation with delivery details and haiku (drafts email but not implemented to send right now)
6. **Status Update**: Order status updated to "scheduled" in BigQuery

## Testing & Validation

### BigQuery ADK Testing
The system includes a comprehensive test suite for the BigQuery ADK integration:

```bash
# Unit Tests - Test application logic and SQL query generation
cd cookie_scheduler_agent/bigquery_utils/
python test_adk_bigquery_unit.py

# Integration Tests - Test ADK toolset integration patterns
python test_adk_integration.py

# Run All Tests - Comprehensive test suite runner
python run_all_tests.py
```

**Test Coverage:**
-  ADK toolset initialization and configuration
-  SQL query generation logic for business operations
-  Parameter validation and error handling
-  Mock agent workflow integration
-  Authentication and credential management
-  Performance and scaling characteristics

**Expected Test Results:**
```
BigQuery ADK Test Suite Runner
=====================================
Unit Tests (test_adk_bigquery_unit.py): PASSED
  Total: 14
  Passed: 14
  Success Rate: 100.0%

Integration Tests (test_adk_integration.py): PASSED
  Total: 9
  Passed: 9
  Success Rate: 100.0%

🎉 ALL TESTS PASSED! BigQuery ADK integration is working correctly.
Test Quality: EXCELLENT
```

### Calendar MCP Testing
```bash
# Test real Google Calendar integration
cd mcp-servers/calendar/
python test_calendar_functions.py

# Expected output:
# Calendar MCP: Successfully connected to Google Calendar
# Calendar events retrieved successfully
# Event creation and availability checking working
```

### Agent Integration Testing
```bash
# Test agent with real calendar integration
python agent.py

# The agent will:
# 1. Import CalendarManager successfully
# 2. Use real Google Calendar if configured
# 3. Fall back to dummy data gracefully
# 4. Process sequential workflow
```

### Testing Architecture

The testing strategy follows best practices for first-party ADK integration:

**What We Test (Application Logic):**
-  Tool configuration and initialization
-  SQL query generation logic
-  Parameter validation and input handling
-  Agent integration patterns
-  Error handling for application-specific scenarios
-  Mock workflow simulations

**What We DON'T Test (ADK Handles):**
- BigQuery connection logic (ADK manages this)
- Authentication mechanisms (Google Cloud SDK handles this)
- Query execution engine (BigQuery service responsibility)
- Retry logic and backoff strategies (ADK implements this)

For detailed testing documentation, see `cookie_scheduler_agent/bigquery_utils/TESTING_STRATEGY.md`.

## File Structure

```
gcp_cookie_delivery_agent/
├── .env.example              # Example environment configuration
├── .env                      # Your environment configuration (create this)
├── requirements.txt          # Python dependencies
├── setup.sh                  # enables GCP APIs and permissions to service accounts
├── deploy_agent.py           # Agent Engine deployment script
└── cookie_scheduler_agent/
    ├── agent.py                    # main adk agent orchestration
    ├── dummy_data.py              # Fallback data for testing
    │
    ├── gmail_langchain/
    │   ├── gmail_manager.py             # Main LangChain Gmail manager class
    │   ├── email_utils.py               # Utility functions for agent integration
    │   ├── test_gmail_integration.py    # Test script
    │   ├── gmail_credentials.json       # Your OAuth2 credentials
    │   ├── gmail_token.json             # Auto-generated tokens
    │   └── README.md                    # Setup documentation
    │
    ├── bigquery_utils/           # BigQuery ADK toolset integration
    │   ├── bigquery_tools.py     # ADK BigQuery toolset implementation
    │   ├── create_bigquery_environment.py # BigQuery setup script
    │   ├── test_adk_bigquery_unit.py      # Comprehensive unit tests
    │   ├── test_adk_integration.py        # Integration tests with ADK
    │   ├── run_all_tests.py               # Test suite runner
    │   ├── TESTING_STRATEGY.md            # Testing documentation
    │   ├── CLEANUP_SUMMARY.md             # Legacy code cleanup notes
    │   ├── BIGQUERY_SETUP.md              # ADK setup guide
    │   └── README.md                      # Directory documentation
    │
    ├── mcp-servers/              # MCP Server implementations
    │   ├── calendar/             # Calendar MCP
    │   │   ├── calendar_mcp_server.py      # Complete CalendarManager class
    │   │   ├── calendar_credentials.json   # OAuth2 credentials (you create)
    │   │   ├── calendar_token.json         # Auto-generated tokens
    │   │   └── test_calendar_functions.py  # Test script for validation
    │   ├── start_calendar_mcp.py           # MCP server startup script
    │   └── setup_calendar_credentials.md   # Setup instructions
```

## Security Notes

### Credential Management
- Never commit `.env`, `*_credentials.json`, or `*_token.json` files to version control
- Use Google Secret Manager for production deployments
- Implement credential rotation policies

### API Permissions
- Use minimal required scopes for OAuth2
- Implement proper IAM roles for BigQuery access
- Monitor API usage and set quotas

### Data Protection
- All customer data is encrypted at rest in BigQuery
- Use HTTPS for all API communications
- Implement audit logging for data access

## Troubleshooting

### Calendar MCP Issues (Most Common)

1. **Import Error: "calendar_mcp_server could not be resolved"**
   ```bash
   # Solution: This is an IDE issue, the code works at runtime
   # The agent uses try/catch for graceful fallback
   # Verify it works: python mcp-servers/calendar/test_calendar_functions.py
   ```

2. **OAuth2 Authentication Failed**
   ```bash
   # 1. Ensure Calendar API is enabled in Google Cloud Console
   # 2. Create OAuth 2.0 Client ID (Desktop Application)  
   # 3. Download and save as mcp-servers/calendar/calendar_credentials.json
   # 4. Delete calendar_token.json to force re-authentication
   # 5. For Argolis users: Ensure you're using secondary Gmail account, not corporate
   ```

3. **Calendar Events Not Appearing**
   ```bash
   # Check your calendar ID in .env:
   BUSINESS_CALENDAR_ID=primary  # or specific calendar ID
   # Verify permissions on the target calendar
   ```

4. **Permissions Error**
   ```bash
   # 1. Ensure Calendar API is enabled in Google Cloud Console
   # 2. Navigate to: "APIs & Services" → "OAuth consent screen"
   # 3. User Type: Make sure you selected "External" (not Internal)
   # 4. Test users: Add your secondary Gmail account as a test user
   # 5. For Argolis users: Corporate accounts won't work - must use secondary account
   ```


### BigQuery ADK Issues

1. **BigQuery ADK Toolset Import Error**
   ```bash
   # Ensure google-adk package is installed with BigQuery support
   pip install google-adk[bigquery]
   
   # Verify authentication
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **BigQuery Permission Denied**
   ```bash
   # Ensure your account has BigQuery permissions
   # Required roles: BigQuery Data Editor, BigQuery Job User
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="user:your-email@domain.com" \
     --role="roles/bigquery.dataEditor"
   ```

3. **Async Compatibility Issues**
   ```bash
   # This should be resolved with the ADK toolset integration
   # If you encounter async errors, run the integration test:
   python test_bigquery_integration.py
   ```

4. **403 Errors**
   ```bash
   # if working with multiple projects, you may need to ensure you are in the correct project
   gcloud auth application-default login
   gcloud config set project <PROJECT_ID>
   gcloud config set billing/quota_project <PROJECT_ID>
   ```

### Debug Mode

Enable detailed logging:

```bash
# In .env file
LOG_LEVEL=DEBUG
```

## Monitoring and Analytics

The system includes built-in analytics via the `get_order_analytics` function:

```python
# Get business insights
analytics = await get_order_analytics(tool_context, days=30)
print(analytics)
# Returns: order counts, average order value, total revenue by status
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the detailed [BQ_README.md](./cookie_scheduler_agent/bigquery_utils/README.md), [GMAIL_README.md](./cookie_scheduler_agent/gmail_langchain/README.md), [CALENDAR_README.md](./cookie_scheduler_agent/mcp_servers/README.md) for more detailed setup instructions and troubleshooting
3. Create an issue in the repository

---

**Note**: This system demonstrates production-ready integration patterns with Google Cloud services. Always test thoroughly in a development environment before deploying to production.
